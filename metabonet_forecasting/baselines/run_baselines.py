"""Stage B baselines, on exactly the same windows as the neural models.

    python -m metabonet_forecasting.baselines.run_baselines \
        --config configs/experiment/e1_baselines.yaml --model persistence

Four of them, in increasing order of what they are allowed to know:

``persistence``    the forecast is the last CGM reading. The floor. Anything that
                   cannot beat it at 120 minutes has learned nothing.
``linear_trend``   least-squares extrapolation of the last 30 minutes, clipped to
                   the sensor's reporting range. Catches "the model only learned
                   to extend the current slope".
``ridge``          linear regression on lagged CGM plus (optionally) dose and carb
                   window sums. The strongest thing that is still a linear model.
``gbdt``           gradient-boosted trees on the same features. The strongest
                   thing that is not a sequence model, and historically hard to
                   beat on tabularised glucose forecasting.

They emit the same ``predictions_test.parquet`` schema as neural runs, so the
whole evaluation stack -- strata, bootstrap, plots, aggregation -- applies without
a special case.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..common.config import REPO_ROOT, load_config, save_config
from ..common.logging import RunLogger, provenance
from ..common.seeding import seed_everything
from ..data import constants as K
from ..data.dataset import build_data
from ..data.windows import WindowDataset
from ..data.splits import seen_in_train

MODELS = ("persistence", "linear_trend", "ridge", "gbdt")

# Lag taps in lattice steps: dense over the last hour, thinning out to 24 h.
LAGS = (0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 18, 24, 36, 48, 72, 96, 144, 287)
WINDOWS = (6, 12, 24, 48, 72, 288)


def _gather(ds: WindowDataset, batch: int = 8192
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Raw (unnormalised) CGM/insulin/carb windows for the tabular baselines.

    Reads the store directly rather than going through ``__getitem__``: these
    models want physical units and no normalisation, and pulling whole blocks is
    an order of magnitude faster than per-item collation.

    The fourth return value is the planned-action block from *after* the anchor,
    or None when the run is causal. It exists so the tabular baselines can enter
    the same comparison: a gradient-boosted tree given future carbohydrate and
    insulin sums is precisely the recipe behind an implausible leaderboard score,
    and it is more convincing to reproduce that than to argue about it.
    """
    store = ds.store
    rows = ds.rows[ds.indices]
    L = ds.spec.context
    ci = store.channel_index
    want = [ci["cgm"], ci["cgm_mask"], ci["ins_bolus"], ci["ins_basal"], ci["carb_grams"]]
    out = np.empty((rows.size, L, len(want)), dtype=np.float32)

    n_f = ds.n_future
    fut = np.zeros((rows.size, n_f, len(want)), dtype=np.float32) if n_f else None
    seg_end = ds.seg_end if n_f else None

    for start in range(0, rows.size, batch):
        chunk = rows[start:start + batch]
        for i, row in enumerate(chunk):
            out[start + i] = store.channels[row - L + 1: row + 1][:, want]
    if n_f:
        for i, (row, ai) in enumerate(zip(rows, ds.indices)):
            usable = int(min(n_f, max(int(seg_end[ai]) - row - 1, 0)))
            if usable > 0:
                fut[i, :usable] = store.channels[row + 1: row + 1 + usable][:, want]
    y = np.asarray(store.targets[ds.indices][:, ds.horizon_ids], dtype=np.float32)
    return out, y, ds.indices, fut


def _future_features(fut: Optional[np.ndarray], modalities: Sequence[str]) -> List[np.ndarray]:
    """Planned-action summaries: cumulative dose/carbs over the forecast span.

    Deliberately the crudest possible encoding -- running sums at several
    horizons -- because the point is to show how little effort it takes to buy an
    implausible score once post-anchor rows are in reach.
    """
    if fut is None or not modalities:
        return []
    steps = fut.shape[1]
    cuts = [c for c in (6, 12, 18, 24) if c <= steps] or [steps]
    feats: List[np.ndarray] = []
    channels: List[int] = []
    if "insulin" in modalities:
        channels += [2, 3]                       # bolus, basal
    if "carbs" in modalities:
        channels += [4]                          # carbs
    for ch in channels:
        for c in cuts:
            feats.append(np.nansum(fut[:, :c, ch], axis=1))
        # time to the first event inside the span, capped -- "a meal is coming
        # soon" is different information from "a meal is coming eventually"
        hit = fut[:, :, ch] > 0
        first = np.where(hit.any(axis=1), hit.argmax(axis=1), steps).astype(np.float32)
        feats.append(first)
    return feats


def _features(block: np.ndarray, model: str,
              fut: Optional[np.ndarray] = None,
              future_modalities: Sequence[str] = ()) -> np.ndarray:
    """Lagged CGM (+ optional dose/carb sums) as a design matrix."""
    L = block.shape[1]
    cgm = block[..., 0]
    feats: List[np.ndarray] = []
    for lag in LAGS:
        if lag < L:
            feats.append(cgm[:, L - 1 - lag])
    now = cgm[:, -1]
    for k in (1, 2, 3, 6, 12, 24):
        if k < L:
            feats.append(now - cgm[:, L - 1 - k])
    for w in WINDOWS:
        if w <= L:
            win = cgm[:, L - w:]
            feats.extend([np.nanmean(win, axis=1), np.nanstd(win, axis=1),
                          np.nanmin(win, axis=1), np.nanmax(win, axis=1)])
    if model in ("ridge", "gbdt"):
        for ch in (2, 3, 4):                     # bolus, basal, carbs
            for w in WINDOWS:
                if w <= L:
                    feats.append(np.nansum(block[:, L - w:, ch], axis=1))
    feats.extend(_future_features(fut, future_modalities))
    X = np.stack(feats, axis=1).astype(np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def _fit_predict(model: str, Xtr, ytr, Xte, seed: int, log) -> np.ndarray:
    """One estimator per horizon; returns [n_test, n_horizons]."""
    n_h = ytr.shape[1]
    preds = np.full((Xte.shape[0], n_h), np.nan, dtype=np.float32)
    for j in range(n_h):
        ok = np.isfinite(ytr[:, j])
        if ok.sum() < 100:
            log.info(f"  horizon {j}: only {int(ok.sum())} labels, skipping")
            continue
        t0 = time.time()
        if model == "ridge":
            from sklearn.linear_model import Ridge
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import make_pipeline
            est = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=None))
        elif model == "gbdt":
            from sklearn.ensemble import HistGradientBoostingRegressor
            est = HistGradientBoostingRegressor(
                max_iter=400, learning_rate=0.06, max_leaf_nodes=63,
                early_stopping=True, validation_fraction=0.1, random_state=seed)
        else:
            raise ValueError(model)
        est.fit(Xtr[ok], ytr[ok, j])
        preds[:, j] = est.predict(Xte)
        log.info(f"  horizon index {j}: fitted on {int(ok.sum()):,} rows "
                 f"in {time.time() - t0:.0f}s")
    return preds


def run(cfg, model: str, run_dir: Path, seed: int = 0) -> Path:
    log = RunLogger(run_dir)
    log.info(f"baseline {model} -> {run_dir}")
    bundle = build_data(cfg)
    horizons = list(bundle.spec.horizons_min)

    future_modalities = list(bundle.spec.future_modalities)
    if future_modalities:
        log.info(f"planned future actions ENABLED: {future_modalities} over "
                 f"{bundle.spec.n_future_steps()} steps -- this is "
                 f"intervention-conditioned prediction, not forecasting")

    test_block, y_test, test_idx, test_future = _gather(bundle.datasets["test"])
    cgm_now = test_block[:, -1, 0]

    if model == "persistence":
        pred = np.repeat(cgm_now[:, None], len(horizons), axis=1)
    elif model == "linear_trend":
        # Least-squares slope over the last 30 minutes (6 lattice steps), then
        # extrapolated and clipped to what a sensor can report. Unclipped linear
        # extrapolation produces negative glucose at 120 minutes.
        k = min(6, test_block.shape[1])
        recent = test_block[:, -k:, 0]
        t = np.arange(k, dtype=np.float32) * K.GRID_MIN
        t = t - t.mean()
        slope = (recent - recent.mean(axis=1, keepdims=True)) @ t / (t ** 2).sum()
        pred = cgm_now[:, None] + slope[:, None] * np.array(horizons, dtype=np.float32)[None, :]
        pred = np.clip(pred, K.CGM_REPORT_LO, K.CGM_REPORT_HI)
    else:
        train_block, y_train, _, train_future = _gather(bundle.datasets["train"])
        Xtr = _features(train_block, model, train_future, future_modalities)
        Xte = _features(test_block, model, test_future, future_modalities)
        log.info(f"design matrix: train {Xtr.shape}, test {Xte.shape}")
        pred = _fit_predict(model, Xtr, y_train, Xte, seed, log)

    anchors = bundle.store.anchors
    df = pd.DataFrame({"anchor": test_idx})
    for col in ("subject", "cohort", "is_test", "known_patient", "cgm_now",
                "cgm_delta_30min", "min_since_carb", "min_since_bolus", "tod_min"):
        df[col] = anchors[col].to_numpy()[test_idx]
    df["seen_in_train"] = seen_in_train(bundle.store, bundle.split, test_idx)
    for j, h in enumerate(horizons):
        valid = np.isfinite(y_test[:, j])
        df[f"y_true_{h}"] = np.where(valid, y_test[:, j], np.nan)
        df[f"y_pred_{h}"] = np.where(valid, pred[:, j], np.nan)

    run_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(run_dir / "predictions_test.parquet", index=False)
    save_config(cfg, run_dir / "config.resolved.yaml")
    log.write_json("manifest.json", {
        "run_name": run_dir.name,
        "config": cfg.to_dict(),
        "data": bundle.summary(),
        "model": {"class": model, "backbone": model, "fusion": "tabular",
                  "future_covariates": bool(future_modalities),
                  "parameters": None, "receptive_field": bundle.spec.context},
        "provenance": provenance(),
    })

    from ..evaluation.evaluate import score_run
    metrics = score_run(run_dir, bundle=bundle)
    for h in horizons:
        o = metrics["strata"]["all"][str(h)]["overall"]
        log.info(f"{model} h={h:>3}min  RMSE {o['rmse']:6.2f}  MARD {o['mard']:5.2f}%  "
                 f"DTS-A {o['dts_A']:5.1f}%")
    log.close()
    return run_dir


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], dest="sets")
    ap.add_argument("--model", required=True, choices=MODELS + ("all",))
    ap.add_argument("--name", default=None)
    ap.add_argument("--runs-dir", default=str(REPO_ROOT / "runs"))
    a = ap.parse_args(argv)

    cfg = load_config(a.config, a.sets)
    seed = int(cfg.get("seed", 0))
    seed_everything(seed)
    models = MODELS if a.model == "all" else (a.model,)
    for model in models:
        name = a.name if (a.name and len(models) == 1) else \
            f"baseline_{model}_ctx{cfg.get_path('data.context', 288)}_s{seed}"
        run(cfg, model, Path(a.runs_dir) / name, seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
