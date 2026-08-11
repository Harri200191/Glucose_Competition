"""Run one experiment: (model, horizon, split) -> metrics + artefacts.

    python -m exp.train --data build/main --model xgboost --horizon 30

Writes to `runs/<tag>/`:
    result.json      metrics on test, on the leaderboard-approx subset, and by subject
    predictions.npy  float32 test predictions, aligned to the test mask
    model.joblib     the fitted estimator (skipped for the neural nets by default)
    importance.csv   feature importances where the learner exposes them
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .build_dataset import load_dataset
from .config import BUILD_DIR, HORIZONS, RUNS_DIR
from .metrics import evaluate, evaluate_by_subject
from .models import FAMILY, MODEL_NAMES, SEQUENCE_MODELS, build_model
from .models.base import FitContext
from .splits import describe_split, leaderboard_mask, make_split


def _subsample(mask: np.ndarray, cap: int | None, seed: int) -> np.ndarray:
    """Cap the number of True entries in a boolean mask, uniformly at random."""
    if cap is None or mask.sum() <= cap:
        return mask
    idx = np.flatnonzero(mask)
    rng = np.random.default_rng(seed)
    keep = rng.choice(idx, cap, replace=False)
    out = np.zeros_like(mask)
    out[keep] = True
    return out


def run(data_dir: Path, model_name: str, horizon: int, split_scheme: str = "official",
        fold: int = 0, holdout_cohort: str | None = None, gpu: int | None = 0,
        max_train: int | None = None, max_eval: int | None = None,
        seed: int = 0, out_dir: Path | None = None, n_jobs: int = -1,
        save_model: bool = True, model_kw: dict | None = None) -> dict:
    t_start = time.time()
    X, Y, meta, manifest = load_dataset(data_dir, mmap=True)
    names = manifest["feature_names"]
    cat_cols = manifest["categorical_columns"]
    cat_idx = [names.index(c) for c in cat_cols]
    h_index = list(HORIZONS).index(horizon)
    y_all = Y[:, h_index].astype(np.float32)

    # ------------------------------------------------------------------ split
    split = make_split(meta, split_scheme, fold=fold, holdout_cohort=holdout_cohort,
                       seed=seed)
    has_y = np.isfinite(y_all)
    tr = split["train"] & has_y
    va = split["val"] & has_y
    te = split["test"] & has_y

    tr = _subsample(tr, max_train, seed)
    te_eval = _subsample(te, max_eval, seed + 1)

    desc = describe_split(meta, {"train": tr, "val": va, "test": te_eval})
    print(desc.to_string(), flush=True)
    overlap_tv = desc.attrs["train_val_subject_overlap"]
    overlap_tt = desc.attrs["train_test_subject_overlap"]
    if split_scheme != "official" and (overlap_tv or overlap_tt):
        raise RuntimeError(f"subject leakage: train/val={overlap_tv} train/test={overlap_tt}")
    if overlap_tv:
        raise RuntimeError(f"train/val share {overlap_tv} subjects -- validation is leaky")

    tr_idx, va_idx, te_idx = np.flatnonzero(tr), np.flatnonzero(va), np.flatnonzero(te_eval)
    print(f"[data] train={len(tr_idx):,} val={len(va_idx):,} test={len(te_idx):,} "
          f"features={X.shape[1]} horizon={horizon}min", flush=True)

    Xtr = np.asarray(X[tr_idx]); ytr = y_all[tr_idx]
    Xva = np.asarray(X[va_idx]); yva = y_all[va_idx]
    Xte = np.asarray(X[te_idx]); yte = y_all[te_idx]

    # ------------------------------------------------------------------ model
    model = build_model(model_name, **(model_kw or {}))
    ctx = FitContext(feature_names=names, categorical_idx=cat_idx,
                     categorical_levels=manifest["categorical_levels"],
                     horizon_min=horizon, seed=seed, gpu=gpu, n_jobs=n_jobs)

    if model_name in SEQUENCE_MODELS:
        ch_path = data_dir / "channels.npy"
        if not ch_path.exists():
            raise FileNotFoundError(
                f"{model_name} needs channels.npy -- rebuild without --no-channels")
        channels = np.load(ch_path, mmap_mode="r")
        rows = meta["row_idx"].to_numpy()
        model.attach_sequences(channels, rows[tr_idx], rows[va_idx])

    t0 = time.time()
    model.fit(Xtr, ytr, Xva if len(va_idx) else None, yva if len(va_idx) else None, ctx)
    fit_s = time.time() - t0

    # TabPFN caps its own evaluation size -- honour it and say so.
    eval_sub = getattr(model, "subsample_eval", lambda n, seed=0: None)(len(te_idx), seed)
    if eval_sub is not None:
        te_idx = te_idx[eval_sub]
        Xte, yte = Xte[eval_sub], yte[eval_sub]
        print(f"[eval] TabPFN: scoring a {len(te_idx):,}-row subsample", flush=True)

    t0 = time.time()
    if model_name in SEQUENCE_MODELS:
        pred = model.predict(Xte, rows=meta["row_idx"].to_numpy()[te_idx])
    else:
        pred = model.predict(Xte)
    predict_s = time.time() - t0

    # ------------------------------------------------------------------ score
    result = {
        "model": model_name, "family": FAMILY.get(model_name, "?"),
        "horizon_min": horizon, "split": split_scheme, "fold": fold,
        "holdout_cohort": holdout_cohort, "seed": seed,
        "n_train": int(len(tr_idx)), "n_val": int(len(va_idx)),
        "n_test": int(len(te_idx)),
        "n_features": int(X.shape[1]),
        "fit_seconds": round(fit_s, 1), "predict_seconds": round(predict_s, 1),
        "total_seconds": round(time.time() - t_start, 1),
        "model_notes": model.notes,
        "data_manifest": {k: manifest[k] for k in
                          ("built_at", "n_samples", "n_features", "n_subjects",
                           "n_cohorts", "config")},
        "test": evaluate(yte, pred),
        "test_by_subject": evaluate_by_subject(yte, pred,
                                               meta["subject"].to_numpy()[te_idx]),
    }

    # Score the leaderboard-approx slice separately (EDA 13: the official scored
    # subset excludes four cohorts, so the headline number differs).
    lb = leaderboard_mask(meta)[te_idx]
    if lb.any():
        result["test_leaderboard_subset"] = evaluate(yte[lb], pred[lb])
        result["test_leaderboard_subset"]["n_cohorts"] = int(
            pd.unique(meta["source_file"].to_numpy()[te_idx][lb]).size)

    # Per-cohort breakdown -- where does the model actually fail?
    coh = meta["source_file"].to_numpy()[te_idx]
    per_cohort = {}
    for c in pd.unique(coh):
        m = coh == c
        if m.sum() >= 500:
            per_cohort[str(c)] = {k: round(v, 4) for k, v in
                                  evaluate(yte[m], pred[m]).items()
                                  if k in ("n", "MARD", "RMSE", "clarke_A", "parkes_A")}
    result["test_by_cohort"] = per_cohort

    # ------------------------------------------------------------------ write
    tag = out_dir or (RUNS_DIR / f"{model_name}_h{horizon}_{split_scheme}"
                      f"{'_' + holdout_cohort if holdout_cohort else ''}"
                      f"{'_f' + str(fold) if split_scheme == 'subject' else ''}")
    tag = Path(tag); tag.mkdir(parents=True, exist_ok=True)
    (tag / "result.json").write_text(json.dumps(result, indent=2, default=str))
    np.save(tag / "predictions.npy", pred)
    np.save(tag / "test_index.npy", te_idx)

    imp = model.feature_importance()
    if imp is not None and len(imp) == len(names):
        (pd.DataFrame({"feature": names, "importance": imp})
           .sort_values("importance", ascending=False)
           .to_csv(tag / "importance.csv", index=False))
    if save_model:
        # Neural nets write a torch checkpoint (their module classes are defined
        # inside a factory and are not picklable by reference); everything else
        # joblib-pickles. `exp.predict_submission` loads either.
        path = tag / ("model.pt" if FAMILY.get(model_name) == "nn" else "model.joblib")
        try:
            model.save(path)
            print(f"[saved] {path} ({path.stat().st_size/1e6:.1f} MB)", flush=True)
        except Exception as e:                       # noqa: BLE001
            print(f"[warn] could not save model: {e}", flush=True)

    t = result["test"]
    print(f"\n[{model_name} h={horizon} {split_scheme}]  "
          f"MARD {t['MARD']:.3f}%  RMSE {t['RMSE']:.2f}  "
          f"Clarke-A {t['clarke_A']:.2f}%  Parkes-A {t['parkes_A']:.2f}%  "
          f"({fit_s:.0f}s fit)  -> {tag}", flush=True)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=BUILD_DIR / "main")
    p.add_argument("--model", required=True, choices=MODEL_NAMES)
    p.add_argument("--horizon", type=int, default=30, choices=list(HORIZONS))
    p.add_argument("--split", default="official", choices=["official", "subject", "cohort"])
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--holdout-cohort", default=None)
    p.add_argument("--gpu", type=int, default=0, help="-1 for CPU")
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--max-eval", type=int, default=None)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--no-save-model", action="store_true")
    p.add_argument("--set", nargs="*", default=[],
                   help="model kwarg overrides, e.g. --set max_depth=10 epochs=40")
    a = p.parse_args()

    kw: dict = {}
    for item in a.set:
        k, v = item.split("=", 1)
        try:
            kw[k] = json.loads(v)
        except json.JSONDecodeError:
            kw[k] = v

    run(a.data, a.model, a.horizon, a.split, a.fold, a.holdout_cohort,
        None if a.gpu < 0 else a.gpu, a.max_train, a.max_eval, a.seed, a.out,
        a.n_jobs, not a.no_save_model, kw)


if __name__ == "__main__":
    main()
