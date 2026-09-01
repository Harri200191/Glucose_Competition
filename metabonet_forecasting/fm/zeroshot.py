"""Zero-shot foundation-model forecasts on the official test split.

    python -m metabonet_forecasting.fm.zeroshot \
        --model chronos-bolt --name fm_chronosbolt_zs --gpu 0 \
        --set data.store=... [--stride 8] [--limit 20000]

Produces a run directory shaped exactly like a trained run — the same
``predictions_test.parquet`` columns, scored by the same ``score_run`` — so the
zero-shot rows drop into ``results/all_runs.csv`` beside everything else.

The model input is the **raw CGM series in mg/dL** read straight from the
store's channel memmap: short gaps arrive interpolated (as the features do),
true gaps arrive as NaN and are handled per backend. No standardisation, no
engineered features — that is the point of the zero-shot condition.

``--stride`` thins the test anchors (every k-th) for a cheap first ranking;
the winner gets a full run with ``--stride 1``.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

GRID_MIN = 5                       # minutes per lattice step
PRED_STEPS = 24                    # 120 min ahead


# --------------------------------------------------------------------- backends
class Backend:
    """predict(batch [B, T] float32 with NaN) -> [B, PRED_STEPS] mg/dL.

    ``past_cov`` / ``fut_cov`` are dicts of name -> [B, T] / [B, PRED_STEPS]
    arrays; only backends that natively support covariates may accept them.
    """

    name = "?"
    supports_covariates = False

    def predict(self, context: np.ndarray, past_cov=None,
                fut_cov=None) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class ChronosBackend(Backend):
    """Chronos / Chronos-Bolt (Amazon). Handles NaN natively via masking."""

    def __init__(self, model_id: str, device: str) -> None:
        import torch
        from chronos import BaseChronosPipeline
        self.torch = torch
        self.name = model_id
        self.pipe = BaseChronosPipeline.from_pretrained(
            model_id, device_map=device,
            torch_dtype=torch.float32)

    def predict(self, context: np.ndarray, past_cov=None,
                fut_cov=None) -> np.ndarray:
        t = self.torch.from_numpy(context)
        _, mean = self.pipe.predict_quantiles(
            t, prediction_length=PRED_STEPS, quantile_levels=[0.1, 0.5, 0.9])
        return mean.float().cpu().numpy()


class Chronos2Backend(Backend):
    """Chronos-2 (Amazon, 2025): multivariate + future-known covariates.

    The only zero-shot backend here that can consume the planned-insulin
    channels, which is what the competition's own data design permits."""

    supports_covariates = True

    def __init__(self, model_id: str, device: str) -> None:
        from chronos import Chronos2Pipeline
        self.name = model_id
        self.pipe = Chronos2Pipeline.from_pretrained(model_id, device_map=device)

    def predict(self, context: np.ndarray, past_cov=None,
                fut_cov=None) -> np.ndarray:
        if past_cov:
            inputs = []
            for i, row in enumerate(context):
                item = {"target": row,
                        "past_covariates": {k: v[i] for k, v in past_cov.items()}}
                if fut_cov:
                    item["future_covariates"] = {k: v[i] for k, v in fut_cov.items()}
                inputs.append(item)
        else:
            inputs = [row for row in context]
        _, mean = self.pipe.predict_quantiles(
            inputs, prediction_length=PRED_STEPS, quantile_levels=[0.1, 0.5, 0.9])
        return np.stack([np.asarray(m.float().cpu())[0] for m in mean])


class TimesFMBackend(Backend):
    """TimesFM 2.x (Google). NaN is forward-filled — it has no missing mask."""

    def __init__(self, model_id: str, device: str) -> None:
        import timesfm
        self.name = model_id
        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_id)
        self.model.compile(timesfm.ForecastConfig(
            max_context=512, max_horizon=64,
            normalize_inputs=True, use_continuous_quantile_head=False))

    def predict(self, context: np.ndarray, past_cov=None,
                fut_cov=None) -> np.ndarray:
        filled = _ffill(context)
        point, _ = self.model.forecast(
            horizon=PRED_STEPS, inputs=[row for row in filled])
        return np.asarray(point)[:, :PRED_STEPS]


class TimesFM3Backend(Backend):
    """TimesFM 3.0 (Google, Aug 2026): 330M decoder-only, natively multivariate.

    The first zero-shot backend here besides Chronos-2 that can read covariates
    it knows into the future, which is what makes the planned-insulin channels
    usable without training anything. Verified against the installed package:

      predict_batch(contexts=[T], horizon=H,
                    past_only_covariates=[[n, T]],
                    past_future_covariates=[[n, T + H]])

    Covariates are VARIATE-FIRST -- ``[n_cov, steps]``. The ``[steps, n_cov]``
    layout raises "Decode function requires horizon > 0", which is not a shape
    error message you would guess from, so it is pinned here by test.

    A channel present in both ``past_cov`` and ``fut_cov`` becomes a
    past-future covariate (its future is genuinely known: it is the planned
    dose); one present only in the past becomes past-only. NaN is forward
    filled -- the model takes no missing mask.
    """

    supports_covariates = True

    def __init__(self, model_id: str, device: str) -> None:
        import timesfm
        self.name = model_id
        self.model = timesfm.TimesFM3Forecaster.from_pretrained(
            model_id, device=device)

    def predict(self, context: np.ndarray, past_cov=None,
                fut_cov=None) -> np.ndarray:
        ctx = [row for row in _ffill(context)]
        b = len(ctx)
        pf = po = None
        if past_cov:
            both = [k for k in past_cov if fut_cov and k in fut_cov]
            only = [k for k in past_cov if k not in set(both)]
            if both:
                pf = [np.stack([np.concatenate([past_cov[k][i], fut_cov[k][i]])
                                for k in both]).astype(np.float32)
                      for i in range(b)]
            if only:
                po = [np.stack([past_cov[k][i] for k in only]).astype(np.float32)
                      for i in range(b)]
        out = self.model.predict_batch(
            contexts=ctx, horizon=PRED_STEPS,
            past_only_covariates=po, past_future_covariates=pf)
        return np.stack([np.asarray(o.forecast)[:PRED_STEPS] for o in out])


class TiRexBackend(Backend):
    """TiRex (NX-AI, 2025): xLSTM-based zero-shot forecaster."""

    def __init__(self, model_id: str, device: str) -> None:
        import torch
        from tirex import load_model
        self.torch = torch
        self.name = model_id
        self.model = load_model(model_id, device=device)

    def predict(self, context: np.ndarray, past_cov=None,
                fut_cov=None) -> np.ndarray:
        t = self.torch.from_numpy(_ffill(context))
        quantiles, mean = self.model.forecast(
            context=t, prediction_length=PRED_STEPS)
        return np.asarray(mean.float().cpu())


BACKENDS = {
    "chronos-bolt":      (ChronosBackend, "amazon/chronos-bolt-base"),
    "chronos-bolt-small": (ChronosBackend, "amazon/chronos-bolt-small"),
    "chronos2":          (Chronos2Backend, "amazon/chronos-2"),
    "timesfm":           (TimesFMBackend, "google/timesfm-2.5-200m-pytorch"),
    "timesfm3":          (TimesFM3Backend, "google/timesfm-3.0-pytorch"),
    "tirex":             (TiRexBackend, "NX-AI/TiRex"),
}


def _ffill(x: np.ndarray) -> np.ndarray:
    """Forward-fill NaN along time; leading NaN becomes the first real value."""
    out = np.array(x, copy=True)
    for row in out:
        mask = np.isnan(row)
        if mask.all():
            row[:] = 120.0                      # nothing to go on; euglycaemia
            continue
        idx = np.where(~mask, np.arange(row.size), 0)
        np.maximum.accumulate(idx, out=idx)
        row[:] = row[idx]
        first = np.flatnonzero(~mask)[0]
        row[:first] = row[first]
    return out


# ------------------------------------------------------------------------ main
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(BACKENDS))
    ap.add_argument("--model-id", default=None,
                    help="override the default checkpoint for the backend")
    ap.add_argument("--config", default="base")
    ap.add_argument("--set", nargs="*", default=[], dest="sets")
    ap.add_argument("--name", default=None)
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--stride", type=int, default=1,
                    help="use every k-th test anchor (cheap ranking pass)")
    ap.add_argument("--covariates", default=None, choices=[None, "insulin"],
                    help="feed insulin (basal+bolus) as past + future-known "
                         "covariates; only for backends that support them. "
                         "Future values never cross the anchor's segment end. "
                         "This is the intervention-conditioned condition — "
                         "label the run _futins like the trained ones.")
    ap.add_argument("--limit", type=int, default=None,
                    help="hard cap on the number of test anchors")
    ap.add_argument("--split", default="test",
                    help="which split to forecast: test | internal_holdout | val. "
                         "Needed so the foundation-model comparator is scored on "
                         "exactly the same anchors as the trained models.")
    ap.add_argument("--no-eval", action="store_true")
    a = ap.parse_args(argv)

    if a.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)

    import pandas as pd
    import torch

    from ..common.config import REPO_ROOT, load_config, save_config
    from ..common.logging import RunLogger, provenance
    from ..data.dataset import build_data
    from ..data.splits import seen_in_train

    cfg = load_config(a.config, a.sets)
    run_name = a.name or f"fm_{a.model.replace('-', '')}_zs"
    run_dir = Path(a.runs_dir or REPO_ROOT / "runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log = RunLogger(run_dir)
    log.info(f"zero-shot {a.model} [{a.split}] -> {run_dir}")

    bundle = build_data(cfg)
    if a.split not in bundle.datasets:
        raise SystemExit(
            f"split {a.split!r} is not in this bundle ({sorted(bundle.datasets)}); "
            f"for internal_holdout set data.internal_holdout.enabled=true")
    ds = bundle.datasets[a.split]
    store = bundle.store
    context = int(cfg.get_path("data.context", 288))
    horizons = [int(h) for h in cfg.get_path("data.horizons_min", [30, 60, 90, 120])]
    horizon_steps = [h // GRID_MIN for h in horizons]

    anchor_ids = ds.indices[:: max(1, a.stride)]
    if a.limit:
        anchor_ids = anchor_ids[: a.limit]
    rows = ds.rows[anchor_ids]
    log.info(f"{len(anchor_ids):,} test anchors (stride {a.stride}), context {context}")

    cgm_chan = int(store.channel_ids(["cgm"])[0])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backend_cls, default_id = BACKENDS[a.model]
    backend = backend_cls(a.model_id or default_id, device)
    log.info(f"backend {backend.name} on {device}")

    cov_names: List[str] = []
    seg_end = None
    if a.covariates == "insulin":
        if not backend.supports_covariates:
            raise SystemExit(f"backend {a.model} does not support covariates")
        cov_names = ["ins_basal", "ins_bolus"]
        cov_ids = {n: int(store.channel_ids([n])[0]) for n in cov_names}
        seg_lookup = store.segments.set_index("seg")["end"]
        seg_end = seg_lookup.loc[
            store.anchors["seg"].to_numpy()[anchor_ids]].to_numpy()
        log.info(f"covariates: {cov_names} (past {context} steps + future "
                 f"{PRED_STEPS} steps, segment-capped)")

    preds = np.full((len(anchor_ids), len(horizons)), np.nan, dtype=np.float32)
    t0 = time.time()
    for lo in range(0, len(anchor_ids), a.batch):
        hi = min(lo + a.batch, len(anchor_ids))
        block = np.stack([
            np.asarray(store.channels[r - context + 1: r + 1, cgm_chan],
                       dtype=np.float32)
            for r in rows[lo:hi]])
        past_cov = fut_cov = None
        if cov_names:
            past_cov, fut_cov = {}, {}
            for n in cov_names:
                cid = cov_ids[n]
                past = np.stack([
                    np.asarray(store.channels[r - context + 1: r + 1, cid],
                               dtype=np.float32) for r in rows[lo:hi]])
                fut = np.zeros((hi - lo, PRED_STEPS), dtype=np.float32)
                for i, r in enumerate(rows[lo:hi]):
                    usable = int(min(PRED_STEPS, max(seg_end[lo + i] - r - 1, 0)))
                    if usable > 0:
                        fut[i, :usable] = np.nan_to_num(np.asarray(
                            store.channels[r + 1: r + 1 + usable, cid],
                            dtype=np.float32))
                past_cov[n] = np.nan_to_num(past)
                fut_cov[n] = fut
        path = backend.predict(block, past_cov, fut_cov)    # [b, PRED_STEPS]
        preds[lo:hi] = path[:, [s - 1 for s in horizon_steps]]
        if (lo // a.batch) % 20 == 0:
            done, rate = hi, hi / max(time.time() - t0, 1e-9)
            log.info(f"  {done:,}/{len(anchor_ids):,}  {rate:,.0f} anchors/s")

    # -------------------------------------------------- predictions parquet
    targets = np.asarray(store.targets)[anchor_ids][:, ds.horizon_ids]
    anchors = store.anchors
    df = pd.DataFrame({"anchor": anchor_ids})
    for col in ("subject", "cohort", "is_test", "known_patient", "cgm_now",
                "cgm_delta_30min", "min_since_carb", "min_since_bolus", "tod_min"):
        df[col] = anchors[col].to_numpy()[anchor_ids]
    df["seen_in_train"] = seen_in_train(store, bundle.split, anchor_ids)
    for j, h in enumerate(horizons):
        truth = targets[:, j]
        df[f"y_true_{h}"] = truth
        df[f"y_pred_{h}"] = np.where(np.isfinite(truth), preds[:, j], np.nan)
    df.to_parquet(run_dir / f"predictions_{a.split}.parquet", index=False)
    log.info(f"wrote {len(df):,} predictions in {time.time() - t0:.0f}s")

    save_config(cfg, run_dir / "config.resolved.yaml")
    log.write_json("manifest.json", {
        "run_name": run_name, "zero_shot": True, "backend": backend.name,
        "stride": a.stride, "limit": a.limit, "context": context,
        "split": a.split,
        "config": cfg.to_dict(), "provenance": provenance(),
    })

    if not a.no_eval:
        from ..evaluation.evaluate import score_run
        metrics = score_run(run_dir, bundle=bundle, split=a.split)
        for h in metrics["horizons_min"]:
            o = metrics["strata"]["all"][str(h)]["overall"]
            log.info(f"{a.split} h={h:>3}min  RMSE {o['rmse']:6.2f} mg/dL  "
                     f"MAE {o['mae']:6.2f}  MARD {o['mard']:5.2f}%  "
                     f"DTS-A {o['dts_A']:5.1f}%  n={o['n']:,}")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
