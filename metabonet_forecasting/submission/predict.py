"""Turn a trained run into a competition submission parquet.

    python -m metabonet_forecasting.submission.predict \
        --run runs/e9_mamba2_film_s0_futboth \
        --store artifacts/store/submission_annual \
        --template artifacts/submission/annual_template.parquet \
        --out artifacts/submission/annual_e9_futboth.parquet

The submission store must have been built with ``prepare.py --anchors-from
<template>``, so its anchors are exactly the template's rows, in the template's
order, produced by the same cleaning, segmentation and exposure code that made
the training data. That is the whole point of routing submissions through the
store rather than a bespoke feature builder: an inference path that computes
features slightly differently from the training path is the classic way to ship
a model that scored well in the notebook and badly on the leaderboard.

What this guarantees, and checks:

* every template row gets a prediction, in the template's own row order;
* the run's *own* frozen standardizer is used -- never refitted on the test data;
* horizons the model was not trained on are left entirely NaN, which is what the
  toolkit requires (a partially filled column is rejected outright);
* predictions are clipped to the sensor's reporting range.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from ..common.config import load_config
from ..common.logging import get_console_logger
from ..data import constants as K
from ..data.store import SequenceStore
from ..data.transforms import Standardizer
from ..data.windows import WindowDataset, WindowSpec, batch_to_device, collate
from ..data.dataset import window_spec_from_config
from ..models import build_model

LOG = get_console_logger("submission")

TEMPLATE_COLUMNS = ["id", "source_file", "date", "pred_30", "pred_60", "pred_90", "pred_120"]


def align_categoricals(sub_store: SequenceStore, train_store: SequenceStore) -> None:
    """Re-express the submission store's categorical codes in training terms.

    Each store builds its own vocabulary while reading its own file, in
    first-appearance order. The holdout contains two genders and one delivery
    modality; the corpus contains four and three. So the same integer means
    different things in the two stores, and an embedding table trained on one
    would be indexed with the other's codes -- silently, since the shapes only
    disagree when the counts happen to differ.

    This rewrites the submission store's static codes by *name* into the training
    store's index space and adopts its level tables, so the embedding lookup means
    what it meant during training. A level the training corpus never saw (the
    holdout's synthetic source_file, for one) becomes -1, which the dataset maps
    to the reserved "unknown" embedding row.
    """
    n_num = len(sub_store.manifest["static_numeric"])
    for field in sub_store.manifest["static_categorical"]:
        col = n_num + list(sub_store.manifest["static_categorical"]).index(field)
        sub_levels = sub_store.level_names(field)
        train_index = {name: i for i, name in enumerate(train_store.level_names(field))}
        remap = np.array([train_index.get(name, -1) for name in sub_levels],
                         dtype=np.float32)
        codes = sub_store.statics[:, col]
        valid = np.isfinite(codes) & (codes >= 0) & (codes < len(remap))
        out = np.full(codes.shape, -1.0, dtype=np.float32)
        out[valid] = remap[codes[valid].astype(int)]
        unknown = int((out < 0).sum())
        if unknown:
            LOG.info(f"  {field}: {unknown} subjects have a level the training corpus "
                     f"never saw -> unknown embedding")
        sub_store.statics[:, col] = out
    # Adopt the training level tables so cardinalities match the trained weights.
    sub_store.manifest["levels"] = train_store.manifest["levels"]


def load_run(run_dir: Path):
    """Config, frozen normalisation and weights, exactly as the run left them."""
    cfg = load_config(str(run_dir / "config.resolved.yaml"))
    std = Standardizer.load(run_dir / "standardizer.json")
    return cfg, std


@torch.no_grad()
def predict_template(run_dir: Path, store_root: Path, template_path: Path,
                     device: str = "cuda", batch_size: int = 256,
                     num_workers: int = 4) -> pd.DataFrame:
    cfg, std = load_run(run_dir)
    spec: WindowSpec = window_spec_from_config(cfg)
    store = SequenceStore(store_root)

    anchors = store.anchors
    if "template_row" not in anchors.columns:
        raise ValueError(
            f"{store_root} was not built for a submission. Rebuild it with:\n"
            f"  python -m metabonet_forecasting.data.prepare --corpus <source> "
            f"--out {store_root} --anchors-from {template_path}")

    template = pd.read_parquet(template_path)
    if len(anchors) != len(template):
        raise ValueError(f"store has {len(anchors):,} anchors but the template has "
                         f"{len(template):,} rows")

    # The model's embeddings were trained against the training store's
    # vocabularies; the submission store must speak the same language.
    train_store = SequenceStore(cfg.get_path("data.store"))
    align_categoricals(store, train_store)

    ds = WindowDataset(store, np.ones(len(anchors), dtype=bool), spec, std,
                       pad_history=True)
    shapes = {
        "channel_counts": ds.channel_counts(),
        "n_time_features": ds.n_time_features,
        "n_static_numeric": ds.n_static_numeric if spec.use_static else 0,
        "static_cardinalities": list(ds.cat_cardinalities) if spec.use_static else [],
        "context": spec.context,
        "n_horizons": ds.n_horizons,
        "future_steps": ds.future_shape[0],
        "future_channels": ds.future_shape[1] + 1 if ds.future_shape[1] else 0,
    }
    model = build_model(cfg, shapes)
    blob = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(blob["state_dict"])
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model = model.to(dev).eval()

    LOG.info(f"run {run_dir.name}: {spec.horizons_min} min horizons, context "
             f"{spec.context}, future={spec.future_modalities or 'none (causal)'}")
    LOG.info(f"predicting {len(ds):,} template rows on {dev}")

    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=collate, pin_memory=(dev.type == "cuda"))

    preds: List[np.ndarray] = []
    index: List[np.ndarray] = []
    quants: List[np.ndarray] = []
    for batch in loader:
        batch = batch_to_device(batch, dev)
        out = model(batch)
        mgdl = std.inverse_targets(out.pred_norm.float(), batch["cgm_now"].float())
        preds.append(mgdl.cpu().numpy())
        index.append(batch["index"].cpu().numpy())
        if out.quantiles is not None:
            q = torch.stack([
                std.inverse_targets(out.quantiles[..., k].float(),
                                    batch["cgm_now"].float())
                for k in range(out.quantiles.shape[-1])], dim=-1)
            quants.append(q.cpu().numpy())

    pred = np.concatenate(preds)
    quant = np.concatenate(quants) if quants else None
    anchor_index = np.concatenate(index)
    # Back to template order: the dataset walks anchors in store order, and the
    # store's anchor table carries the template row each one came from.
    template_row = anchors["template_row"].to_numpy()[anchor_index]

    out = template.copy()
    for col in ("pred_30", "pred_60", "pred_90", "pred_120"):
        out[col] = np.nan
    for j, h in enumerate(spec.horizons_min):
        col = f"pred_{h}"
        if col not in out.columns:
            LOG.info(f"model predicts {h} min but the template has no {col}; skipping")
            continue
        values = np.full(len(out), np.nan, dtype=np.float64)
        values[template_row] = np.clip(pred[:, j], K.CGM_REPORT_LO, K.CGM_REPORT_HI)
        out[col] = values

    _validate(out, template, spec)

    # Side table of quantiles for the decision layer. NEVER uploaded: the
    # submission file must match the template's columns exactly, so anything
    # extra lives next to it in its own file, aligned by template row order.
    side = None
    if quant is not None:
        levels = list(cfg.get_path("model.head.quantiles", []) or [])
        side = pd.DataFrame({"id": template["id"].to_numpy()})
        for j, h in enumerate(spec.horizons_min):
            for k, lv in enumerate(levels):
                values = np.full(len(template), np.nan, dtype=np.float64)
                values[template_row] = np.clip(
                    quant[:, j, k], K.CGM_REPORT_LO, K.CGM_REPORT_HI)
                side[f"q{int(round(lv * 100)):02d}_{h}"] = values
    return out, side


def _validate(out: pd.DataFrame, template: pd.DataFrame, spec: WindowSpec) -> None:
    """Refuse to write anything the toolkit would reject."""
    if list(out.columns) != list(template.columns):
        raise ValueError(f"column mismatch: {list(out.columns)} != {list(template.columns)}")
    if len(out) != len(template):
        raise ValueError("row count changed")
    for key in ("id", "source_file", "date"):
        if not out[key].reset_index(drop=True).equals(template[key].reset_index(drop=True)):
            raise ValueError(f"row order changed in column {key!r}")

    filled = []
    for h in (30, 60, 90, 120):
        col = f"pred_{h}"
        n_valid = int(out[col].notna().sum())
        if h in spec.horizons_min:
            if n_valid != len(out):
                raise ValueError(
                    f"{col}: {len(out) - n_valid} missing predictions. A partially "
                    f"filled column is rejected by the scorer; it must be complete "
                    f"or entirely NaN.")
            filled.append(h)
        elif n_valid:
            raise ValueError(f"{col}: model does not predict this horizon but the "
                             f"column has {n_valid} values")
    if not filled:
        raise ValueError("no horizon was populated")
    LOG.info(f"validated: {len(out):,} rows, horizons filled {filled}, "
             f"others entirely NaN")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="a finished run directory")
    ap.add_argument("--store", required=True, help="submission store (--anchors-from)")
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=4)
    a = ap.parse_args(argv)

    out, side = predict_template(Path(a.run), Path(a.store), Path(a.template),
                                 a.device, a.batch_size, a.num_workers)
    dest = Path(a.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dest, index=False)
    if side is not None:
        side_path = dest.with_suffix("").with_name(dest.stem + ".quantiles.parquet")
        side.to_parquet(side_path, index=False)
        LOG.info(f"wrote quantile side table {side_path}")

    described = {f"pred_{h}": (round(float(out[f'pred_{h}'].mean()), 2)
                               if out[f"pred_{h}"].notna().any() else None)
                 for h in (30, 60, 90, 120)}
    LOG.info(f"wrote {dest}  ({len(out):,} rows)  mean prediction: {described}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
