"""Turn the raw parquet into a modelling matrix on disk.

    python -m exp.build_dataset --out build/main

Writes:
    features.npy    float32 [n_samples, n_features]
    targets.npy     float32 [n_samples, 4]        CGM at t+30/60/90/120, NaN if absent
    meta.parquet    source_file, id, date, is_test, seg_id, row_idx, cgm_now
    manifest.json   feature names, full config, cleaning log, segment audit, counts

Nothing here is model-specific. Every model in `exp/models` consumes this artefact.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (BUILD_DIR, CORPUS_PARQUET, GRID_MIN, HORIZONS, HORIZON_STEPS,
                     ExperimentConfig)
from .clean import COLUMN_DECISIONS, LOAD_COLUMNS, clean_frame, dropped_columns
from .features import blocked_lead, blocked_rolling, build_feature_matrix, feature_plan
from .segments import (audit_segments, compute_segments, interpolate_short_gaps,
                       nan_run_lengths, segment_bounds)


def build(out_dir: Path, cfg: ExperimentConfig, limit_subjects: int | None = None,
          cohorts: list[str] | None = None, verbose: bool = True,
          emit_channels: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    log = lambda *a: print(*a, flush=True) if verbose else None
    stats: dict = {}
    t_start = time.time()

    # ---------------------------------------------------------------- load
    t0 = time.time()
    df = pd.read_parquet(CORPUS_PARQUET, columns=LOAD_COLUMNS)
    log(f"[load]     {len(df):,} rows x {df.shape[1]} cols in {time.time()-t0:.1f}s")
    stats["rows_raw"] = int(len(df))

    if cohorts:
        df = df[df["source_file"].isin(cohorts)].reset_index(drop=True)
        log(f"[filter]   cohorts={cohorts} -> {len(df):,} rows")
    if limit_subjects:
        # Boolean mask, never a merge: a merge can reorder rows, and the whole
        # pipeline depends on the corpus staying in (subject, time) order.
        keys = (df[["source_file", "id"]].drop_duplicates()
                  .sample(limit_subjects, random_state=cfg.seed))
        want = set(map(tuple, keys.to_numpy()))
        mask = np.fromiter(((s, i) in want for s, i in
                            zip(df["source_file"].to_numpy(), df["id"].to_numpy())),
                           bool, len(df))
        df = df[mask].reset_index(drop=True)
        log(f"[filter]   {limit_subjects} subjects -> {len(df):,} rows")

    # ---------------------------------------------------------------- clean
    t0 = time.time()
    df = clean_frame(df)
    stats["clean_log"] = df.attrs["clean_log"]
    log(f"[clean]    {time.time()-t0:.1f}s  {stats['clean_log']}")

    # ------------------------------------------------------------- segment
    t0 = time.time()
    seg_id = compute_segments(df["source_file"], df["id"], df["date"])
    stats["segments"] = audit_segments(seg_id, df["source_file"], df["id"], df["date"])
    log(f"[segment]  {time.time()-t0:.1f}s  {stats['segments']}")

    # drop rows in segments too short to ever produce an anchor
    starts, ends = segment_bounds(seg_id)
    seg_len = ends - starts + 1
    min_len = cfg.segment.min_segment_min // GRID_MIN
    keep_seg = seg_len >= min_len
    if not keep_seg.all():
        row_keep = np.repeat(keep_seg, seg_len)
        dropped = int((~row_keep).sum())
        df = df[row_keep].reset_index(drop=True)
        seg_id = compute_segments(df["source_file"], df["id"], df["date"])
        log(f"[segment]  dropped {int((~keep_seg).sum())} short segments "
            f"({dropped:,} rows, min {cfg.segment.min_segment_min} min)")
        stats["rows_dropped_short_segments"] = dropped
    else:
        stats["rows_dropped_short_segments"] = 0

    cgm_raw = df["CGM"].to_numpy(dtype=np.float64)
    real = np.isfinite(cgm_raw)
    stats["rows_after_segment_filter"] = int(len(df))
    stats["cgm_real_frac"] = float(real.mean())

    # ------------------------------------------------- gap lengths + interp
    t0 = time.time()
    gap_len, _, _ = nan_run_lengths(~real, seg_id)
    df["_gap_len"] = gap_len
    interp_steps = cfg.segment.interp_gap_min // GRID_MIN
    cgm_f, was_interp = interpolate_short_gaps(cgm_raw, seg_id, interp_steps)
    stats["cgm_interpolated_slots"] = int(was_interp.sum())
    stats["cgm_still_missing_after_interp"] = int((~np.isfinite(cgm_f)).sum())
    log(f"[interp]   {time.time()-t0:.1f}s  filled {was_interp.sum():,} slots "
        f"(gaps <= {cfg.segment.interp_gap_min} min); "
        f"{(~np.isfinite(cgm_f)).sum():,} still missing")

    # -------------------------------------------------------- anchor choice
    t0 = time.time()
    n = len(df)
    hist_steps = cfg.features.required_history_min // GRID_MIN
    starts, _ = segment_bounds(seg_id)
    seg_start = starts[seg_id - seg_id.min()]
    pos_in_seg = np.arange(n, dtype=np.int64) - seg_start

    anchor = pos_in_seg >= hist_steps                       # full history available
    stats["anchor_after_history"] = int(anchor.sum())

    if cfg.sample.require_anchor_cgm:
        anchor &= real
        stats["anchor_after_real_cgm"] = int(anchor.sum())

    # no CGM gap longer than max_gap_min anywhere in the required history
    max_gap = blocked_rolling(gap_len.astype(np.float64), hist_steps, "max", seg_id,
                              min_frac=1.0) * GRID_MIN
    anchor &= np.isfinite(max_gap) & (max_gap <= cfg.segment.max_gap_min)
    stats["anchor_after_gap_quality"] = int(anchor.sum())

    # targets: real readings only, never interpolated
    Y = np.full((n, len(HORIZONS)), np.nan, dtype=np.float32)
    for j, k in enumerate(HORIZON_STEPS):
        Y[:, j] = blocked_lead(cgm_raw, k, seg_id).astype(np.float32)
    has_any_target = np.isfinite(Y).any(axis=1)
    anchor &= has_any_target
    stats["anchor_after_target"] = int(anchor.sum())

    # stride: EDA 9.6 shows r ~ 0.99 at a 5-minute lag, so neighbouring anchors are
    # near-duplicates. Stride on position *within the segment* so the phase is stable.
    stride = max(1, cfg.sample.train_stride)
    anchor &= (pos_in_seg % stride == 0)
    stats["anchor_after_stride"] = int(anchor.sum())
    stats["stride"] = stride

    anchor_idx = np.flatnonzero(anchor)
    if cfg.sample.max_train_samples and len(anchor_idx) > cfg.sample.max_train_samples:
        rng = np.random.default_rng(cfg.seed)
        anchor_idx = np.sort(rng.choice(anchor_idx, cfg.sample.max_train_samples,
                                        replace=False))
        stats["anchor_after_cap"] = int(len(anchor_idx))
    log(f"[anchors]  {time.time()-t0:.1f}s  {len(anchor_idx):,} samples "
        f"({len(anchor_idx)/n*100:.2f}% of rows)")

    # ------------------------------------------------------------ features
    t0 = time.time()
    X, names, fmeta = build_feature_matrix(df, seg_id, cgm_f, was_interp,
                                           anchor_idx, cfg.features)
    log(f"[features] {time.time()-t0:.1f}s  {X.shape[0]:,} x {X.shape[1]} "
        f"({X.nbytes/1e9:.2f} GB)")

    # ------------------------------------------------- raw channels (for RNN/CNN)
    # Sequence models need the full-resolution window, but materialising one window
    # per sample would cost 288 x 5 x 4 B x 10 M = 57 GB. Instead persist the five
    # channels at full length (n_rows x 5 x 4 B ~ 3 GB) and let the loader gather
    # windows lazily via `row_idx`. Transforms are fixed and applied once here:
    #   cgm         : cgm_f / 100 - 1.5      (interpolated series, ~zero-centred)
    #   cgm_present : 1.0 where the reading is REAL, 0.0 where interpolated/missing
    #   bolus/basal/carbs : log1p(x)         (tames the spike-train tails, EDA 5.4)
    if emit_channels:
        t0 = time.time()
        from .features import build_channels
        ch = build_channels(df, cgm_f, real)
        np.save(out_dir / "channels.npy", ch)
        stats["channels_gb"] = round(ch.nbytes / 1e9, 3)
        log(f"[channels] {time.time()-t0:.1f}s  {ch.shape} ({ch.nbytes/1e9:.2f} GB)")
        del ch

    # --------------------------------------------------------------- write
    t0 = time.time()
    np.save(out_dir / "features.npy", X)
    np.save(out_dir / "targets.npy", Y[anchor_idx])
    meta = pd.DataFrame({
        "source_file": df["source_file"].to_numpy()[anchor_idx],
        "id": df["id"].to_numpy()[anchor_idx],
        "date": df["date"].to_numpy()[anchor_idx],
        "is_test": df["is_test"].to_numpy()[anchor_idx],
        "seg_id": seg_id[anchor_idx],
        "row_idx": anchor_idx,
        "cgm_now": cgm_f[anchor_idx],
    })
    meta["subject"] = meta["source_file"].astype(str) + "::" + meta["id"].astype(str)
    meta.to_parquet(out_dir / "meta.parquet", index=False)

    nan_frac = np.isnan(X).mean(axis=0)
    manifest = {
        "built_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "source": str(CORPUS_PARQUET),
        "config": asdict(cfg),
        "horizons_min": list(HORIZONS),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "feature_names": names,
        "feature_groups": fmeta["groups"],
        "categorical_columns": fmeta["categorical_columns"],
        "categorical_levels": {k: [str(x) for x in v]
                               for k, v in fmeta["categorical_levels"].items()},
        "feature_nan_fraction": {n: round(float(f), 5) for n, f in zip(names, nan_frac)},
        "target_counts": {f"h{h}": int(np.isfinite(Y[anchor_idx][:, j]).sum())
                          for j, h in enumerate(HORIZONS)},
        "n_subjects": int(meta["subject"].nunique()),
        "n_cohorts": int(meta["source_file"].nunique()),
        "dropped_columns": dropped_columns(),
        "column_decisions": COLUMN_DECISIONS,
        "stats": stats,
        "build_seconds": round(time.time() - t_start, 1),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    log(f"[write]    {time.time()-t0:.1f}s -> {out_dir}")
    log(f"[done]     {time.time()-t_start:.1f}s total, {X.shape[0]:,} samples, "
        f"{meta['subject'].nunique():,} subjects")
    return manifest


def load_dataset(out_dir: Path, mmap: bool = True):
    X = np.load(out_dir / "features.npy", mmap_mode="r" if mmap else None)
    Y = np.load(out_dir / "targets.npy")
    meta = pd.read_parquet(out_dir / "meta.parquet")
    manifest = json.loads((out_dir / "manifest.json").read_text())
    return X, Y, meta, manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=BUILD_DIR / "main")
    p.add_argument("--stride", type=int, default=6,
                   help="lattice steps between anchors (6 = one sample per 30 min)")
    p.add_argument("--max-gap-min", type=int, default=30,
                   help="longest tolerated CGM gap inside the 24h history")
    p.add_argument("--history-hours", type=int, default=24)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--subjects", type=int, default=None,
                   help="use only N random subjects (smoke tests)")
    p.add_argument("--cohorts", nargs="*", default=None)
    p.add_argument("--no-cohort-feature", action="store_true",
                   help="exclude source_file from the feature set")
    p.add_argument("--no-missingness-features", action="store_true")
    p.add_argument("--no-channels", action="store_true",
                   help="skip channels.npy (sequence models tcn/gru then unavailable)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    cfg = ExperimentConfig()
    cfg.seed = a.seed
    cfg.sample.train_stride = a.stride
    cfg.sample.max_train_samples = a.max_samples
    cfg.segment.max_gap_min = a.max_gap_min
    cfg.features.required_history_min = a.history_hours * 60
    cfg.features.use_cohort_feature = not a.no_cohort_feature
    cfg.features.use_missingness_features = not a.no_missingness_features

    build(a.out, cfg, limit_subjects=a.subjects, cohorts=a.cohorts,
          emit_channels=not a.no_channels)


if __name__ == "__main__":
    main()
