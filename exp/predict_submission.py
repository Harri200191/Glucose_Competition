"""Build a competition submission parquet.

    python -m exp.predict_submission --competition live --model catboost
    python -m exp.predict_submission --competition annual --model catboost

The template rows are arbitrary 5-minute slots, not the strided anchors in
`build/main` (only ~16 % of them coincide), so features are rebuilt from the raw
corpus for **exactly** the template rows, using the identical clean → segment →
interpolate → feature path as training. Nothing is re-derived differently.

Output obeys the toolkit's rules (`toolkit/README.md`):
  * exact template rows, columns and **row order**;
  * all four `pred_*` columns present;
  * a populated column has **no NaN anywhere** -- any row whose features could not be
    built falls back to persistence, then to the training median, so a column is
    never partially filled;
  * horizons you skip stay entirely NaN.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .clean import LOAD_COLUMNS, clean_frame
from .config import (BUILD_DIR, CORPUS_PARQUET, GRID_MIN, HOLDOUT_PARQUET, HORIZONS,
                     ROOT, RUNS_DIR, ExperimentConfig)
from .features import build_channels, build_feature_matrix
from .models import SEQUENCE_MODELS
from .models.base import BaseModel
from .segments import compute_segments, interpolate_short_gaps, nan_run_lengths

TEMPLATES = {
    "live": ROOT / "toolkit" / "data" / "live_leaderboard" / "template.parquet",
    "annual": ROOT / "toolkit" / "data" / "annual_competition" / "template.parquet",
}
SOURCES = {"live": CORPUS_PARQUET, "annual": HOLDOUT_PARQUET}


def _log(msg: str) -> None:
    print(msg, flush=True)


def build_template_features(competition: str, cfg: ExperimentConfig,
                            want_channels: bool = False):
    """Feature matrix aligned row-for-row with the template.

    `want_channels` additionally returns the full-resolution channel array the
    sequence models (`tcn`, `gru`) window over; `anchor_idx` indexes into it.
    """
    template = pd.read_parquet(TEMPLATES[competition])
    _log(f"[template] {len(template):,} rows, "
         f"{template.groupby(['source_file','id']).ngroups} subjects")

    t0 = time.time()
    df = pd.read_parquet(SOURCES[competition], columns=LOAD_COLUMNS)
    _log(f"[load]     {len(df):,} rows in {time.time()-t0:.1f}s")

    # Keep only the subjects the template asks about. Safe because no feature ever
    # crosses a subject boundary; it cuts the work by ~4x for the live template.
    want = set(zip(template["source_file"].astype(str), template["id"].astype(str)))
    keep = np.fromiter(((s, i) in want for s, i in
                        zip(df["source_file"].astype(str), df["id"].astype(str))),
                       bool, len(df))
    df = df[keep].reset_index(drop=True)
    _log(f"[subset]   {len(df):,} rows for {len(want)} template subjects")

    df = clean_frame(df)
    seg_id = compute_segments(df["source_file"], df["id"], df["date"])
    _log(f"[segment]  {int(seg_id[-1]-seg_id[0]+1):,} segments")

    cgm_raw = df["CGM"].to_numpy(dtype=np.float64)
    real = np.isfinite(cgm_raw)
    gap_len, _, _ = nan_run_lengths(~real, seg_id)
    df["_gap_len"] = gap_len
    cgm_f, was_interp = interpolate_short_gaps(
        cgm_raw, seg_id, cfg.segment.interp_gap_min // GRID_MIN)

    # Locate each template row in the corpus, preserving template order exactly.
    pos = pd.DataFrame({
        "source_file": df["source_file"].astype(str),
        "id": df["id"].astype(str),
        "date": df["date"],
        "_pos": np.arange(len(df), dtype=np.int64),
    })
    tpl = template[["source_file", "id", "date"]].copy()
    tpl["source_file"] = tpl["source_file"].astype(str)
    tpl["id"] = tpl["id"].astype(str)
    merged = tpl.merge(pos, on=["source_file", "id", "date"], how="left", sort=False)
    assert len(merged) == len(template), "merge changed the row count"

    matched = merged["_pos"].notna().to_numpy()
    n_missing = int((~matched).sum())
    _log(f"[match]    {matched.sum():,}/{len(template):,} template rows located"
         f"{f'  ({n_missing:,} MISSING -> fallback)' if n_missing else ''}")

    anchor_idx = merged["_pos"].fillna(0).to_numpy().astype(np.int64)
    t0 = time.time()
    X, names, meta = build_feature_matrix(df, seg_id, cgm_f, was_interp,
                                          anchor_idx, cfg.features)
    _log(f"[features] {X.shape[0]:,} x {X.shape[1]} in {time.time()-t0:.1f}s")

    # Rows we could not locate get no features at all.
    X[~matched] = np.nan
    cgm_now = np.where(matched, cgm_f[anchor_idx], np.nan).astype(np.float64)

    channels = build_channels(df, cgm_f, real) if want_channels else None
    if channels is not None:
        _log(f"[channels] {channels.shape} ({channels.nbytes/1e9:.2f} GB)")
    return template, X, names, meta, cgm_now, matched, anchor_idx, channels


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--competition", choices=["live", "annual"], default="live")
    p.add_argument("--model", default="catboost",
                   help="model name; runs/<model>_h<H>_official/model.joblib must exist")
    p.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    p.add_argument("--horizons", nargs="*", type=int, default=list(HORIZONS))
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--data", type=Path, default=BUILD_DIR / "main",
                   help="built dataset, used only to sanity-check feature order")
    a = p.parse_args()

    cfg = ExperimentConfig()
    out = a.out or (RUNS_DIR / f"submission_{a.competition}_{a.model}.parquet")

    # ---- feature order must match what the models were trained on -------------
    trained_names = None
    man = a.data / "manifest.json"
    if man.exists():
        trained_names = json.loads(man.read_text())["feature_names"]

    is_seq = a.model in SEQUENCE_MODELS
    template, X, names, meta, cgm_now, matched, anchor_idx, channels = \
        build_template_features(a.competition, cfg, want_channels=is_seq)
    if trained_names is not None and names != trained_names:
        raise SystemExit(
            "feature order differs from the trained dataset -- refusing to predict.\n"
            f"  built here: {len(names)} features\n  trained on: {len(trained_names)}\n"
            "Rebuild the models or align exp/config.py.")

    sub = template.copy()
    for c in ("pred_30", "pred_60", "pred_90", "pred_120"):
        sub[c] = np.nan

    # fallback ladder: persistence, then the corpus median
    fallback = np.where(np.isfinite(cgm_now), cgm_now, 140.0)

    summary = {}
    for h in a.horizons:
        tag = f"{a.model}_h{h}_official"
        run = a.runs_dir / tag
        pt, jl = run / "model.pt", run / "model.joblib"
        mp = pt if pt.exists() else jl
        if not mp.exists():
            _log(f"[h{h}] no saved model in {run} -- leaving pred_{h} entirely NaN")
            continue

        t0 = time.time()
        if mp.suffix == ".pt":
            from .models.neural import NeuralModel
            model = NeuralModel.load_checkpoint(mp)
            if model.is_seq_:
                model.attach_sequences(channels, anchor_idx)
                pred = np.asarray(model.predict(X, rows=anchor_idx), dtype=np.float64)
            else:
                pred = np.asarray(model.predict(X), dtype=np.float64)
        else:
            model: BaseModel = BaseModel.load(mp)
            pred = np.asarray(model.predict(X), dtype=np.float64)

        bad = ~np.isfinite(pred) | (pred <= 0)
        n_bad = int(bad.sum())
        pred[bad] = fallback[bad]
        pred = np.clip(pred, 20.0, 600.0)          # physiological guard rails
        sub[f"pred_{h}"] = pred
        summary[h] = dict(n_fallback=n_bad, seconds=round(time.time() - t0, 1),
                          mean=round(float(pred.mean()), 2))
        _log(f"[h{h}] predicted {len(pred):,} rows in {time.time()-t0:.1f}s "
             f"(mean {pred.mean():.1f} mg/dL"
             f"{f', {n_bad:,} fallback' if n_bad else ''})")

    # ---- validate the submission contract ------------------------------------
    cols = ["id", "source_file", "date", "pred_30", "pred_60", "pred_90", "pred_120"]
    sub = sub[cols]
    assert len(sub) == len(template), "row count changed"
    assert (sub["date"].to_numpy() == template["date"].to_numpy()).all(), "row order changed"

    populated = []
    for c in ["pred_30", "pred_60", "pred_90", "pred_120"]:
        n_nan = int(sub[c].isna().sum())
        if n_nan == 0:
            populated.append(c)
        elif n_nan != len(sub):
            raise SystemExit(f"{c} is partially filled ({n_nan:,} NaN) -- the toolkit "
                             "rejects that. Fill it completely or leave it all NaN.")
    if not populated:
        raise SystemExit("no horizon was populated")

    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_parquet(out, index=False)

    import hashlib
    md5 = hashlib.md5(out.read_bytes()).hexdigest()
    _log(f"\nwrote {out}  ({out.stat().st_size/1e6:.1f} MB)")
    _log(f"  rows       {len(sub):,}")
    _log(f"  populated  {', '.join(populated)}")
    _log(f"  md5        {md5}")
    (out.with_suffix(".json")).write_text(json.dumps(
        dict(competition=a.competition, model=a.model, rows=len(sub),
             populated=populated, md5=md5, per_horizon=summary,
             template_rows_matched=int(matched.sum()),
             built_at=pd.Timestamp.now().isoformat(timespec="seconds")), indent=2))
    try:
        shown = f"../{out.relative_to(ROOT)}"
    except ValueError:                       # --out pointed outside the repo
        shown = str(out.resolve())
    _log(f"\nvalidate with:\n  cd toolkit && python run.py {shown} "
         f"--competition {a.competition} --horizon all")


if __name__ == "__main__":
    main()
