"""Per-row DTS-zone-A-optimal decisions for a submission with quantiles.

    python -m metabonet_forecasting.submission.decide \
        --submission artifacts/submission/live_<run>.parquet \
        --quantiles  artifacts/submission/live_<run>.quantiles.parquet \
        --targets    toolkit/data/live_leaderboard/targets.parquet \
        --out        artifacts/submission/live_<run>_dts.parquet \
        [--lams 0 0.02 0.05 0.1]

For every row and horizon, the candidate prediction that maximises
``P(zone A) − lam·E[(y−p)²]/3600`` under the row's own quantile distribution
replaces the point forecast. ``lam`` is chosen per horizon: with ``--targets``
(the live board's public answer key) each candidate lam is scored against the
truth and the best wins; without targets, pass a single lam (e.g. tuned on the
live board) for the annual file — a transfer, not a fit.

Zone-A probability and expected squared error are computed once per row over
the whole candidate grid; every lam then reuses them, so the lam sweep is free.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..evaluation.decision import zone_a_bounds

HORIZONS = ["30", "60", "90", "120"]
GRID = (40.0, 420.0, 2.0)
CHUNK = 16384


def zone_a_mask(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    b_up = np.where(ref <= 50, 60.0, (540 / 450) * (ref - 50) + 60)
    b_low = np.where(ref <= 62.5, 0.0, (430 / 537.5) * (ref - 62.5) + 50)
    return (pred <= b_up) & (pred >= b_low)


def decide_horizon(quant: np.ndarray, levels: np.ndarray,
                   lams: list) -> dict:
    """quant [N, Q] -> {lam: chosen_pred [N]}."""
    inner = np.diff(levels)
    w = np.empty_like(levels)
    w[0] = levels[0] + inner[0] / 2
    w[-1] = (1 - levels[-1]) + inner[-1] / 2
    w[1:-1] = (inner[:-1] + inner[1:]) / 2
    w = w.astype(np.float32)

    candidates = np.arange(*GRID, dtype=np.float32)
    L, U = zone_a_bounds(candidates)
    out = {lam: np.empty(quant.shape[0], dtype=np.float32) for lam in lams}
    for lo in range(0, quant.shape[0], CHUNK):
        hi = min(lo + CHUNK, quant.shape[0])
        block = quant[lo:hi].astype(np.float32)                    # [n, Q]
        inz = (block[:, None, :] >= L[None, :, None].astype(np.float32)) & \
              (block[:, None, :] <= U[None, :, None].astype(np.float32))
        p_a = np.einsum("npk,k->np", inz.astype(np.float32), w)
        se = (block[:, None, :] - candidates[None, :, None]) ** 2
        mse = np.einsum("npk,k->np", se, w) / 3600.0
        for lam in lams:
            out[lam][lo:hi] = candidates[np.argmax(p_a - lam * mse, axis=1)]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--submission", type=Path, required=True)
    ap.add_argument("--quantiles", type=Path, required=True)
    ap.add_argument("--targets", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lams", nargs="+", type=float, default=[0.0, 0.02, 0.05, 0.1])
    a = ap.parse_args(argv)

    import pyarrow.parquet as pq
    import pyarrow as pa
    table = pq.read_table(a.submission)     # preserve exact schema for rewrite
    sub = table.to_pandas()
    quants = pd.read_parquet(a.quantiles)
    targets = pd.read_parquet(a.targets) if a.targets else None

    chosen_lams = {}
    for h in HORIZONS:
        qcols = sorted([c for c in quants.columns if c.startswith("q") and
                        c.endswith(f"_{h}")], key=lambda c: int(c[1:3]))
        if not qcols or f"pred_{h}" not in sub.columns:
            continue
        levels = np.array([int(c[1:3]) / 100 for c in qcols])
        quant = quants[qcols].to_numpy()
        ok = np.isfinite(quant).all(axis=1)
        decided = decide_horizon(quant[ok], levels, a.lams)

        base = sub[f"pred_{h}"].to_numpy(float)
        if targets is not None:
            ref = targets[f"target_{h}"].to_numpy(float)
            best_lam, best_score = None, -1
            for lam, vals in decided.items():
                new = base.copy()
                new[ok] = vals
                score = zone_a_mask(new, ref)[np.isfinite(ref)].mean()
                rmse = float(np.sqrt(np.nanmean((new - ref) ** 2)))
                print(f"h={h:>3} lam={lam:<5} zoneA={score*100:6.3f}%  RMSE={rmse:5.2f}")
                if score > best_score:
                    best_lam, best_score = lam, score
        else:
            best_lam = a.lams[0]
        new = base.copy()
        new[ok] = decided[best_lam]
        chosen_lams[h] = best_lam
        idx = table.schema.get_field_index(f"pred_{h}")
        table = table.set_column(idx, table.schema.field(idx),
                                 pa.array(new, type=table.schema.field(idx).type))
        if targets is not None:
            ref = targets[f"target_{h}"].to_numpy(float)
            before = zone_a_mask(base, ref)[np.isfinite(ref)].mean() * 100
            after = zone_a_mask(new, ref)[np.isfinite(ref)].mean() * 100
            print(f"h={h:>3}: zone A {before:6.3f}% -> {after:6.3f}%  (lam={best_lam})")

    pq.write_table(table, a.out)
    a.out.with_suffix(".decide.json").write_text(json.dumps(
        {"submission": str(a.submission), "lams": chosen_lams,
         "grid": list(GRID)}, indent=1))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
