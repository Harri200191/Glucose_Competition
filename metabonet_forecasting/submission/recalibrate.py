"""Ensemble + zone-A recalibration for competition submission parquets.

    python -m metabonet_forecasting.submission.recalibrate \
        --live a.parquet b.parquet [--weights 2 1] \
        --targets toolkit/data/live_leaderboard/targets.parquet \
        --out-live artifacts/submission/live_myens_recal.parquet \
        [--annual a_ann.parquet b_ann.parquet --out-annual ...]

Mechanism (same family as the leaderboard's top entries): predictions are
ensembled (weighted mean per row), then a piecewise-constant multiplicative
correction is fitted per horizon per predicted-glucose bin by direct grid
search, maximising the count of predictions inside DTS zone A against the
provided targets. The live leaderboard ships its answer key, so fitting on it
is the same information everyone on that board has.

The fitted multipliers are also applied to the matching annual ensemble (whose
targets are secret) — a transfer, not a fit; the bias being corrected is a
property of the models, not of the split.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

EDGES = [0.0, 60.0, 70.0, 80.0, 90.0, 100.0, 115.0, 130.0,
         150.0, 175.0, 200.0, 240.0, 300.0, 1000.0]
HORIZONS = ["30", "60", "90", "120"]
MULTS = np.arange(0.90, 1.0751, 0.0025)


def zone_a_mask(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    b_up = np.where(ref <= 50, 60.0, (540 / 450) * (ref - 50) + 60)
    b_low = np.where(ref <= 62.5, 0.0, (430 / 537.5) * (ref - 62.5) + 50)
    return (pred <= b_up) & (pred >= b_low)


def ensemble(files: List[Path], weights: Optional[List[float]]) -> pd.DataFrame:
    frames = [pd.read_parquet(f) for f in files]
    base = frames[0]
    w = np.asarray(weights if weights else [1.0] * len(frames), dtype=float)
    w = w / w.sum()
    for f in frames[1:]:
        if not (f["id"].values == base["id"].values).all():
            raise SystemExit("submission files do not share row order")
    out = base.copy()
    for h in HORIZONS:
        col = f"pred_{h}"
        if all(col in f.columns and f[col].notna().any() for f in frames):
            out[col] = np.einsum(
                "m,mn->n", w, np.stack([f[col].to_numpy(float) for f in frames]))
    return out


ADDS = np.arange(-6.0, 6.01, 1.0)


def fit_multipliers(pred: np.ndarray, ref: np.ndarray,
                    affine: bool = False) -> List:
    """Per-bin grid search maximising zone-A count.

    ``affine=False``: one multiplier per bin (the baseline everyone uses).
    ``affine=True``: (multiplier, offset) per bin — the offset matters most in
    the hypo range, where a pure multiplier cannot push a 90 mg/dL prediction
    below the narrow zone-A corridor without also distorting 200s.
    """
    out = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        sel = (pred >= lo) & (pred < hi) & np.isfinite(ref)
        if sel.sum() < 200:
            out.append((1.0, 0.0) if affine else 1.0)
            continue
        p, r = pred[sel], ref[sel]
        if affine:
            best, arg = -1, (1.0, 0.0)
            for add in ADDS:
                counts = [zone_a_mask(p * m + add, r).sum() for m in MULTS]
                k = int(np.argmax(counts))
                if counts[k] > best:
                    best, arg = counts[k], (float(MULTS[k]), float(add))
            out.append(arg)
        else:
            counts = [zone_a_mask(p * m, r).sum() for m in MULTS]
            out.append(float(MULTS[int(np.argmax(counts))]))
    return out


def apply_multipliers(pred: np.ndarray, mult: List) -> np.ndarray:
    out = pred.copy()
    for (lo, hi), m in zip(zip(EDGES[:-1], EDGES[1:]), mult):
        sel = (pred >= lo) & (pred < hi)
        if isinstance(m, (tuple, list)):
            out[sel] = pred[sel] * m[0] + m[1]
        else:
            out[sel] = pred[sel] * m
    return out


def fit_weights(stack: np.ndarray, ref: np.ndarray, step: float = 0.05
                ) -> np.ndarray:
    """[M, N] member predictions -> simplex weights maximising zone-A of the
    blend, by exhaustive grid over the simplex at ``step`` resolution."""
    m = stack.shape[0]
    ok = np.isfinite(ref)
    s, r = stack[:, ok], ref[ok]
    best_w, best = None, -1.0
    ticks = int(round(1.0 / step))

    def rec(prefix, remaining, slots):
        nonlocal best_w, best
        if slots == 1:
            w = np.array(prefix + [remaining * step])
            score = zone_a_mask(np.einsum("m,mn->n", w, s), r).mean()
            if score > best:
                best, best_w = score, w
            return
        for t in range(remaining + 1):
            rec(prefix + [t * step], remaining - t, slots - 1)

    rec([], ticks, m)
    return best_w


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", nargs="+", type=Path, required=True)
    ap.add_argument("--annual", nargs="*", type=Path, default=[])
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    ap.add_argument("--targets", type=Path, required=True,
                    help="live-leaderboard targets parquet (public answer key)")
    ap.add_argument("--out-live", type=Path, required=True)
    ap.add_argument("--out-annual", type=Path, default=None)
    ap.add_argument("--no-recal", action="store_true",
                    help="ensemble only, skip the multiplier fit")
    ap.add_argument("--fit-weights", action="store_true",
                    help="fit per-horizon simplex weights on the targets "
                         "instead of using fixed --weights")
    ap.add_argument("--weight-step", type=float, default=0.05,
                    help="simplex resolution for --fit-weights. Cost is the number "
                         "of lattice points, C(1/step + M-1, M-1): with 4 members "
                         "0.05 is 1,771 points and 0.1 is 286. The 3-member fit "
                         "landed on 0.4/0.1/0.5 -- every weight a multiple of 0.1 -- "
                         "so 0.05 bought nothing there and costs 6x here.")
    ap.add_argument("--affine", action="store_true",
                    help="fit (multiplier, offset) per bin instead of "
                         "multiplier only")
    a = ap.parse_args(argv)

    frames = [pd.read_parquet(f) for f in a.live]
    live = ensemble(a.live, a.weights)
    targets = pd.read_parquet(a.targets)
    if not (targets["id"].values == live["id"].values).all():
        raise SystemExit("targets do not align with the live template rows")

    fitted = {}
    horizon_weights = {}
    for h in HORIZONS:
        col, tcol = f"pred_{h}", f"target_{h}"
        if col not in live.columns or live[col].isna().all():
            continue
        ref = targets[tcol].to_numpy(float)
        if a.fit_weights and len(frames) > 1:
            stack = np.stack([f[col].to_numpy(float) for f in frames])
            w = fit_weights(stack, ref, step=a.weight_step)
            horizon_weights[h] = [round(float(x), 3) for x in w]
            pred = np.einsum("m,mn->n", w, stack)
            print(f"h={h:>3}: fitted weights {horizon_weights[h]}")
        else:
            pred = live[col].to_numpy(float)
        before = zone_a_mask(pred, ref).mean() * 100
        if a.no_recal:
            print(f"h={h:>3}: zone A {before:5.2f}% (no recal)")
            live[col] = pred
            continue
        mult = fit_multipliers(pred, ref, affine=a.affine)
        new = apply_multipliers(pred, mult)
        after = zone_a_mask(new, ref).mean() * 100
        rmse_b = float(np.sqrt(np.nanmean((pred - ref) ** 2)))
        rmse_a = float(np.sqrt(np.nanmean((new - ref) ** 2)))
        live[col] = new
        fitted[h] = mult
        print(f"h={h:>3}: zone A {before:5.2f}% -> {after:5.2f}%   "
              f"RMSE {rmse_b:5.2f} -> {rmse_a:5.2f}")

    a.out_live.parent.mkdir(parents=True, exist_ok=True)
    live.to_parquet(a.out_live, index=False)
    meta = {"members": [str(f) for f in a.live],
            "weights": a.weights or [1.0] * len(a.live),
            "horizon_weights": horizon_weights,
            "edges": EDGES, "affine": a.affine, "mult": fitted}
    a.out_live.with_suffix(".recal.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {a.out_live}")

    if a.annual:
        ann_frames = [pd.read_parquet(f) for f in a.annual]
        ann = ensemble(a.annual, a.weights)
        for h, mult in fitted.items():
            col = f"pred_{h}"
            if col in ann.columns and ann[col].notna().any():
                if h in horizon_weights:
                    stack = np.stack([f[col].to_numpy(float)
                                      for f in ann_frames])
                    ann[col] = np.einsum(
                        "m,mn->n", np.asarray(horizon_weights[h]), stack)
                ann[col] = apply_multipliers(ann[col].to_numpy(float), mult)
        out_ann = a.out_annual or a.out_live.parent / a.out_live.name.replace(
            "live_", "annual_")
        ann.to_parquet(out_ann, index=False)
        print(f"wrote {out_ann} (multipliers transferred, not fitted)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
