"""Log-space recentring: the zone-A-optimal multiplicative correction.

    # fit on a run's validation predictions, apply to a submission parquet
    python -m metabonet_forecasting.submission.recentre \
        --fit-run runs/my_run --fit-split val \
        --submission artifacts/submission/live_my_run.parquet \
        --out artifacts/submission/live_my_run_rc.parquet

    # or correct another run's predictions in place-ish (writes a sibling dir)
    python -m metabonet_forecasting.submission.recentre \
        --fit-run runs/my_run --fit-split val \
        --apply-run runs/my_run --apply-split internal_holdout

Why this exists
---------------
The organisers' scorer defines DTS zone A as ``b_low(y) <= p <= b_up(y)`` with

    b_up(y)  = (540/450)*(y - 50)   + 60  ==  1.2 * y     (exact, above y=50)
    b_low(y) = (430/537.5)*(y-62.5) + 50  ==  0.8 * y     (exact, above y=62.5)

so zone A is exactly **"within +/-20% of the truth"** -- a *relative* criterion.
Take logs and it becomes

    log(0.8) <= log(p/y) <= log(1.2)
     -0.22314 <= log(p/y) <= +0.18232

which is **not centred on zero**. Its midpoint is ``CENTRE = -0.020411``. You may
be 22.3% low but only 18.2% high, so the zone-A-optimal forecast sits about 2%
*below* the conditional geometric mean -- not at it, and certainly not above it.

Trained models here sit above it: mean ``log(p/y)`` runs +0.007 at 30 min to
+0.028 at 120 min. Closing that gap is one multiplication per horizon:

    p  <-  p * exp(CENTRE - mu),     mu = mean log(p/y) on labelled data

Measured on ``g1_mamba2_film_futins_ep3`` with the multiplier fitted on one half
of the patients and scored on the other: **+0.55 zone-A points**, reproducing at
+0.547 on a different random split. It matches an exhaustive grid search over
multipliers to within 0.01 points, and transfers across populations -- fitting
``mu`` on non-IOBP2 patients and applying it to IOBP2 (the annual-holdout
population) costs 0.006 to 0.058 points.

Only ``mu`` comes from data, and it is one number per horizon estimated from
~10^5 points, so unlike a per-bin multiplier table there is nothing here that can
meaningfully overfit -- and nothing that needs the leaderboard's answer key.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..data import constants as K

# Midpoint of the zone-A window in log space. A property of the scorer, not a
# fitted quantity: (log 0.8 + log 1.2) / 2.
CENTRE: float = (math.log(0.8) + math.log(1.2)) / 2.0

HORIZONS: Sequence[int] = (30, 60, 90, 120)


def fit_mu(pred: np.ndarray, true: np.ndarray) -> Optional[float]:
    """Mean ``log(pred/true)`` over rows where both are usable.

    Returns ``None`` when there is nothing to fit on, so callers can fall back to
    leaving that horizon alone rather than silently applying a garbage factor.
    """
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(true, dtype=np.float64)
    ok = np.isfinite(p) & np.isfinite(t) & (p > 0) & (t > 0)
    if not ok.any():
        return None
    return float(np.mean(np.log(p[ok] / t[ok])))


def multiplier(mu: float) -> float:
    """The correction factor that moves mean log(p/y) from ``mu`` to ``CENTRE``."""
    return float(math.exp(CENTRE - mu))


def fit_from_predictions(df: pd.DataFrame,
                         horizons: Iterable[int] = HORIZONS,
                         ) -> Dict[int, Dict[str, float]]:
    """Per-horizon ``mu`` and multiplier from a ``predictions_*.parquet`` frame."""
    out: Dict[int, Dict[str, float]] = {}
    for h in horizons:
        pc, tc = f"y_pred_{h}", f"y_true_{h}"
        if pc not in df.columns or tc not in df.columns:
            continue
        mu = fit_mu(df[pc].to_numpy(), df[tc].to_numpy())
        if mu is None:
            continue
        ok = (np.isfinite(df[pc].to_numpy()) & np.isfinite(df[tc].to_numpy())
              & (df[tc].to_numpy() > 0))
        out[h] = {"mu": mu, "multiplier": multiplier(mu), "n": int(ok.sum())}
    return out


def apply_multipliers(df: pd.DataFrame, fitted: Dict[int, Dict[str, float]],
                      prefix: str, clip: bool = True) -> pd.DataFrame:
    """Scale ``<prefix><h>`` columns in place on a copy, NaNs preserved.

    ``clip`` keeps the result inside the corpus's reporting range; the scaling is
    small, so this almost never binds, but a submission must not carry a value
    the scorer would not accept.
    """
    out = df.copy()
    for h, info in fitted.items():
        col = f"{prefix}{h}"
        if col not in out.columns:
            continue
        v = out[col].to_numpy(dtype=np.float64) * info["multiplier"]
        if clip:
            v = np.clip(v, K.CGM_REPORT_LO, K.CGM_REPORT_HI)
        out[col] = np.where(np.isfinite(out[col].to_numpy(dtype=np.float64)),
                            v, np.nan)
    return out


def ensemble_frames(frames: Sequence[pd.DataFrame], prefix: str,
                    weights: Optional[Sequence[float]] = None) -> pd.DataFrame:
    """Weighted mean of ``<prefix><h>`` columns across identically-ordered frames.

    Order matters and it is **ensemble first, recentre second**. Each member has
    its own multiplicative bias; averaging first produces one blended bias, which
    a single factor then removes exactly. Recentring each member first and then
    averaging leaves the average of several corrected-but-still-different biases,
    which is not the same thing and measured worse.
    """
    if len(frames) == 1:
        return frames[0].copy()
    base = frames[0]
    for f in frames[1:]:
        if len(f) != len(base):
            raise SystemExit(
                f"members disagree on row count ({len(base)} vs {len(f)}); they "
                f"must be built from the same template")
        if "id" in base.columns and "id" in f.columns:
            if not (f["id"].to_numpy() == base["id"].to_numpy()).all():
                raise SystemExit("members disagree on row order (id mismatch)")
        if "anchor" in base.columns and "anchor" in f.columns:
            if not (f["anchor"].to_numpy() == base["anchor"].to_numpy()).all():
                raise SystemExit("members disagree on row order (anchor mismatch)")
    w = np.asarray(weights if weights else [1.0] * len(frames), dtype=float)
    if w.size != len(frames):
        raise SystemExit("--weights must match the number of members")
    w = w / w.sum()
    out = base.copy()
    for h in HORIZONS:
        col = f"{prefix}{h}"
        cols = [f[col].to_numpy(dtype=np.float64) for f in frames
                if col in f.columns]
        if len(cols) != len(frames):
            continue
        # A member's NaN makes the blend NaN on purpose: silently averaging the
        # rest would give that row fewer models than every other row.
        out[col] = np.einsum("m,mn->n", w, np.stack(cols))
    return out


def _load_fit(fit_run: Optional[Path], fit_split: str,
              fit_predictions: Optional[Path]) -> pd.DataFrame:
    if fit_predictions is not None:
        return pd.read_parquet(fit_predictions)
    if fit_run is None:
        raise SystemExit("pass --fit-run or --fit-predictions")
    path = Path(fit_run) / f"predictions_{fit_split}.parquet"
    if not path.exists():
        raise SystemExit(
            f"{path} not found. Fit the correction on labelled predictions the "
            f"model did not train on -- train with training.predict_val=true, or "
            f"point --fit-split at a split you do have.")
    return pd.read_parquet(path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    fit = ap.add_argument_group("where to estimate mu")
    fit.add_argument("--fit-run", type=Path, default=None)
    fit.add_argument("--fit-split", default="val",
                     help="val by default: labelled, and not trained on")
    fit.add_argument("--fit-predictions", type=Path, default=None,
                     help="a predictions parquet directly, instead of a run")

    tgt = ap.add_argument_group("what to correct (choose one)")
    tgt.add_argument("--submission", type=Path, nargs="+", default=None,
                     help="competition parquet(s) with pred_30..pred_120. Give "
                          "several and they are ENSEMBLED first, then recentred "
                          "once on the blend -- that order is deliberate.")
    tgt.add_argument("--apply-run", type=Path, nargs="+", default=None,
                     help="run director(ies) with predictions_<split>.parquet; "
                          "several are ensembled then recentred")
    tgt.add_argument("--apply-split", default="test")
    tgt.add_argument("--weights", type=float, nargs="+", default=None,
                     help="ensemble weights, one per member (default: equal)")

    ap.add_argument("--out", type=Path, default=None,
                    help="output path (submission) or run dir (apply-run); "
                         "defaults to a sibling with a _rc suffix")
    ap.add_argument("--no-clip", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the multipliers and write nothing")
    a = ap.parse_args(argv)

    fitted = fit_from_predictions(_load_fit(a.fit_run, a.fit_split,
                                            a.fit_predictions))
    if not fitted:
        raise SystemExit("no horizon had both y_pred_* and y_true_* to fit on")

    print(f"CENTRE = {CENTRE:.6f}   (zone A midpoint in log space)")
    for h, info in sorted(fitted.items()):
        print(f"  h={h:>3}  mu={info['mu']:+.5f}  ->  x{info['multiplier']:.5f}"
              f"   (n={info['n']:,})")
    if a.dry_run:
        return 0

    if a.submission is not None:
        import pyarrow as pa
        import pyarrow.parquet as pq
        # The FIRST member defines the schema the organisers get back.
        table = pq.read_table(a.submission[0])
        members = [pq.read_table(f).to_pandas() for f in a.submission]
        if len(members) > 1:
            print(f"ensembling {len(members)} submissions "
                  f"(weights {a.weights or 'equal'}) before recentring")
        df = ensemble_frames(members, "pred_", a.weights)
        new = apply_multipliers(df, fitted, "pred_", clip=not a.no_clip)
        for h in fitted:
            col = f"pred_{h}"
            if col not in df.columns:
                continue
            i = table.schema.get_field_index(col)
            table = table.set_column(
                i, table.schema.field(i),
                pa.array(new[col].to_numpy(), type=table.schema.field(i).type))
        dest = a.out or a.submission[0].with_name(a.submission[0].stem + "_rc.parquet")
        dest.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, dest)

    elif a.apply_run is not None:
        members = []
        for r in a.apply_run:
            src = Path(r) / f"predictions_{a.apply_split}.parquet"
            if not src.exists():
                raise SystemExit(f"{src} not found")
            members.append(pd.read_parquet(src).sort_values("anchor")
                           .reset_index(drop=True))
        if len(members) > 1:
            common = set(members[0]["anchor"])
            for m in members[1:]:
                common &= set(m["anchor"])
            members = [m[m["anchor"].isin(common)].sort_values("anchor")
                       .reset_index(drop=True) for m in members]
            print(f"ensembling {len(members)} runs on {len(common):,} shared "
                  f"anchors (weights {a.weights or 'equal'}) before recentring")
        blended = ensemble_frames(members, "y_pred_", a.weights)
        new = apply_multipliers(blended, fitted, "y_pred_", clip=not a.no_clip)
        first = Path(a.apply_run[0])
        dest_dir = a.out or first.parent / (first.name + "_rc")
        dest_dir.mkdir(parents=True, exist_ok=True)
        new.to_parquet(dest_dir / f"predictions_{a.apply_split}.parquet",
                       index=False)
        for aux in ("config.resolved.yaml", "manifest.json", "standardizer.json"):
            src_aux = first / aux
            if src_aux.exists() and not (dest_dir / aux).exists():
                (dest_dir / aux).write_bytes(src_aux.read_bytes())
        dest = dest_dir
    else:
        raise SystemExit("pass --submission or --apply-run")

    meta = {"centre": CENTRE,
            "members": [str(x) for x in (a.submission or a.apply_run or [])],
            "weights": a.weights,
            "fit": str(a.fit_predictions or (a.fit_run and
                                             f"{a.fit_run}:{a.fit_split}")),
            "clip": not a.no_clip,
            "per_horizon": {str(h): {"mu": round(v["mu"], 6),
                                     "multiplier": round(v["multiplier"], 6),
                                     "n_fit": v["n"]}
                            for h, v in sorted(fitted.items())}}
    meta_path = (Path(dest) / "recentre.json" if Path(dest).is_dir()
                 else Path(dest).with_suffix(".recentre.json"))
    meta_path.write_text(json.dumps(meta, indent=1))
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
