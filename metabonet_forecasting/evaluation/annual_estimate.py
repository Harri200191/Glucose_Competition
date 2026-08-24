"""What this model would score on the annual holdout, with honest error bars.

    python -m metabonet_forecasting.evaluation.annual_estimate runs/<r> \
        --split internal_holdout [--per-subject 25] [--boot 2000]

The internal holdout scores every anchor its 89 patients have -- ~216,000 of
them. The annual competition scores 2,228: about 25 per patient. Both estimate
the same quantity, but only one of them has the annual set's *noise*.

So this reports two numbers per horizon:

``full``      zone A over every anchor. The low-variance point estimate, and the
              right thing to compare two models on.
``annual-like`` the distribution of zone A over repeated draws of
              ``per_subject`` anchors per patient. Its spread is what the annual
              leaderboard will actually do to you.

Reading them together is the point. A 0.4-point gap between two models is real
in the ``full`` column and invisible inside the ``annual-like`` interval, which
means it is real but not bankable on a single 2,228-point draw.

The resample is per patient, matching how the annual template is built (a fixed
small number of anchors from each person), not a flat sample over all anchors --
those have visibly different variance because anchors within a patient are
correlated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .metrics import dts_zones

HORIZON_KEY = "y_true_"


def horizons_of(df: pd.DataFrame) -> List[int]:
    return sorted(int(c.rsplit("_", 1)[-1]) for c in df.columns
                  if c.startswith(HORIZON_KEY))


def _zone_a_flags(df: pd.DataFrame, h: int):
    y = df[f"y_true_{h}"].to_numpy(float)
    p = df[f"y_pred_{h}"].to_numpy(float)
    ok = np.isfinite(y) & np.isfinite(p) & (y > 0)
    flag = np.zeros(len(df), dtype=bool)
    flag[ok] = dts_zones(y[ok], p[ok]) == "A"
    return flag, ok


def resample(df: pd.DataFrame, horizon: int, per_subject: int = 25,
             n_boot: int = 2000, seed: int = 0) -> Dict[str, float]:
    """Zone-A% over ``n_boot`` draws of ``per_subject`` anchors per patient."""
    flag, ok = _zone_a_flags(df, horizon)
    subj = df["subject"].to_numpy()
    rng = np.random.default_rng(seed)

    # Group once; the draw is then pure indexing.
    order = np.argsort(subj, kind="mergesort")
    s_sorted = subj[order]
    bounds = np.flatnonzero(np.diff(s_sorted)) + 1
    groups = [g[ok[g]] for g in np.split(order, bounds)]
    groups = [g for g in groups if g.size]
    if not groups:
        return {}

    scores = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        picked = [g[rng.integers(0, g.size, min(per_subject, g.size))]
                  if g.size >= per_subject
                  else g
                  for g in groups]
        idx = np.concatenate(picked)
        scores[b] = flag[idx].mean() * 100.0
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return {"mean": float(scores.mean()), "sd": float(scores.std()),
            "lo95": float(lo), "hi95": float(hi),
            "n_per_draw": int(sum(min(g.size, per_subject) for g in groups)),
            "n_subjects": len(groups)}


def report(df: pd.DataFrame, per_subject: int = 25, n_boot: int = 2000,
           seed: int = 0) -> pd.DataFrame:
    rows = []
    for h in horizons_of(df):
        flag, ok = _zone_a_flags(df, h)
        full = float(flag[ok].mean() * 100) if ok.any() else float("nan")
        r = resample(df, h, per_subject, n_boot, seed)
        rows.append({"horizon": h, "n_full": int(ok.sum()),
                     "full_dts_A": round(full, 2),
                     "annual_like_mean": round(r.get("mean", float("nan")), 2),
                     "annual_like_sd": round(r.get("sd", float("nan")), 2),
                     "lo95": round(r.get("lo95", float("nan")), 2),
                     "hi95": round(r.get("hi95", float("nan")), 2),
                     "n_per_draw": r.get("n_per_draw")})
    return pd.DataFrame(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--split", default="internal_holdout")
    ap.add_argument("--per-subject", type=int, default=25,
                    help="anchors per patient per draw; 25 matches the annual "
                         "template's density")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    pred = a.run / f"predictions_{a.split}.parquet"
    if not pred.exists():
        raise SystemExit(f"{pred} not found")
    df = pd.read_parquet(pred)
    tab = report(df, a.per_subject, a.boot, a.seed)

    print(f"{a.run.name} [{a.split}]  {df['subject'].nunique()} patients, "
          f"{len(df):,} anchors")
    print(tab.to_string(index=False))
    mean_full = tab["full_dts_A"].mean()
    mean_like = tab["annual_like_mean"].mean()
    half = (tab["hi95"] - tab["lo95"]).mean() / 2
    print(f"\nmean DTS-A over horizons:  full {mean_full:.2f}%   "
          f"annual-like {mean_like:.2f}% (+/- {half:.2f} at 95% on a single "
          f"{a.per_subject}-per-patient draw)")

    out = a.out or (a.run / f"annual_estimate_{a.split}.csv")
    tab.to_csv(out, index=False)
    (out.with_suffix(".json")).write_text(json.dumps(
        {"run": a.run.name, "split": a.split, "per_subject": a.per_subject,
         "n_boot": a.boot, "mean_full_dts_A": round(float(mean_full), 3),
         "mean_annual_like_dts_A": round(float(mean_like), 3)}, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
