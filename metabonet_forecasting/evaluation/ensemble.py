"""Prediction ensembling across runs.

Averages per-anchor forecasts from two or more run directories into a new
synthetic run, which the ordinary scorer then evaluates — no retraining, and a
free accuracy gain whenever the members' errors are less than perfectly
correlated.

    python -m metabonet_forecasting.evaluation.ensemble \
        runs/a runs/b runs/c --name ens_abc [--weights 2 1 1] [--split test]

Point forecasts are averaged (optionally weighted); quantile columns shared by
every member are averaged level-by-level (Vincent averaging), so the decision
layer can run on the ensemble too. Rows are aligned on ``anchor`` with an inner
join — a member scored on a thinned anchor set simply restricts the ensemble.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

HORIZONS = [30, 60, 90, 120]
META = ["subject", "cohort", "is_test", "known_patient", "cgm_now",
        "cgm_delta_30min", "min_since_carb", "min_since_bolus", "tod_min",
        "seen_in_train"]


def combine(frames: List[pd.DataFrame], weights: Optional[List[float]] = None,
            ) -> pd.DataFrame:
    w = np.asarray(weights if weights else [1.0] * len(frames), dtype=float)
    w = w / w.sum()

    keyed = [f.set_index("anchor").sort_index() for f in frames]
    common = keyed[0].index
    for f in keyed[1:]:
        common = common.intersection(f.index)
    keyed = [f.loc[common] for f in keyed]

    out = keyed[0][[c for c in META if c in keyed[0].columns]].copy()
    value_cols = [c for c in keyed[0].columns
                  if c.startswith(("y_pred_", "y_q"))
                  and all(c in f.columns for f in keyed)]
    for c in [c for c in keyed[0].columns if c.startswith("y_true_")]:
        out[c] = keyed[0][c]
    for c in value_cols:
        stack = np.stack([f[c].to_numpy(dtype=float) for f in keyed])
        # A member's NaN (no target there) stays NaN in the mean on purpose:
        # scoring masks those rows anyway.
        out[c] = np.einsum("m,mn->n", w, stack)
    return out.reset_index()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--runs-dir", type=Path, default=None,
                    help="where to write the ensemble run (default: parent of "
                         "the first member)")
    ap.add_argument("--weights", nargs="+", type=float, default=None)
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--no-eval", action="store_true")
    a = ap.parse_args(argv)
    if a.weights and len(a.weights) != len(a.runs):
        raise SystemExit("--weights must match the number of runs")

    fname = f"predictions_{a.split}.parquet"
    frames = [pd.read_parquet(r / fname) for r in a.runs]
    out = combine(frames, a.weights)

    out_dir = (a.runs_dir or a.runs[0].parent) / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_dir / fname, index=False)
    (out_dir / "ensemble.json").write_text(json.dumps(
        {"members": [str(r) for r in a.runs],
         "weights": a.weights or [1.0] * len(a.runs),
         "n_rows": int(len(out)), "split": a.split}, indent=1))
    print(f"wrote {out_dir} ({len(out):,} rows, {len(frames)} members)")

    if not a.no_eval and a.split == "test":
        from .evaluate import score_run
        metrics = score_run(out_dir)
        for h in metrics["horizons_min"]:
            o = metrics["strata"]["all"][str(h)]["overall"]
            print(f"h={h:>3}  RMSE {o['rmse']:6.2f}  MAE {o['mae']:6.2f}  "
                  f"DTS-A {o['dts_A']:5.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
