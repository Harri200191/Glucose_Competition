"""Score a submission parquet against the live leaderboard's public answer key.

    python scripts/analysis/score_live.py artifacts/submission/live_b72_ens.parquet

The live board ships its targets, so a submission can be scored locally before it
is uploaded -- same rows, same metric, no guessing. Rows the model left NaN are
reported separately rather than silently dropped: a submission that skips a third
of the board scores well per-row and badly overall, and the coverage column is
what makes that visible.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from metabonet_forecasting.evaluation.metrics import dts_zones

H = (30, 60, 90, 120)
TARGETS = "toolkit/data/live_leaderboard/targets.parquet"


def score(path: Path, targets: pd.DataFrame) -> dict:
    sub = pd.read_parquet(path)
    key = ["id", "source_file", "date"]
    m = targets.merge(sub, on=key, how="left", suffixes=("", "_p"))
    row = {"file": path.name, "rows": len(m)}
    vals = []
    for h in H:
        yt = m[f"target_{h}"].to_numpy(float)
        yp = m[f"pred_{h}"].to_numpy(float)
        scorable = np.isfinite(yt)
        both = scorable & np.isfinite(yp)
        # Denominator is every row the board can score, not just the ones we
        # answered -- an unanswered row is a miss, not an abstention.
        a = ((dts_zones(yt, yp) == "A") & both).sum()
        v = 100 * a / max(scorable.sum(), 1)
        row[f"A@{h}"] = round(v, 2)
        row[f"cov@{h}"] = round(100 * both.sum() / max(scorable.sum(), 1), 1)
        vals.append(v)
    row["mean"] = round(float(np.mean(vals)), 3)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--targets", default=TARGETS)
    a = ap.parse_args()
    t = pd.read_parquet(a.targets)
    rows = [score(f, t) for f in a.files if f.exists()]
    if not rows:
        print("no files found", file=sys.stderr)
        return 1
    df = pd.DataFrame(rows).sort_values("mean", ascending=False)
    pd.set_option("display.width", 220)
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
