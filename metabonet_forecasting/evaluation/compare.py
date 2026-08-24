"""Head-to-head model comparison with paired, patient-level uncertainty.

    python -m metabonet_forecasting.evaluation.compare runs/a runs/b runs/c \
        --split internal_holdout --reference a --margin 2.0

Two models trained on the same data differ by less than the noise between
patients. Comparing their *averages* therefore says almost nothing: the spread
you see is mostly which patients happened to be hard, not which model is
better. So everything here is **paired** -- for each patient, model A's score
minus model B's score on that same patient's own anchors -- and the uncertainty
comes from a **cluster bootstrap over patients**, because anchors inside one
patient are nowhere near independent.

Runs are inner-joined on ``anchor`` first, so every model is judged on exactly
the same rows. A member scored on a thinned anchor set simply restricts the
comparison rather than silently biasing it.

Equivalence, not just significance
----------------------------------
"Not significantly different" usually means "not enough data to tell", which is
not a finding. When the claim is that architectures are interchangeable, the
question has to be inverted: can we rule out a difference big enough to matter?
``--margin`` is that threshold, in percentage points of DTS zone A, and it must
be chosen **before** looking at the numbers.

    equivalent    the whole CI sits inside +/- margin -> interchangeable
    better/worse  the whole CI sits on one side of zero
    inconclusive  the CI is too wide to say either -> collect more, do not spin

``direction`` and ``verdict`` are reported separately on purpose: a CI of
[+0.2, +1.5] against a 2-point margin is *both* reliably better and practically
equivalent, and collapsing that into one word loses the point.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .metrics import dts_zones, mae, rmse

HORIZONS = (30, 60, 90, 120)


def load_aligned(runs: Sequence[Path], split: str) -> Dict[str, pd.DataFrame]:
    """Every run's predictions, restricted to the anchors they all share."""
    frames = {}
    for r in runs:
        path = Path(r) / f"predictions_{split}.parquet"
        if not path.exists():
            print(f"  skip {Path(r).name}: no predictions_{split}.parquet")
            continue
        frames[Path(r).name] = pd.read_parquet(path).set_index("anchor").sort_index()
    if not frames:
        raise SystemExit(f"no run had predictions_{split}.parquet")
    common = None
    for f in frames.values():
        common = f.index if common is None else common.intersection(f.index)
    return {k: v.loc[common] for k, v in frames.items()}


def per_patient_zone_a(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Zone-A hits and scorable count per patient, for one horizon."""
    y = df[f"y_true_{horizon}"].to_numpy(float)
    p = df[f"y_pred_{horizon}"].to_numpy(float)
    ok = np.isfinite(y) & np.isfinite(p) & (y > 0)
    hit = np.zeros(len(df), dtype=float)
    hit[ok] = (dts_zones(y[ok], p[ok]) == "A").astype(float)
    out = pd.DataFrame({"subject": df["subject"].to_numpy(),
                        "hit": hit, "n": ok.astype(float)})
    return out.groupby("subject").sum()


def leaderboard(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in frames.items():
        row = {"run": name}
        za = []
        for h in HORIZONS:
            if f"y_true_{h}" not in df.columns:
                continue
            y = df[f"y_true_{h}"].to_numpy(float)
            p = df[f"y_pred_{h}"].to_numpy(float)
            ok = np.isfinite(y) & np.isfinite(p) & (y > 0)
            a = float((dts_zones(y[ok], p[ok]) == "A").mean() * 100) if ok.any() else np.nan
            row[f"A@{h}"] = round(a, 2)
            row[f"rmse@{h}"] = round(rmse(y[ok], p[ok]), 2) if ok.any() else np.nan
            za.append(a)
        row["A_mean"] = round(float(np.nanmean(za)), 2) if za else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("A_mean", ascending=False)


def paired(frames: Dict[str, pd.DataFrame], reference: str, margin: float,
           n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Paired patient-level difference vs ``reference``, per horizon."""
    rng = np.random.default_rng(seed)
    ref_df = frames[reference]
    rows = []
    for h in HORIZONS:
        if f"y_true_{h}" not in ref_df.columns:
            continue
        ref_pp = per_patient_zone_a(ref_df, h)
        for name, df in frames.items():
            if name == reference:
                continue
            pp = per_patient_zone_a(df, h)
            joined = ref_pp.join(pp, lsuffix="_ref", rsuffix="_x", how="inner")
            joined = joined[joined["n_ref"] > 0]
            # Patient-level rates, then the paired difference for that patient.
            a_ref = joined["hit_ref"] / joined["n_ref"] * 100
            a_x = joined["hit_x"] / joined["n_x"].replace(0, np.nan) * 100
            keep = np.isfinite(a_ref) & np.isfinite(a_x)
            d = (a_x - a_ref)[keep].to_numpy()
            w = joined["n_ref"][keep].to_numpy()
            if d.size < 3:
                continue
            # Cluster bootstrap: resample PATIENTS, not anchors.
            idx = rng.integers(0, d.size, size=(n_boot, d.size))
            boot = np.average(d[idx], weights=w[idx], axis=1)
            point = float(np.average(d, weights=w))
            lo, hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
            if hi < margin and lo > -margin:
                verdict = "equivalent"
            elif lo > 0 or hi < 0:
                verdict = "different"
            else:
                verdict = "inconclusive"
            direction = "better" if lo > 0 else "worse" if hi < 0 else "tie"
            rows.append({"horizon": h, "run": name, "vs": reference,
                         "n_patients": int(d.size),
                         "delta_A": round(point, 3), "lo95": round(lo, 3),
                         "hi95": round(hi, 3), "verdict": verdict,
                         "direction": direction})
    return pd.DataFrame(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--split", default="internal_holdout")
    ap.add_argument("--reference", default=None,
                    help="run name everything is compared against "
                         "(default: the first run given)")
    ap.add_argument("--margin", type=float, default=2.0,
                    help="equivalence margin in DTS zone-A points; choose it "
                         "BEFORE looking at results")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    frames = load_aligned(a.runs, a.split)
    n_anchor = len(next(iter(frames.values())))
    n_subj = next(iter(frames.values()))["subject"].nunique()
    print(f"{len(frames)} runs on {n_anchor:,} shared anchors, {n_subj} patients "
          f"[{a.split}]\n")

    board = leaderboard(frames)
    print("== leaderboard ==")
    print(board.to_string(index=False))

    ref = a.reference or list(frames)[0]
    if ref not in frames:
        raise SystemExit(f"reference {ref!r} not among {list(frames)}")
    pair = paired(frames, ref, a.margin, a.boot, a.seed)
    print(f"\n== paired vs {ref}, cluster-bootstrapped over patients, "
          f"equivalence margin +/-{a.margin} pts ==")
    print(pair.to_string(index=False) if len(pair) else "  (nothing to compare)")

    out = a.out or Path("results") / f"compare_{a.split}"
    out.mkdir(parents=True, exist_ok=True)
    board.to_csv(out / "leaderboard.csv", index=False)
    pair.to_csv(out / "paired.csv", index=False)
    (out / "compare.json").write_text(json.dumps(
        {"split": a.split, "reference": ref, "margin": a.margin,
         "n_boot": a.boot, "n_anchors": int(n_anchor), "n_patients": int(n_subj),
         "runs": list(frames)}, indent=1))

    with open(out / "REPORT.md", "w") as fh:
        fh.write(f"# Model comparison ({a.split})\n\n")
        fh.write(f"{len(frames)} runs, {n_anchor:,} shared anchors, "
                 f"{n_subj} patients. Reference: `{ref}`. "
                 f"Equivalence margin +/-{a.margin} DTS zone-A points.\n\n")
        fh.write("## Leaderboard\n\n" + board.to_markdown(index=False) + "\n\n")
        fh.write("## Paired differences\n\n")
        fh.write("Positive `delta_A` means the run beats the reference on the "
                 "same patients. `verdict` is about *practical* size, "
                 "`direction` about sign.\n\n")
        fh.write((pair.to_markdown(index=False) if len(pair) else "_none_") + "\n")
    print(f"\nwrote {out}/REPORT.md, leaderboard.csv, paired.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
