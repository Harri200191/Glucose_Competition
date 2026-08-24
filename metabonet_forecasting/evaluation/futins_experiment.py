"""Does future insulin leak the answer? Test it where the mechanism differs.

    python -m metabonet_forecasting.evaluation.futins_experiment \
        --causal runs/fx_causal --futins runs/fx_futins --split test

The claim under test
--------------------
Feeding a model the insulin delivered *after* the anchor is not a free feature:
for a closed-loop (AID) patient the pump computes that dose **from** the glucose
the model is being asked to predict, so the dose is a label in disguise. For
someone injecting manually the dose is a genuine prior plan, and the same
feature should help much less.

So the experiment is not "does future insulin help" -- it obviously does. It is:

    gain_AID  -  gain_nonAID   >>  0

That *interaction* is the finding. A large gap is evidence the feature works by
leakage rather than by physiology, and it is measured here with a cluster
bootstrap over patients in each group independently.

Design
------
Both models are trained on **all** patients with an otherwise identical recipe,
and the split happens at evaluation. That is deliberate: training separate
models per modality would confound the comparison with training-set size (264
non-AID patients against 1,621), and the question is about the feature, not
about how much data each subgroup can support. ``run_futins_experiment.sh`` can
additionally train modality-restricted models as a robustness check.

Modality comes from the store's subject table, not from the predictions file.
**22 subjects carry code -1 (unknown) and are dropped**, not folded into either
group -- note that ``levels[-1]`` silently resolves to "MDI" in Python, which is
exactly the kind of quiet miscount this guards against.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .compare import md_table
from .metrics import dts_zones

HORIZONS = (30, 60, 90, 120)
UNKNOWN = -1


def modality_of(store, subjects: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Per-anchor modality label: 'AID', 'non-AID' or 'unknown'."""
    levels = store.manifest["levels"]["insulin_delivery_modality"]
    code = store.subjects.set_index("subject")["modality"].reindex(subjects).to_numpy()
    aid = levels.index("AID")
    out = np.where(code == UNKNOWN, "unknown",
                   np.where(code == aid, "AID", "non-AID"))
    return out, levels


def _patient_zone_a(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    y = df[f"y_true_{horizon}"].to_numpy(float)
    p = df[f"y_pred_{horizon}"].to_numpy(float)
    ok = np.isfinite(y) & np.isfinite(p) & (y > 0)
    hit = np.zeros(len(df))
    hit[ok] = (dts_zones(y[ok], p[ok]) == "A").astype(float)
    return pd.DataFrame({"subject": df["subject"].to_numpy(),
                         "group": df["_group"].to_numpy(),
                         "hit": hit, "n": ok.astype(float)}
                        ).groupby(["group", "subject"]).sum()


def _boot_mean(d: np.ndarray, w: np.ndarray, rng, n_boot: int) -> np.ndarray:
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    return np.average(d[idx], weights=w[idx], axis=1)


def run(causal: pd.DataFrame, futins: pd.DataFrame, n_boot: int = 2000,
        seed: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-group gain from future insulin, and the AID vs non-AID interaction."""
    rng = np.random.default_rng(seed)
    rows, inter = [], []
    for h in HORIZONS:
        if f"y_true_{h}" not in causal.columns:
            continue
        c = _patient_zone_a(causal, h)
        f = _patient_zone_a(futins, h)
        j = c.join(f, lsuffix="_c", rsuffix="_f", how="inner")
        j = j[(j["n_c"] > 0) & (j["n_f"] > 0)]
        rate_c = j["hit_c"] / j["n_c"] * 100
        rate_f = j["hit_f"] / j["n_f"] * 100

        draws = {}
        for group in ("AID", "non-AID"):
            if group not in j.index.get_level_values("group"):
                continue
            m = j.index.get_level_values("group") == group
            d = (rate_f - rate_c)[m].to_numpy()
            w = j["n_c"][m].to_numpy()
            keep = np.isfinite(d)
            d, w = d[keep], w[keep]
            if d.size < 3:
                continue
            boot = _boot_mean(d, w, rng, n_boot)
            draws[group] = boot
            lo, hi = np.percentile(boot, [2.5, 97.5])
            rows.append({
                "horizon": h, "group": group, "n_patients": int(d.size),
                "causal_A": round(float(np.average(rate_c[m][keep], weights=w)), 2),
                "futins_A": round(float(np.average(rate_f[m][keep], weights=w)), 2),
                "gain": round(float(np.average(d, weights=w)), 3),
                "lo95": round(float(lo), 3), "hi95": round(float(hi), 3)})
        if {"AID", "non-AID"} <= set(draws):
            diff = draws["AID"] - draws["non-AID"]
            lo, hi = np.percentile(diff, [2.5, 97.5])
            inter.append({
                "horizon": h,
                "gain_AID_minus_nonAID": round(float(diff.mean()), 3),
                "lo95": round(float(lo), 3), "hi95": round(float(hi), 3),
                "verdict": ("AID leaks more" if lo > 0 else
                            "non-AID gains more" if hi < 0 else
                            "no detectable difference")})
    return pd.DataFrame(rows), pd.DataFrame(inter)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--causal", type=Path, required=True)
    ap.add_argument("--futins", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--store", default=None,
                    help="default: the store named in the causal run's config")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results/futins_experiment"))
    a = ap.parse_args(argv)

    frames = {}
    for tag, run_dir in (("causal", a.causal), ("futins", a.futins)):
        p = run_dir / f"predictions_{a.split}.parquet"
        if not p.exists():
            raise SystemExit(f"{p} not found")
        frames[tag] = pd.read_parquet(p).set_index("anchor").sort_index()
    common = frames["causal"].index.intersection(frames["futins"].index)
    frames = {k: v.loc[common] for k, v in frames.items()}

    root = a.store
    if root is None:
        import yaml
        cfg = a.causal / "config.resolved.yaml"
        root = (yaml.safe_load(cfg.read_text()) or {}).get("data", {}).get("store")
    from ..data.store import SequenceStore
    store = SequenceStore(root)

    grp, _ = modality_of(store, frames["causal"]["subject"].to_numpy())
    for v in frames.values():
        v["_group"] = grp
    n_unknown = int((grp == "unknown").sum())
    frames = {k: v[v["_group"] != "unknown"] for k, v in frames.items()}

    counts = frames["causal"].groupby("_group")["subject"].nunique().to_dict()
    print(f"future-insulin experiment [{a.split}]  {len(common):,} shared anchors")
    print(f"  groups: " + ", ".join(f"{k} {v} patients" for k, v in counts.items()))
    if n_unknown:
        print(f"  dropped {n_unknown:,} anchors of patients with unknown modality")

    per_group, interaction = run(frames["causal"], frames["futins"], a.boot, a.seed)
    print("\n== zone A with and without future insulin, paired over patients ==")
    print(per_group.to_string(index=False))
    print("\n== the finding: is the gain bigger for AID? ==")
    print(interaction.to_string(index=False) if len(interaction) else "  (n/a)")

    a.out.mkdir(parents=True, exist_ok=True)
    per_group.to_csv(a.out / "per_group.csv", index=False)
    interaction.to_csv(a.out / "interaction.csv", index=False)
    (a.out / "meta.json").write_text(json.dumps(
        {"causal": str(a.causal), "futins": str(a.futins), "split": a.split,
         "n_anchors": int(len(common)), "patients": counts,
         "dropped_unknown_anchors": n_unknown, "n_boot": a.boot}, indent=1))
    with open(a.out / "REPORT.md", "w") as fh:
        fh.write("# Future insulin: AID versus non-AID\n\n")
        fh.write(f"Split `{a.split}`, {len(common):,} shared anchors. "
                 + ", ".join(f"**{k}**: {v} patients" for k, v in counts.items())
                 + f". {n_unknown:,} anchors of unknown-modality patients dropped.\n\n")
        fh.write("Both models were trained on all patients with an identical "
                 "recipe; only `data.future_modalities` differs. Gains are "
                 "paired per patient and cluster-bootstrapped over patients.\n\n")
        fh.write("## Gain from future insulin\n\n" + md_table(per_group) + "\n\n")
        fh.write("## Interaction — the actual result\n\n")
        fh.write("`gain_AID - gain_nonAID` > 0 means the feature helps closed-loop "
                 "patients more, which is what leakage predicts: the pump computes "
                 "the dose from the glucose being forecast.\n\n")
        fh.write(md_table(interaction) + "\n")
    print(f"\nwrote {a.out}/REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
