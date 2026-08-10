"""Sweep the experiment matrix and collect results into one table.

    python -m exp.run_all --data build/main --models xgboost catboost mlp --horizons 30 60

Runs each (model, horizon, split) combination in a **subprocess**, so a CUDA OOM or a
segfault in one learner cannot take down the sweep. Results are gathered from the
per-run `result.json` files, so an interrupted sweep can be resumed with `--skip-done`.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from .config import BUILD_DIR, HORIZONS, RUNS_DIR
from .models import FAMILY, MODEL_NAMES

# A sensible default ladder: cheapest and most informative first, so a sweep that
# gets cut short has still produced the numbers you most need.
DEFAULT_MODELS = ["persistence", "linear_trend", "xgboost", "lightgbm", "histgb",
                  "catboost", "randomforest", "extratrees", "mlp", "resmlp",
                  "ft_transformer", "tcn", "gru", "tabpfn"]

# Per-model sample caps. These learners do not scale to 10 M rows in a sane wall
# clock, and the cap is part of the result, so it is recorded rather than hidden.
TRAIN_CAPS = {
    "randomforest": 3_000_000,
    "extratrees": 3_000_000,
    "tabpfn": None,          # TabPFN subsamples internally (bagged contexts)
}
EVAL_CAPS = {
    "randomforest": None,
    "tabpfn": 200_000,
}


def run_tag(model: str, horizon: int, split: str, fold: int,
            cohort: str | None) -> str:
    return (f"{model}_h{horizon}_{split}"
            f"{'_' + cohort if cohort else ''}"
            f"{'_f' + str(fold) if split == 'subject' else ''}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=BUILD_DIR / "main")
    p.add_argument("--models", nargs="*", default=DEFAULT_MODELS, choices=MODEL_NAMES)
    p.add_argument("--horizons", nargs="*", type=int, default=list(HORIZONS))
    p.add_argument("--split", default="official",
                   choices=["official", "subject", "cohort"])
    p.add_argument("--folds", nargs="*", type=int, default=[0])
    p.add_argument("--cohorts", nargs="*", default=[None],
                   help="for --split cohort: which cohorts to hold out")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--max-train", type=int, default=None,
                   help="global cap; per-model caps in TRAIN_CAPS still apply")
    p.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    p.add_argument("--skip-done", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    combos = []
    for m in a.models:
        for h in a.horizons:
            if a.split == "subject":
                for f in a.folds:
                    combos.append((m, h, f, None))
            elif a.split == "cohort":
                for c in a.cohorts:
                    combos.append((m, h, 0, c))
            else:
                combos.append((m, h, 0, None))

    print(f"{len(combos)} runs queued\n")
    t_all = time.time()
    for i, (m, h, f, c) in enumerate(combos, 1):
        tag = run_tag(m, h, a.split, f, c)
        out = a.runs_dir / tag
        if a.skip_done and (out / "result.json").exists():
            print(f"[{i}/{len(combos)}] skip {tag} (done)")
            continue

        cmd = [sys.executable, "-m", "exp.train", "--data", str(a.data),
               "--model", m, "--horizon", str(h), "--split", a.split,
               "--fold", str(f), "--gpu", str(a.gpu), "--seed", str(a.seed),
               "--out", str(out)]
        if c:
            cmd += ["--holdout-cohort", c]
        cap = TRAIN_CAPS.get(m, a.max_train)
        if cap:
            cmd += ["--max-train", str(cap)]
        ecap = EVAL_CAPS.get(m)
        if ecap:
            cmd += ["--max-eval", str(ecap)]

        print(f"\n[{i}/{len(combos)}] {tag}\n    {' '.join(cmd)}", flush=True)
        if a.dry_run:
            continue
        t0 = time.time()
        r = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent.parent))
        status = "ok" if r.returncode == 0 else f"FAILED rc={r.returncode}"
        print(f"    {status} in {time.time()-t0:.0f}s", flush=True)

    if not a.dry_run:
        collect(a.runs_dir)
    print(f"\nsweep finished in {(time.time()-t_all)/60:.1f} min")


def collect(runs_dir: Path = RUNS_DIR) -> pd.DataFrame:
    """Gather every result.json under `runs_dir` into one table."""
    rows = []
    for f in sorted(runs_dir.glob("*/result.json")):
        try:
            r = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        t = r.get("test", {})
        lb = r.get("test_leaderboard_subset", {})
        s = r.get("test_by_subject", {})
        rows.append({
            "model": r["model"], "family": r.get("family", FAMILY.get(r["model"], "?")),
            "horizon": r["horizon_min"], "split": r["split"],
            "cohort": r.get("holdout_cohort"), "fold": r.get("fold"),
            "n_train": r.get("n_train"), "n_test": r.get("n_test"),
            "MARD": t.get("MARD"), "RMSE": t.get("RMSE"), "MAE": t.get("MAE"),
            "clarke_A": t.get("clarke_A"), "clarke_AB": t.get("clarke_AB"),
            "parkes_A": t.get("parkes_A"), "parkes_AB": t.get("parkes_AB"),
            # present only once exp/dts_boundaries.json is supplied (see RUN.md §8)
            "dts_A": t.get("dts_A"), "dts_AB": t.get("dts_AB"),
            "MARD_hypo": t.get("MARD_hypo_lt70"),
            "MARD_subj_mean": s.get("MARD_subject_mean"),
            "MARD_subj_p90": s.get("MARD_subject_p90"),
            "MARD_leaderboard": lb.get("MARD"),
            "fit_s": r.get("fit_seconds"), "run": f.parent.name,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        print("no results found")
        return df
    df = df.sort_values(["split", "horizon", "MARD"])
    out = runs_dir / "results.csv"
    df.to_csv(out, index=False)

    print(f"\n{'='*100}\nRESULTS  ({len(df)} runs)  -> {out}\n{'='*100}")
    for (split, h), g in df.groupby(["split", "horizon"]):
        print(f"\n--- {split}, horizon {h} min "
              f"{'-'*(70 - len(str(split)))}")
        cols = ["model", "family", "n_train", "MARD", "RMSE", "clarke_A",
                "parkes_A", "MARD_subj_mean", "fit_s"]
        if g["dts_A"].notna().any():          # official grid enabled
            cols.insert(6, "dts_A")
        print(g[cols].to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    return df


if __name__ == "__main__":
    main()
