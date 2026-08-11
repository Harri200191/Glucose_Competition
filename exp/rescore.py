"""Recompute metrics for finished runs, without retraining.

    python -m exp.rescore --data build/main

Every run saves `predictions.npy` and `test_index.npy` alongside `result.json`, so
when the metric set changes — a new error grid, a new breakdown — the scores can be
regenerated from the stored predictions in seconds instead of re-fitting models for
hours.

Only the metric blocks are rewritten. Timings, model notes and the data manifest
copy are preserved, and the previous scores are kept under `result.json →
previous_test` so a change in the metric definition stays auditable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .build_dataset import load_dataset
from .config import BUILD_DIR, HORIZONS, RUNS_DIR
from .metrics import evaluate, evaluate_by_subject
from .splits import leaderboard_mask


def rescore_run(run_dir: Path, Y: np.ndarray, meta: pd.DataFrame) -> dict | None:
    res_path = run_dir / "result.json"
    pred_path = run_dir / "predictions.npy"
    idx_path = run_dir / "test_index.npy"
    if not (res_path.exists() and pred_path.exists() and idx_path.exists()):
        return None

    result = json.loads(res_path.read_text())
    pred = np.load(pred_path)
    te_idx = np.load(idx_path)
    h_index = list(HORIZONS).index(result["horizon_min"])
    y = Y[te_idx, h_index].astype(np.float64)

    if len(pred) != len(te_idx):
        print(f"  skip {run_dir.name}: {len(pred)} predictions vs "
              f"{len(te_idx)} test rows -- dataset changed since this run?")
        return None

    result.setdefault("previous_test", result.get("test"))
    result["test"] = evaluate(y, pred)
    result["test_by_subject"] = evaluate_by_subject(
        y, pred, meta["subject"].to_numpy()[te_idx])

    lb = leaderboard_mask(meta)[te_idx]
    if lb.any():
        result["test_leaderboard_subset"] = evaluate(y[lb], pred[lb])

    coh = meta["source_file"].to_numpy()[te_idx]
    per = {}
    for c in pd.unique(coh):
        m = coh == c
        if m.sum() >= 500:
            per[str(c)] = {k: round(v, 4) for k, v in evaluate(y[m], pred[m]).items()
                           if k in ("n", "MARD", "RMSE", "dts_A", "clarke_A", "parkes_A")}
    result["test_by_cohort"] = per
    result["rescored_at"] = pd.Timestamp.now().isoformat(timespec="seconds")

    res_path.write_text(json.dumps(result, indent=2, default=str))
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=BUILD_DIR / "main")
    p.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    a = p.parse_args()

    _, Y, meta, _ = load_dataset(a.data, mmap=True)
    runs = sorted(d for d in a.runs_dir.iterdir()
                  if d.is_dir() and (d / "result.json").exists())
    print(f"rescoring {len(runs)} run(s) against {a.data}\n")

    done = 0
    for d in runs:
        r = rescore_run(d, Y, meta)
        if r:
            done += 1
            t, prev = r["test"], r.get("previous_test") or {}
            delta = ""
            if prev.get("MARD") is not None:
                delta = f"  (MARD was {prev['MARD']:.3f})"
            print(f"  {d.name:<42s} MARD {t['MARD']:7.3f}  "
                  f"DTS-A {t.get('dts_A', float('nan')):6.2f}%{delta}")
    print(f"\nrescored {done}/{len(runs)}")

    from .run_all import collect
    collect(a.runs_dir)


if __name__ == "__main__":
    main()
