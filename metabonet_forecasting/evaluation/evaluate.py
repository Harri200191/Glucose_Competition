"""Score a finished run.

    python -m metabonet_forecasting.evaluation.evaluate runs/mamba2_film_ctx288_s0
    python -m metabonet_forecasting.evaluation.evaluate runs/* --bootstrap 1000

Reads ``predictions_test.parquet`` and writes ``metrics.json`` next to it. Nothing
here needs the model, the GPU or the store, so evaluation can be re-run, extended
or fixed without retraining -- which is the point of storing per-anchor forecasts
rather than summary numbers.

Structure of ``metrics.json``:

    run, horizons
    strata: {all|novel_patient|known_patient|leaderboard:
                {horizon: {overall|glycemia:*|event:*|trend:*: {metric: value}}}}
    by_source: {horizon: {source_file: {...}}}
    bootstrap: {stratum: {horizon: {metric: {point, lo, hi}}}}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..common.logging import get_console_logger
from ..data import constants as K
from . import clinical_metrics as CM
from .bootstrap import bootstrap_metric

LOG = get_console_logger("evaluate")


def prediction_strata(df: pd.DataFrame, cohort_names: List[str]) -> Dict[str, np.ndarray]:
    """The evaluation strata of plan 7.7, computed from the predictions frame.

    ``seen_in_train`` is written by the run itself and reflects that run's split,
    so it is preferred over the corpus's ``subject_split_across_traintest`` flag;
    the flag is the fallback for prediction files written before it existed.
    """
    column = "seen_in_train" if "seen_in_train" in df.columns else "known_patient"
    known = df[column].to_numpy().astype(bool)
    excluded = [i for i, name in enumerate(cohort_names)
                if name in K.LEADERBOARD_EXCLUDED_COHORTS]
    cohort = df["cohort"].to_numpy()
    return {
        "all": np.ones(len(df), dtype=bool),
        "novel_patient": ~known,
        "known_patient": known,
        "leaderboard": ~np.isin(cohort, excluded),
    }


def score_run(run_dir, bundle=None, n_boot: int = 0, seed: int = 0,
              min_stratum_n: int = 50) -> Dict:
    run_dir = Path(run_dir)
    pred_path = run_dir / "predictions_test.parquet"
    if not pred_path.exists():
        raise FileNotFoundError(f"{pred_path} not found; train the run first")
    df = pd.read_parquet(pred_path)

    horizons = sorted(int(c.split("_")[-1]) for c in df.columns if c.startswith("y_true_"))
    cohort_names = _cohort_names(run_dir, bundle)

    strata = prediction_strata(df, cohort_names)
    out: Dict = {"run": run_dir.name, "horizons_min": horizons,
                 "n_predictions": int(len(df)),
                 "n_subjects": int(df["subject"].nunique()),
                 "strata": {}, "by_source": {}, "bootstrap": {}}

    for stratum, mask in strata.items():
        sub = df.loc[mask]
        if len(sub) < min_stratum_n:
            out["strata"][stratum] = {"n": int(len(sub))}
            continue
        out["strata"][stratum] = {
            str(h): CM.evaluate_horizon(sub, h, min_n=min_stratum_n) for h in horizons}

    for h in horizons:
        out["by_source"][str(h)] = CM.evaluate_by_source(df, h, cohort_names)

    if n_boot:
        for stratum, mask in strata.items():
            sub = df.loc[mask]
            if len(sub) < min_stratum_n:
                continue
            block: Dict[str, Dict[str, Dict[str, float]]] = {}
            for h in horizons:
                y = sub[f"y_true_{h}"].to_numpy(dtype=np.float64)
                p = sub[f"y_pred_{h}"].to_numpy(dtype=np.float64)
                s = sub["subject"].to_numpy()
                block[str(h)] = {m: bootstrap_metric(y, p, s, m, n_boot, seed)
                                 for m in ("rmse", "mae", "mard")}
            out["bootstrap"][stratum] = block

    with open(run_dir / "metrics.json", "w") as fh:
        json.dump(out, fh, indent=2, default=float)

    # A flat one-row-per-(stratum, horizon) CSV so aggregation never has to parse
    # the nested JSON.
    rows = []
    for stratum, per_h in out["strata"].items():
        if "n" in per_h:
            continue
        for h, block in per_h.items():
            for name, values in block.items():
                rows.append(dict(run=run_dir.name, stratum=stratum, horizon=int(h),
                                 slice=name, **{k: v for k, v in values.items()}))
    pd.DataFrame(rows).to_csv(run_dir / "metrics.csv", index=False)
    return out


def _cohort_names(run_dir: Path, bundle=None) -> List[str]:
    if bundle is not None:
        return bundle.store.cohort_names()
    manifest = run_dir / "manifest.json"
    if manifest.exists():
        with open(manifest) as fh:
            blob = json.load(fh)
        store = blob.get("config", {}).get("data", {}).get("store")
        if store and (Path(store) / "manifest.json").exists():
            with open(Path(store) / "manifest.json") as fh:
                return list(json.load(fh)["levels"].get("source_file", []))
    return []


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run directories")
    ap.add_argument("--bootstrap", type=int, default=0,
                    help="patient-level bootstrap replicates (0 = skip)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    for run in a.runs:
        run_dir = Path(run)
        if not (run_dir / "predictions_test.parquet").exists():
            LOG.info(f"skip {run_dir} (no predictions)")
            continue
        m = score_run(run_dir, n_boot=a.bootstrap, seed=a.seed)
        for h in m["horizons_min"]:
            o = m["strata"]["all"][str(h)]["overall"]
            LOG.info(f"{run_dir.name} h={h:>3}min  RMSE {o['rmse']:6.2f}  "
                     f"MAE {o['mae']:6.2f}  MARD {o['mard']:5.2f}%  "
                     f"DTS-A {o['dts_A']:5.1f}%  n={o['n']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
