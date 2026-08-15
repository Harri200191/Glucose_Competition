"""Collect finished runs into one table (and one markdown report).

    python -m metabonet_forecasting.evaluation.aggregate runs --out results
    python -m metabonet_forecasting.evaluation.aggregate runs --compare \
        mamba2_film_ctx288_cgm-insulin-carbs_s0 mamba2_raw_concat_ctx288_cgm-insulin-carbs_s0 \
        --horizon 60

Every row carries the experimental factors it varied -- backbone, fusion,
context, modalities, seed -- so the ablation tables of stages C-F are group-bys
rather than bookkeeping.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..common.logging import get_console_logger
from ..data import constants as K
from .bootstrap import paired_bootstrap, seed_variability

LOG = get_console_logger("aggregate")

FACTOR_COLUMNS = ["backbone", "fusion", "context", "modalities", "seed", "split",
                  "event_features", "masks", "missing_policy", "source_identity"]


def run_factors(run_dir: Path) -> Dict[str, object]:
    """Read a run's experimental factors out of its manifest."""
    manifest = run_dir / "manifest.json"
    if not manifest.exists():
        return {"run": run_dir.name}
    with open(manifest) as fh:
        blob = json.load(fh)
    cfg = blob.get("config", {})
    data = cfg.get("data", {})
    model = cfg.get("model", {})
    info = blob.get("model", {})
    # The model's own description wins over the config: baseline runs inherit a
    # model: block from base.yaml that they never used, and reading the config
    # first would file every baseline under whatever backbone base.yaml names.
    return {
        "run": run_dir.name,
        "backbone": info.get("backbone") or model.get("backbone", {}).get("name"),
        "fusion": info.get("fusion") or model.get("fusion", {}).get("name"),
        "context": data.get("context"),
        "modalities": "+".join(data.get("modalities", [])),
        "seed": cfg.get("seed"),
        "split": data.get("split", {}).get("scheme"),
        "event_features": data.get("use_event_features"),
        "masks": data.get("use_masks"),
        "missing_policy": data.get("missing_policy"),
        "source_identity": data.get("use_source_identity"),
        "parameters": info.get("parameters"),
        "receptive_field": info.get("receptive_field"),
        "best_val_rmse": _best_val(run_dir),
    }


def _best_val(run_dir: Path) -> Optional[float]:
    path = run_dir / "training_result.json"
    if not path.exists():
        return None
    with open(path) as fh:
        return json.load(fh).get("best_val_rmse")


def collect(runs_dir: Path, pattern: str = "*") -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for run_dir in sorted(Path(runs_dir).glob(pattern)):
        csv = run_dir / "metrics.csv"
        if not csv.is_file():
            continue
        df = pd.read_csv(csv)
        for key, value in run_factors(run_dir).items():
            if key != "run":
                df[key] = value
        if len(df):
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no scored runs under {runs_dir} matching {pattern!r}; "
                                f"run evaluation first")
    columns = list(dict.fromkeys(c for f in frames for c in f.columns))
    return pd.concat([f.reindex(columns=columns) for f in frames], ignore_index=True)


def primary_table(df: pd.DataFrame, stratum: str = "all",
                  horizons: Sequence[int] = K.ALL_HORIZONS_MIN) -> pd.DataFrame:
    """The headline table: overall metrics per run and horizon, one row per run."""
    sel = df[(df["stratum"] == stratum) & (df["slice"] == "overall")
             & (df["horizon"].isin(list(horizons)))]
    wide = sel.pivot_table(index=["run"] + [c for c in FACTOR_COLUMNS if c in sel.columns],
                           columns="horizon", values=["rmse", "mard", "dts_A"],
                           aggfunc="first")
    wide.columns = [f"{m}_{h}" for m, h in wide.columns]
    return wide.reset_index().sort_values(by=[c for c in ("rmse_60", "rmse_30")
                                              if c in wide.columns])


def seed_summary(df: pd.DataFrame, stratum: str = "all", horizon: int = 60,
                 metric: str = "rmse") -> pd.DataFrame:
    """Mean and spread across seeds for every distinct configuration."""
    sel = df[(df["stratum"] == stratum) & (df["slice"] == "overall")
             & (df["horizon"] == horizon)]
    keys = [c for c in FACTOR_COLUMNS if c in sel.columns and c != "seed"]
    rows = []
    for values, group in sel.groupby(keys, dropna=False):
        stat = seed_variability(group[metric].to_numpy())
        rows.append(dict(zip(keys, values if isinstance(values, tuple) else (values,)),
                         **{f"{metric}_{k}": v for k, v in stat.items()}))
    return pd.DataFrame(rows).sort_values(f"{metric}_mean")


def to_markdown(df: pd.DataFrame, float_fmt: str = "{:.2f}") -> str:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype.kind == "f":
            out[col] = out[col].map(lambda v: float_fmt.format(v) if pd.notna(v) else "")
    header = "| " + " | ".join(str(c) for c in out.columns) + " |"
    sep = "| " + " | ".join("---" for _ in out.columns) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |"
            for row in out.itertuples(index=False)]
    return "\n".join([header, sep] + body)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs_dir", nargs="?", default="runs")
    ap.add_argument("--pattern", default="*")
    ap.add_argument("--out", default="results")
    ap.add_argument("--stratum", default="all")
    ap.add_argument("--horizons",
                    default=",".join(str(h) for h in K.ALL_HORIZONS_MIN))
    ap.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"), default=None,
                    help="paired patient-level bootstrap between two runs")
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--metric", default="rmse")
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args(argv)

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if a.compare:
        da = pd.read_parquet(Path(a.runs_dir) / a.compare[0] / "predictions_test.parquet")
        db = pd.read_parquet(Path(a.runs_dir) / a.compare[1] / "predictions_test.parquet")
        res = paired_bootstrap(da, db, a.horizon, a.metric, a.n_boot)
        res["A"], res["B"] = a.compare
        LOG.info(json.dumps(res, indent=2, default=float))
        with open(out_dir / f"compare_{a.compare[0]}__vs__{a.compare[1]}.json", "w") as fh:
            json.dump(res, fh, indent=2, default=float)
        return 0

    df = collect(Path(a.runs_dir), a.pattern)
    df.to_csv(out_dir / "all_runs.csv", index=False)

    horizons = [int(h) for h in a.horizons.split(",")]
    table = primary_table(df, a.stratum, horizons)
    table.to_csv(out_dir / "primary_table.csv", index=False)
    seeds = seed_summary(df, a.stratum, horizons[min(1, len(horizons) - 1)])
    seeds.to_csv(out_dir / "seed_summary.csv", index=False)

    report = [f"# Results ({len(df['run'].unique())} runs, stratum = {a.stratum})", "",
              "## Primary table", "", to_markdown(table), "",
              "## Seed variability", "", to_markdown(seeds), ""]
    (out_dir / "REPORT.md").write_text("\n".join(report))
    LOG.info(f"wrote {out_dir}/all_runs.csv, primary_table.csv, seed_summary.csv, REPORT.md")
    print(to_markdown(table))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
