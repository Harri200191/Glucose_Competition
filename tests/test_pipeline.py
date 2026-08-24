"""End-to-end: config -> data -> model -> training -> predictions -> metrics.

Runs the real entry points on the synthetic store, on CPU, for a handful of
steps. It is not a test of accuracy; it is a test that the pieces still fit
together and that a run leaves behind everything a result needs to be traceable.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from metabonet_forecasting.common.config import apply_overrides, load_config
from metabonet_forecasting.evaluation.aggregate import collect, primary_table
from metabonet_forecasting.training.train import main as train_main


def _train(tmp_path, store_path, name, extra=None):
    args = [
        "--config", "base",
        "--name", name,
        "--runs-dir", str(tmp_path / "runs"),
        "--set",
        f"data.store={store_path}",
        "data.context=48",
        "data.horizons_min=[30, 60]",
        "data.sampling.max_train_windows=300",
        "data.sampling.max_val_windows=150",
        "data.sampling.max_test_windows=150",
        "data.sampling.standardizer_sample=300",
        "data.loader.batch_size=16",
        "data.loader.num_workers=0",
        "model.d_model=32",
        "model.backbone.name=tcn",
        "model.backbone.n_levels=2",
        "model.backbone.auto_levels=false",
        "training.epochs=1",
        "training.max_steps_per_epoch=4",
        "training.device=cpu",
        "training.amp=false",
        "training.warmup_steps=1",
        "training.log_every=0",
    ] + (extra or [])
    assert train_main(args) == 0
    return tmp_path / "runs" / name


def test_training_run_produces_a_complete_traceable_artifact(tmp_path, store_path):
    run_dir = _train(tmp_path, store_path, "e2e_tcn")

    for artefact in ("config.resolved.yaml", "manifest.json", "standardizer.json",
                     "train_log.csv", "best.pt", "predictions_test.parquet",
                     "metrics.json", "metrics.csv", "training_result.json"):
        assert (run_dir / artefact).exists(), f"missing {artefact}"

    manifest = json.loads((run_dir / "manifest.json").read_text())
    prov = manifest["provenance"]
    assert prov["git_commit"] and prov["versions"]["torch"]
    assert manifest["model"]["parameters"] > 0
    assert manifest["data"]["context"] == 48

    preds = pd.read_parquet(run_dir / "predictions_test.parquet")
    assert {"anchor", "subject", "cohort", "y_true_30", "y_pred_30",
            "y_true_60", "y_pred_60"} <= set(preds.columns)
    assert len(preds) > 0
    assert preds["anchor"].is_unique

    metrics = json.loads((run_dir / "metrics.json").read_text())
    overall = metrics["strata"]["all"]["30"]["overall"]
    assert overall["n"] > 0 and overall["rmse"] > 0
    assert "dts_A" in overall
    assert "event:postprandial" in metrics["strata"]["all"]["30"]


def test_two_runs_aggregate_into_one_comparable_table(tmp_path, store_path):
    _train(tmp_path, store_path, "agg_a")
    _train(tmp_path, store_path, "agg_b", ["seed=1"])
    df = collect(tmp_path / "runs")
    table = primary_table(df, horizons=(30, 60))
    assert len(table) == 2
    assert {"backbone", "fusion", "context", "modalities", "seed"} <= set(table.columns)
    assert "rmse_30" in table.columns and table["rmse_30"].notna().all()


def test_baseline_runner_emits_the_same_schema(tmp_path, store_path):
    from metabonet_forecasting.baselines.run_baselines import run

    cfg = load_config("base", [
        f"data.store={store_path}", "data.context=48", "data.horizons_min=[30, 60]",
        "data.sampling.max_train_windows=300", "data.sampling.max_test_windows=200",
        "data.sampling.standardizer_sample=300", "data.loader.num_workers=0",
    ])
    run_dir = run(cfg, "persistence", tmp_path / "runs" / "baseline_persistence")
    preds = pd.read_parquet(run_dir / "predictions_test.parquet")
    # persistence: the forecast is exactly the last reading, at every horizon
    valid = preds["y_pred_30"].notna()
    assert (preds.loc[valid, "y_pred_30"] == preds.loc[valid, "cgm_now"]).all()
    assert (run_dir / "metrics.json").exists()


def test_config_includes_and_overrides_resolve_in_order():
    cfg = load_config("model/mamba2", ["data.context=96", "model.backbone.n_layers=2"])
    assert cfg.get_path("model.backbone.name") == "mamba2"
    assert cfg.get_path("model.backbone.n_layers") == 2      # override wins
    assert cfg.get_path("data.context") == 96
    assert cfg.get_path("training.epochs") == 20             # inherited from base
    assert "configs/base.yaml" in cfg["_config_files"][0]
    assert cfg["_overrides"] == ["data.context=96", "model.backbone.n_layers=2"]


def test_experiment_configs_all_load():
    from metabonet_forecasting.common.config import CONFIG_ROOT
    for path in sorted((CONFIG_ROOT / "experiment").glob("*.yaml")):
        cfg = load_config(f"experiment/{path.stem}")
        assert cfg.get_path("data.store")
        assert cfg.get_path("data.horizons_min")
