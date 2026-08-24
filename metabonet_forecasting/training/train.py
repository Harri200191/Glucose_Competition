"""Training entry point.

    python -m metabonet_forecasting.training.train \
        --config configs/experiment/e2_backbones.yaml \
        --set model.backbone.name=mamba2 data.context=288 seed=1 \
        --name mamba2_ctx288_s1

One run produces one directory under ``runs/`` containing the resolved config,
the provenance manifest, the training log, the best checkpoint, per-anchor test
predictions, and (unless ``--no-eval``) the scored metrics.

Evaluation is a separate module reading ``predictions_test.parquet``, so the
clinical strata, bootstrap intervals and plots can be re-run or extended without
retraining anything.
"""
from __future__ import annotations

import argparse
import os
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch

from ..common.config import REPO_ROOT, load_config, save_config
from ..common.logging import RunLogger, provenance
from ..common.seeding import seed_everything
from ..data.dataset import build_data, model_input_shapes
from ..models import build_model
from .loop import Trainer


def default_run_name(cfg) -> str:
    model = cfg.get_path("model.backbone.name", "model")
    fusion = cfg.get_path("model.fusion.name", "native")
    ctx = cfg.get_path("data.context", 288)
    seed = cfg.get_path("seed", 0)
    mods = "-".join(cfg.get_path("data.modalities", ["cgm"]))
    return f"{model}_{fusion}_ctx{ctx}_{mods}_s{seed}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="YAML config under configs/")
    ap.add_argument("--set", nargs="*", default=[], dest="sets",
                    help="dotted overrides, e.g. training.epochs=3 data.context=96")
    ap.add_argument("--name", default=None, help="run directory name")
    ap.add_argument("--runs-dir", default=str(REPO_ROOT / "runs"))
    ap.add_argument("--no-eval", action="store_true",
                    help="write predictions but skip scoring")
    ap.add_argument("--predict-val", action="store_true",
                    help="also write predictions for the validation split")
    live = ap.add_mutually_exclusive_group()
    live.add_argument("--exclude-live", dest="live_policy", action="store_const",
                      const="exclude",
                      help="remove every live-leaderboard patient from train and val. "
                           "The live set is drawn from the official test portion, but 125 "
                           "of its 279 patients also contribute earlier recordings to "
                           "training under the benchmark's temporal known-patient split; "
                           "this drops those recordings so the live score is a clean "
                           "novel-patient measurement. Test is left untouched.")
    live.add_argument("--include-live", dest="live_policy", action="store_const",
                      const="include",
                      help="the opposite: also move those patients' test-side anchors "
                           "INTO train, so the model trains on the leaderboard data "
                           "itself. This contaminates the internal test split for those "
                           "patients, so predictions_test stops being an honest estimate "
                           "-- only the submission means anything. Maximises the live "
                           "leaderboard, tells you nothing about the annual holdout.")
    ap.set_defaults(live_policy=None)
    ap.add_argument("--init-from", default=None,
                    help="path to a best.pt checkpoint to warm-start the model weights "
                         "from. The config must build an identical architecture; the "
                         "optimiser and schedule start fresh.")
    ap.add_argument("--gpu", default=None,
                    help="physical GPU index to train on, e.g. --gpu 6. Masks every other "
                         "device via CUDA_VISIBLE_DEVICES before torch initialises CUDA, so "
                         "no context is created on cards belonging to someone else. "
                         "Equivalent to CUDA_VISIBLE_DEVICES=6 in the environment; use "
                         "--set training.device=cpu to force CPU instead.")
    a = ap.parse_args(argv)

    # Must happen before the first CUDA call. torch may already be imported --
    # that is fine, the context is created lazily -- but it must not have touched
    # a device yet, so keep this at the top of main().
    if a.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)

    cfg = load_config(a.config, a.sets)
    if a.live_policy:
        cfg.set_path("data.split.live_policy", a.live_policy)
    seed = int(cfg.get("seed", 0))
    seed_everything(seed, deterministic=bool(cfg.get_path("training.deterministic", False)))

    run_name = a.name or default_run_name(cfg)
    run_dir = Path(a.runs_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log = RunLogger(run_dir)
    log.info(f"run {run_name} -> {run_dir}")

    t0 = time.time()
    bundle = build_data(cfg)
    shapes = model_input_shapes(bundle)
    log.info(f"data: {bundle.store.describe()}")
    log.info(f"windows: {json.dumps(bundle.summary()['windows'])}  "
             f"context={bundle.spec.context}  horizons={bundle.spec.horizons_min}")

    model = build_model(cfg, shapes)
    if a.init_from:
        blob = torch.load(a.init_from, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["state_dict"])
        log.info(f"warm-started weights from {a.init_from} "
                 f"(meta: {json.dumps(blob.get('meta', {}), default=str)})")
    log.info(f"model: {json.dumps(model.describe(), default=str)}")
    rf = model.receptive_field()
    if rf is not None and rf < bundle.spec.context:
        log.info(f"NOTE receptive field {rf} < context {bundle.spec.context}: this run "
                 f"cannot see the whole window")

    save_config(cfg, run_dir / "config.resolved.yaml")
    bundle.standardizer.save(run_dir / "standardizer.json")
    log.write_json("manifest.json", {
        "run_name": run_name,
        "config": cfg.to_dict(),
        "data": bundle.summary(),
        "model": model.describe(),
        "shapes": shapes,
        "store_manifest": {k: v for k, v in bundle.store.manifest.items()
                           if k in ("schema_version", "config", "n_rows", "n_anchors",
                                    "n_subjects", "stats")},
        "provenance": provenance(),
    })

    trainer = Trainer(model, bundle, cfg, run_dir, log)
    result = trainer.train()
    log.write_json("training_result.json", result)

    test_loader = bundle.loader("test", cfg, shuffle=False)
    preds = trainer.predict(test_loader)
    preds.to_parquet(run_dir / "predictions_test.parquet", index=False)
    log.info(f"wrote {len(preds):,} test predictions")
    if a.predict_val or bool(cfg.get_path("training.predict_val", False)):
        val_preds = trainer.predict(bundle.loader("val", cfg, shuffle=False))
        val_preds.to_parquet(run_dir / "predictions_val.parquet", index=False)

    if not a.no_eval:
        from ..evaluation.evaluate import score_run
        metrics = score_run(run_dir, bundle=bundle)
        for horizon in metrics["horizons_min"]:
            o = metrics["strata"]["all"][str(horizon)]["overall"]
            log.info(f"test h={horizon:>3}min  RMSE {o['rmse']:6.2f} mg/dL  "
                     f"MAE {o['mae']:6.2f}  MARD {o['mard']:5.2f}%  "
                     f"DTS-A {o['dts_A']:5.1f}%  n={o['n']:,}")

    log.info(f"done in {time.time() - t0:.0f}s")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
