"""Fine-tune Chronos-2 on this corpus, with insulin as a future-known covariate.

    python -m metabonet_forecasting.fm.finetune --gpu 5 \
        --name c2ft_lora --mode lora --steps 5000 --batch 128 --lr 1e-5 \
        --set data.store=...

Training series are built per (training-anchor) segment, spanning only rows a
train anchor may legitimately see: from ``min(train anchor row) - context`` to
``max(train anchor row) + 24`` inside the segment. The store already guarantees
every anchor's targets lie on the anchor's own side of the official temporal
boundary, so this span never touches a test-period target. Validation series
come from the split's subject-disjoint val subjects, so early stopping is tuned
on people the test set never contains.

The result is saved with ``save_pretrained`` under ``runs/<name>/model``;
score it with the ordinary zero-shot harness pointed at that path:

    python -m metabonet_forecasting.fm.zeroshot --model chronos2 \
        --model-id runs/<name>/model --covariates insulin --name <name>_eval ...
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

PRED_STEPS = 24
COV_NAMES = ["ins_basal", "ins_bolus"]


def build_series(store, anchor_mask: np.ndarray, context: int,
                 max_series: Optional[int] = None,
                 min_len: int = 512) -> List[Dict]:
    """Per-segment dicts of {target, past_covariates} on the allowed row span."""
    a = store.anchors
    rows = a["row"].to_numpy()
    segs = a["seg"].to_numpy()
    seg_tab = store.segments.set_index("seg")
    cgm = int(store.channel_ids(["cgm"])[0])
    cov = {n: int(store.channel_ids([n])[0]) for n in COV_NAMES}

    picked = np.flatnonzero(anchor_mask)
    out: List[Dict] = []
    for seg in np.unique(segs[picked]):
        sel = picked[segs[picked] == seg]
        lo = max(int(rows[sel].min()) - context + 1, int(seg_tab.loc[seg, "start"]))
        hi = min(int(rows[sel].max()) + PRED_STEPS + 1, int(seg_tab.loc[seg, "end"]))
        if hi - lo < min_len:
            continue
        block = np.asarray(store.channels[lo:hi], dtype=np.float32)
        out.append({
            "target": block[:, cgm],
            "past_covariates": {n: np.nan_to_num(block[:, c])
                                for n, c in cov.items()},
        })
        if max_series and len(out) >= max_series:
            break
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--model-id", default="amazon/chronos-2")
    ap.add_argument("--mode", default="lora", choices=["lora", "full"])
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=None,
                    help="default: 1e-5 for lora, 1e-6 for full")
    ap.add_argument("--context-length", type=int, default=1024)
    ap.add_argument("--config", default="base")
    ap.add_argument("--set", nargs="*", default=[], dest="sets")
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--max-series", type=int, default=None)
    ap.add_argument("--max-val-series", type=int, default=256)
    a = ap.parse_args(argv)

    if a.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)

    from chronos import Chronos2Pipeline
    from chronos.chronos2.preprocess import from_list_of_dicts

    from ..common.config import REPO_ROOT, load_config, save_config
    from ..common.logging import RunLogger, provenance
    from ..data.dataset import build_data

    cfg = load_config(a.config, a.sets)
    run_dir = Path(a.runs_dir or REPO_ROOT / "runs") / a.name
    run_dir.mkdir(parents=True, exist_ok=True)
    log = RunLogger(run_dir)

    bundle = build_data(cfg)
    store = bundle.store
    context = int(cfg.get_path("data.context", 288))

    t0 = time.time()
    train_series = build_series(store, bundle.split["train"], context,
                                a.max_series)
    val_series = build_series(store, bundle.split["val"], context,
                              a.max_val_series)
    n_steps = sum(len(s["target"]) for s in train_series)
    log.info(f"{len(train_series)} train series ({n_steps:,} steps), "
             f"{len(val_series)} val series, built in {time.time() - t0:.0f}s")

    prepared_train = from_list_of_dicts(train_series, PRED_STEPS,
                                        known_covariates_names=COV_NAMES)
    prepared_val = from_list_of_dicts(val_series, PRED_STEPS,
                                      known_covariates_names=COV_NAMES)

    device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "x") != "" else "cpu"
    pipe = Chronos2Pipeline.from_pretrained(a.model_id, device_map=device)
    lr = a.lr if a.lr is not None else (1e-5 if a.mode == "lora" else 1e-6)
    log.info(f"fine-tuning {a.model_id}: mode={a.mode} steps={a.steps} "
             f"batch={a.batch} lr={lr} context={a.context_length}")

    t0 = time.time()
    tuned = pipe.fit(
        prepared_train, prediction_length=PRED_STEPS,
        validation_inputs=prepared_val,
        finetune_mode=a.mode, learning_rate=lr, num_steps=a.steps,
        batch_size=a.batch, context_length=a.context_length,
        output_dir=run_dir / "trainer")
    log.info(f"fine-tuned in {(time.time() - t0) / 60:.1f} min")

    model_dir = run_dir / "model"
    tuned.save_pretrained(model_dir)
    save_config(cfg, run_dir / "config.resolved.yaml")
    log.write_json("manifest.json", {
        "run_name": a.name, "fine_tune": True, "base_model": a.model_id,
        "mode": a.mode, "steps": a.steps, "batch": a.batch, "lr": lr,
        "context_length": a.context_length,
        "n_train_series": len(train_series), "n_val_series": len(val_series),
        "config": cfg.to_dict(), "provenance": provenance(),
    })
    log.info(f"saved {model_dir}")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
