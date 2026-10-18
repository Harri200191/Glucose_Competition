"""Fine-tune Timestral on this corpus, through the exact inference function.

    python -m metabonet_forecasting.fm.finetune_timestral --gpu 5 \
        --name tslft_futins --covariates insulin --steps 8000 --batch 32 \
        --set data.store=...

Then score it through the ordinary zero-shot harness, pointed at the saved
checkpoint (same covariate condition it was trained under):

    python -m metabonet_forecasting.fm.zeroshot --model timestral \
        --model-id runs/<name>/model --covariates insulin \
        --split internal_holdout --name <name>_eval ...

WHY THIS EXISTS
---------------
Zero-shot, Timestral-209M with insulin as a known-future covariate is the
strongest borrowed forecaster this repo has scored. Matched on the first 4,096
internal-holdout anchors (2 patients -- a plumbing sample, not a ranking;
approximate +/-20% zone A):

    Timestral-209M zero-shot + insulin     81.45
    TimesFM 3.0    zero-shot + insulin     79.34    (ia_tfm3_cov)
    TimesFM 3.0    fine-tuned              82.32    (tfm3ft_eval)
    fm_ctrl        (1M params, trained)    85.68

Fine-tuning bought TimesFM 3.0 +2.58 on the full holdout. The same operation
on a stronger starting point is the obvious next arm, and the only one that can
say whether Timestral closes the gap to the in-house models.

NOT OBVIOUS, AND VERIFIED (mistralai/timestral@a534bdb, V100 32 GB)
------------------------------------------------------------------
``TimestralForecaster.forecast`` is ``@torch.inference_mode()``, so training
through it produces no graph. ``functools.wraps`` keeps the original as
``forecast.__wrapped__``; this calls that. Forward is BIT-IDENTICAL to
``forecast()`` (max |diff| 0.0) and gradients reach 203/203 parameter tensors.
If a release drops the wrapper this raises instead of training nothing.

Because it is the same function, symmetric averaging (forecast of -context,
reflected and averaged) and the non-negative clip are inside the loss. That
doubles the cost of a step and removes every train/serve gap: the checkpoint is
trained on exactly what ``zeroshot.py`` will evaluate.

Missing CGM stays NaN. Timestral masks it natively, so training sees the input
inference sees -- the TimesFM harness had to interpolate to achieve the same.

Variate attention runs over the WHOLE packed batch behind a dense group mask,
so memory grows roughly with batch squared. Full fine-tune of the 209M model
with insulin covariates: B=32 11.3 GiB, 0.56 s/step; B=64 18.8 GiB, 1.04 s/step;
B=256 out of memory on 32 GiB. The same cost makes the covariate arms of the
zero-shot harness ~8x slower than CGM-only.

fp32 throughout: upstream validates inference in fp32 only, and the V100s here
have no native bf16.

LOSS: pinball over all 21 levels, in LOG space by default. Quantiles commute
with the log, so the same quantiles are being estimated, but the loss becomes
relative error -- the quantity DTS zone A scores (+/-20% relative), and the
reason ``data.target_mode=log_ratio`` is the competition-aligned arm.
``--loss-space raw`` is the mg/dL pinball the TimesFM 3.0 fine-tune used.

MODE: ``full`` by default. Insulin reaches the target only through the
``time_variate`` blocks, one in every four layers across the stack; tuning the
last few layers (TimesFM's ``partial``) would leave all but one of them frozen.
``partial`` is here for the comparison.

Training rows come from ``finetune.build_series``, which restricts each
segment to the span a training anchor may see; val from the subject-disjoint
val split. With ``exclude_whole_source`` (default) no IOBP2 patient is in
either, so the internal holdout stays unseen.

LICENCE: code Apache-2.0; weights under a modified MIT licence (see each HF
model card).
"""
from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .finetune import PRED_STEPS, build_series
from .finetune_timesfm import _windows

GRID_MIN = 5


def _pinball(torch, pred, true, mask, q, space: str):
    """pred [B,H,Q], true [B,H], mask [B,H] -> masked mean pinball loss.

    In log space both sides are floored at 1 mg/dL first: the non-negative clip
    can put a low quantile at exactly 0, and masked targets were zero-filled.
    """
    if space == "log":
        pred = torch.log(pred.clamp_min(1.0))
        true = torch.log(true.clamp_min(1.0))
    err = true.unsqueeze(-1) - pred
    loss = torch.maximum(q * err, (q - 1.0) * err).mean(-1)
    m = mask.float()
    return (loss * m).sum() / m.sum().clamp_min(1.0)


def _batch(win, idx, covariates: Optional[str], context: int):
    """Window tuples -> (context, past, known, future, future mask) arrays.

    ``past``/``known`` follow the zero-shot harness's conditions exactly:
    ``insulin`` = known over context + horizon, ``insulin_past`` = context only.
    """
    ctx = np.stack([win[i][0] for i in idx])                 # [B, T], NaN kept
    fut = np.stack([win[i][2] for i in idx])                 # [B, H]
    past = known = None
    if covariates is not None:
        cov = np.stack([win[i][1] for i in idx])             # [B, C, T + H]
        if covariates == "insulin":
            known = cov
        else:
            past = np.ascontiguousarray(cov[..., :context])
    return ctx, past, known, np.nan_to_num(fut), np.isfinite(fut)


def _save(model, path: Path) -> None:
    """Write the two files ``TimestralForecaster.from_pretrained`` reads.

    ``rope.inv_freq`` is a non-persistent buffer, rebuilt on load, so the state
    dict holds exactly the keys the strict loader expects.
    """
    from safetensors.torch import save_file
    path.mkdir(parents=True, exist_ok=True)
    model.config.save(path / "config.json")
    save_file({k: v.detach().contiguous().cpu()
               for k, v in model.state_dict().items()},
              str(path / "model.safetensors"))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--model-id", default="mistralai/Timestral-209M")
    ap.add_argument("--covariates", default="insulin",
                    choices=["none", "insulin", "insulin_past"],
                    help="must match the --covariates the checkpoint is "
                         "evaluated under; insulin = future-known (_futins)")
    ap.add_argument("--mode", default="full", choices=["full", "partial"],
                    help="partial = last --unfrozen layers + quantile head")
    ap.add_argument("--unfrozen", type=int, default=4)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=32,
                    help="memory ~ batch^2 with covariates; 32 = 11.3 GiB")
    ap.add_argument("--lr", type=float, default=None,
                    help="default: 1e-5 full, 1e-4 partial")
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--loss-space", default="log", choices=["log", "raw"])
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--config", default="base")
    ap.add_argument("--set", nargs="*", default=[], dest="sets")
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--max-series", type=int, default=None)
    ap.add_argument("--max-val-series", type=int, default=256)
    ap.add_argument("--per-series", type=int, default=128,
                    help="windows sampled from each training segment span")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    covariates = None if a.covariates == "none" else a.covariates

    if a.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)

    import torch
    from timestral import TimestralForecaster

    from ..common.config import REPO_ROOT, load_config, save_config
    from ..common.logging import RunLogger, provenance
    from ..data.dataset import build_data
    from ..evaluation.metrics import dts_zone_a_mask

    torch.manual_seed(a.seed)
    cfg = load_config(a.config, a.sets)
    run_dir = Path(a.runs_dir or REPO_ROOT / "runs") / a.name
    run_dir.mkdir(parents=True, exist_ok=True)
    log = RunLogger(run_dir)

    bundle = build_data(cfg)
    store = bundle.store
    context = int(cfg.get_path("data.context", 288))
    horizons = [int(h) for h in cfg.get_path("data.horizons_min", [30, 60, 90, 120])]
    h_idx = [h // GRID_MIN - 1 for h in horizons]
    rng = np.random.default_rng(a.seed)

    t0 = time.time()
    train_series = build_series(store, bundle.split["train"], context, a.max_series)
    val_series = build_series(store, bundle.split["val"], context, a.max_val_series)
    train_win = _windows(train_series, context, PRED_STEPS, rng, a.per_series)
    val_win = _windows(val_series, context, PRED_STEPS,
                       np.random.default_rng(a.seed + 1), 8)
    del train_series, val_series
    log.info(f"{len(train_win):,} train / {len(val_win):,} val windows "
             f"in {time.time() - t0:.0f}s")
    if not train_win or not val_win:
        log.info("no usable windows; nothing to train on")
        return 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fc = TimestralForecaster.from_pretrained(a.model_id, device=device)
    model = fc.model
    raw_forecast = getattr(TimestralForecaster.forecast, "__wrapped__", None)
    if raw_forecast is None:
        raise RuntimeError(
            "TimestralForecaster.forecast is no longer a wrapped inference_mode "
            "function; training through it would build no graph. Drive "
            "_forecast_raw directly instead.")
    if covariates == "insulin" and not fc.supports_known_covariates:
        raise SystemExit(f"{a.model_id} cannot condition on known-future covariates")

    if a.mode == "partial":
        layers = model.transformer.layers
        if a.unfrozen > len(layers):
            raise ValueError(f"--unfrozen {a.unfrozen} exceeds {len(layers)} layers")
        for p in model.parameters():
            p.requires_grad_(False)
        for mod in [*layers[-a.unfrozen:], model.transformer.out_norm, model.head]:
            for p in mod.parameters():
                p.requires_grad_(True)
    train_p = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in train_p)
    log.info(f"mode={a.mode}: {n_train:,} of "
             f"{sum(p.numel() for p in model.parameters()):,} params trainable")

    lr = a.lr if a.lr is not None else (1e-5 if a.mode == "full" else 1e-4)
    opt = torch.optim.AdamW(train_p, lr=lr, weight_decay=0.01)
    # Linear warmup, cosine to 10%: the first updates land on a model already
    # good at this task (zero-shot 81 on the smoke slice), and a cold full-LR
    # step is the cheapest way to throw that away.
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (
        (s + 1) / a.warmup if s < a.warmup else
        0.1 + 0.45 * (1 + math.cos(math.pi * (s - a.warmup)
                                   / max(1, a.steps - a.warmup)))))
    q = torch.tensor(fc.quantile_levels, dtype=torch.float32, device=device)
    med = fc.median_index
    log.info(f"fine-tuning {a.model_id}: covariates={a.covariates} "
             f"steps={a.steps} batch={a.batch} lr={lr} loss={a.loss_space}")

    def forward(win, idx):
        ctx, past, known, fut, fmask = _batch(win, idx, covariates, context)
        out = raw_forecast(fc, ctx, PRED_STEPS, past_covariates=past,
                           known_covariates=known).quantiles[:, 0]   # [B,H,Q]
        return (out, torch.from_numpy(fut).to(device),
                torch.from_numpy(fmask).to(device))

    def evaluate():
        tot, n, hits, cnt = 0.0, 0, np.zeros(len(h_idx)), np.zeros(len(h_idx))
        with torch.no_grad():
            for i in range(0, len(val_win), a.batch):
                idx = list(range(i, min(i + a.batch, len(val_win))))
                out, fut, fmask = forward(val_win, idx)
                tot += float(_pinball(torch, out, fut, fmask, q, a.loss_space)) * len(idx)
                n += len(idx)
                y = np.where(fmask.cpu().numpy(), fut.cpu().numpy(), np.nan)[:, h_idx]
                p = out[..., med].cpu().numpy()[:, h_idx]
                hits += dts_zone_a_mask(y, p).sum(0)
                cnt += np.isfinite(y).sum(0)
        return tot / max(n, 1), 100 * hits / np.maximum(cnt, 1)

    def fmt(z):
        return " ".join(f"{h}:{v:.1f}" for h, v in zip(horizons, z)) + f" avg {z.mean():.2f}"

    # The base model is the step-0 checkpoint: a run that never beats it leaves
    # the released weights behind, never a worse last state, and its eval run
    # then reads as exactly the zero-shot number.
    best, z0 = evaluate()
    _save(model, run_dir / "model")
    bad, n_applied, n_skipped, t0 = 0, 0, 0, time.time()
    log.info(f"  step 0  val pinball {best:.5f}  zoneA {fmt(z0)}")
    history = [{"step": 0, "val_pinball": best, "val_zone_a": z0.tolist()}]
    order = rng.permutation(len(train_win))
    pos = 0
    for step in range(1, a.steps + 1):
        if pos + a.batch > len(order):
            order = rng.permutation(len(train_win)); pos = 0
        idx = order[pos: pos + a.batch]; pos += a.batch
        out, fut, fmask = forward(train_win, idx)
        loss = _pinball(torch, out, fut, fmask, q, a.loss_space)
        if not torch.isfinite(loss):
            n_skipped += 1; opt.zero_grad(set_to_none=True); sched.step(); continue
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(train_p, 1.0)
        if not torch.isfinite(gnorm):
            n_skipped += 1; opt.zero_grad(set_to_none=True); sched.step(); continue
        opt.step(); opt.zero_grad(set_to_none=True); sched.step(); n_applied += 1

        if step % a.eval_every == 0:
            # A run where every step was skipped would otherwise walk the whole
            # schedule and save the untouched base model as a "fine-tune".
            if n_applied == 0:
                raise RuntimeError(f"{step} steps, 0 applied ({n_skipped} non-finite); "
                                   "the model is not training")
            v, z = evaluate()
            mark = ""
            if v < best - 1e-6:
                best, bad, mark = v, 0, "  *best*"
                _save(model, run_dir / "model")
            else:
                bad += 1
            history.append({"step": step, "val_pinball": v, "val_zone_a": z.tolist()})
            log.info(f"  step {step}/{a.steps}  train {float(loss):.5f}  val {v:.5f}  "
                     f"zoneA {fmt(z)}{mark}  [{n_applied} applied, {n_skipped} skipped]"
                     f"  ({(time.time() - t0) / 60:.1f} min)")
            if bad >= a.patience:
                log.info(f"early stop at step {step}; best val {best:.5f}")
                break

    if len(history) > 1 and min(h["val_pinball"] for h in history[1:]) >= history[0]["val_pinball"]:
        log.info("validation never improved on the base model; the saved "
                 "checkpoint IS the released model -- read the eval as 'no gain'")
    save_config(cfg, run_dir / "config.resolved.yaml")
    log.write_json("manifest.json", {
        "run_name": a.name, "fine_tune": True, "base_model": a.model_id,
        "covariates": a.covariates, "mode": a.mode,
        "unfrozen": a.unfrozen if a.mode == "partial" else None,
        "steps": a.steps, "batch": a.batch, "lr": lr, "warmup": a.warmup,
        "loss_space": a.loss_space, "trainable_params": int(n_train),
        "best_val_pinball": float(best), "history": history,
        "n_train_windows": len(train_win), "n_val_windows": len(val_win),
        "config": cfg.to_dict(), "provenance": provenance(),
    })
    log.info(f"saved {run_dir / 'model'}  (best val pinball {best:.5f})")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
