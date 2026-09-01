"""Fine-tune TimesFM 3.0 on this corpus, with insulin as a future-known covariate.

    python -m metabonet_forecasting.fm.finetune_timesfm --gpu 5 \
        --name tfm3ft --mode partial --steps 4000 --batch 8 \
        --set data.store=...

Then score it through the ordinary zero-shot harness, pointed at the saved
checkpoint:

    python -m metabonet_forecasting.fm.zeroshot --model timesfm3 \
        --model-id runs/<name>/model --covariates insulin \
        --split internal_holdout --name <name>_eval ...

WHY THIS EXISTS
---------------
``finetune.py`` fine-tunes Chronos-2 and is hardcoded to it -- its imports, its
preprocessing and its covariate format are all Chronos-shaped. TimesFM 3.0
landed in August 2026 and needs its own loop, so this is that loop. The half
that is genuinely reusable, ``build_series`` -- which restricts training rows to
spans a *training* anchor may legitimately see, so no test-period target is ever
visible -- is imported from there rather than re-derived.

WHAT IT IS FIXING
-----------------
Zero-shot on the internal holdout, TimesFM 3.0 with insulin covariates against a
1M-parameter model trained on this data:

    h= 30   91.2%  vs 93.6%   -2.4
    h=120   60.9%  vs 69.3%   -8.4
    mean    74.7%  vs 80.6%   -5.9

The gap grows monotonically with horizon. At 30 minutes the task is close to
local extrapolation and a general forecaster nearly matches; at 120 minutes it
needs insulin pharmacokinetics and meal responses, which no amount of general
time-series pretraining supplies. Fine-tuning is the operation that would
install exactly that, which is why the shape of the gap argues for trying it.

TWO IMPLEMENTATION NOTES THAT ARE NOT OBVIOUS
---------------------------------------------
``TimesFM3Torch.decode`` is decorated ``@torch.no_grad()``, so training through
it silently produces no gradients. ``functools.wraps`` preserves the original,
so this uses ``decode.__wrapped__``; verified to backprop into 445 parameter
groups. If a future release drops the wrapper this raises rather than training a
model that cannot learn.

``decode`` returns ``(B, V, H, Q)`` for ALL variates -- target plus covariates.
The target is variate 0; the covariate forecasts are discarded. Taking the mean
over V would train the model to predict insulin, which is not the task.

LICENCE: the TimesFM 3.0 weights are under timesfm-non-commercial-license-v1.0,
which forbids commercial and production use. This is a research comparator.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from .finetune import COV_NAMES, PRED_STEPS, build_series


def _windows(series: List[dict], context: int, horizon: int, rng,
             per_series: int) -> List[tuple]:
    """Sample (target, covariates, future-truth) windows out of the long spans."""
    out = []
    need = context + horizon
    for s in series:
        tgt = s["target"]
        if len(tgt) < need:
            continue
        cov = np.stack([s["past_covariates"][n] for n in COV_NAMES])  # [C, L]
        starts = rng.integers(0, len(tgt) - need + 1, size=per_series)
        for st in starts:
            ctx = tgt[st: st + context]
            fut = tgt[st + context: st + need]
            # A window whose future is entirely missing teaches nothing.
            if not np.isfinite(fut).any() or not np.isfinite(ctx).any():
                continue
            out.append((ctx.astype(np.float32),
                        cov[:, st: st + need].astype(np.float32),
                        fut.astype(np.float32)))
    return out


def _batch(win, idx, device, torch, interp):
    """One training batch, filled EXACTLY the way inference fills it.

    ``predict_batch`` linear-interpolates missing values before decoding. Filling
    with zeros here instead would put literal 0 mg/dL readings into the context
    -- a value no sensor produces -- and would train the model on inputs it never
    sees at inference. Same helper, same result.

    NO mask is passed to decode, deliberately. The mask argument reads True as
    "masked out", the opposite of this repo's convention: handing it an all-True
    validity mask collapses every forecast to 0.0. Verified against
    ``predict_batch`` on identical inputs -- omitting the mask reproduces it to
    0.000, any mask does not. The context is interpolated and full length, so
    there is nothing to mask anyway.
    """
    ctx_np = np.stack([win[i][0] for i in idx])
    ctx_f = interp(ctx_np)                        # [B, T], NaNs interpolated
    ctx = torch.from_numpy(np.ascontiguousarray(ctx_f)).to(device)
    cov = torch.from_numpy(np.stack([win[i][1] for i in idx])).to(device)
    fut_np = np.stack([win[i][2] for i in idx])
    fmask = torch.from_numpy(np.isfinite(fut_np)).to(device)
    fut = torch.from_numpy(np.nan_to_num(fut_np)).to(device)   # masked out in the loss
    return ctx.unsqueeze(1), cov, fut, fmask


def _save(model, path, revin_was) -> None:
    """Write the checkpoint with the shipped inference settings restored."""
    prev = getattr(model, "use_iterative_cpm_revin", None)
    if revin_was is not None:
        model.use_iterative_cpm_revin = revin_was
    try:
        model.save_pretrained(path)
    finally:
        if revin_was is not None:
            model.use_iterative_cpm_revin = prev


def _pinball(torch, pred, true, mask, q):
    """pred [B,H,Q], true [B,H], mask [B,H] -> masked mean pinball loss.

    ``torch`` is passed in rather than taken from module scope: this module
    imports torch inside ``main`` (after CUDA_VISIBLE_DEVICES is set), so a
    module-level reference only exists once a run has started and the helper
    cannot be used or tested on its own.
    """
    err = true.unsqueeze(-1) - pred
    loss = torch.maximum(q * err, (q - 1.0) * err).mean(-1)
    m = mask.float()
    return (loss * m).sum() / m.sum().clamp_min(1.0)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--model-id", default="google/timesfm-3.0-pytorch")
    ap.add_argument("--mode", default="partial", choices=["partial", "full"],
                    help="partial = train only the last --unfrozen blocks plus "
                         "the output head; full = every parameter")
    ap.add_argument("--unfrozen", type=int, default=4,
                    help="partial mode: how many trailing transformer blocks")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=None,
                    help="default: 1e-4 for partial, 1e-5 for full")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--config", default="base")
    ap.add_argument("--set", nargs="*", default=[], dest="sets")
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--max-series", type=int, default=None)
    ap.add_argument("--max-val-series", type=int, default=128)
    ap.add_argument("--per-series", type=int, default=24,
                    help="windows sampled from each segment span")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    if a.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)

    import torch
    import timesfm
    from timesfm3.timesfm3_forecaster import linear_interpolation

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
    rng = np.random.default_rng(a.seed)

    t0 = time.time()
    train_series = build_series(store, bundle.split["train"], context, a.max_series)
    val_series = build_series(store, bundle.split["val"], context, a.max_val_series)
    train_win = _windows(train_series, context, PRED_STEPS, rng, a.per_series)
    val_win = _windows(val_series, context, PRED_STEPS,
                       np.random.default_rng(a.seed + 1), 8)
    log.info(f"{len(train_series)} train / {len(val_series)} val series -> "
             f"{len(train_win):,} train / {len(val_win):,} val windows "
             f"in {time.time() - t0:.0f}s")
    if not train_win or not val_win:
        log.info("no usable windows; nothing to train on")
        return 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fc = timesfm.TimesFM3Forecaster.from_pretrained(a.model_id, device=device)
    model = fc.model

    # --- the fix that makes this trainable at all -------------------------
    # decode() computes std = sqrt(clamp_min(var, 0.0)) inside its iterative CPM
    # RevIN. d/dx sqrt(x) is 1/(2 sqrt(x)), i.e. infinite at x = 0, and the
    # variance really is 0 on flat or padded stretches. Under @torch.no_grad
    # that never matters, which is exactly why the decorator is there; in
    # backward it makes EVERY gradient non-finite -- all 445 parameter tensors,
    # in fp32 and bf16 alike.
    #
    # Turning the flag off for training is free: on this path the forward output
    # is BIT-IDENTICAL with it on or off (max |diff| 0.00e+00 over real windows),
    # while gradients go from 445/445 non-finite to 0/445. It is restored before
    # the checkpoint is written so inference behaves exactly as shipped.
    revin_was = getattr(model, "use_iterative_cpm_revin", None)
    if revin_was:
        model.use_iterative_cpm_revin = False
        log.info("disabled use_iterative_cpm_revin for training "
                 "(sqrt-at-zero makes its backward non-finite; forward is "
                 "bit-identical, and it is restored before saving)")

    raw_decode = getattr(type(model).decode, "__wrapped__", None)
    if raw_decode is None:
        raise RuntimeError(
            "TimesFM3Torch.decode is no longer a wrapped @torch.no_grad function; "
            "training through it would produce no gradients. Update this script "
            "to drive model.forward() with patched inputs instead.")

    qs = list(getattr(model, "quantiles", None)
              or [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    q = torch.tensor(qs, dtype=torch.float32, device=device)

    if a.mode == "partial":
        # Addressed by name, not by a class-name heuristic. The obvious filter
        # -- class names ending in "block"/"layer" -- matches exactly ONE module
        # in this architecture (a ResidualBlock) and silently trains the wrong
        # 68M parameters, so the structure is asserted instead.
        try:
            layers = model.transformer_stack.layers      # 20x MixingTransformer
            head = model.output_head
        except AttributeError as e:
            raise RuntimeError(
                "TimesFM3Torch no longer exposes transformer_stack.layers / "
                f"output_head ({e}); fix the partial-freeze selection rather "
                "than falling back to a name heuristic.") from e
        if a.unfrozen > len(layers):
            raise ValueError(f"--unfrozen {a.unfrozen} exceeds {len(layers)} layers")
        for p in model.parameters():
            p.requires_grad_(False)
        for blk in layers[-a.unfrozen:]:
            for p in blk.parameters():
                p.requires_grad_(True)
        for p in head.parameters():
            p.requires_grad_(True)
        log.info(f"partial: last {a.unfrozen}/{len(layers)} transformer layers "
                 f"+ output_head")
    train_p = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in train_p)
    log.info(f"mode={a.mode}: {n_train:,} of "
             f"{sum(p.numel() for p in model.parameters()):,} params trainable")
    if n_train == 0:
        raise RuntimeError("partial mode froze everything; raise --unfrozen")

    lr = a.lr if a.lr is not None else (1e-4 if a.mode == "partial" else 1e-5)
    opt = torch.optim.AdamW(train_p, lr=lr, weight_decay=0.01)
    # fp16 autocast makes the gradients here non-finite, which GradScaler then
    # skips -- every step, silently, so the model never moves and validation is
    # bit-identical forever. bf16 has fp32's exponent range and needs no scaler;
    # where it is unavailable, run in fp32 rather than pretend to train.
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else None
    log.info(f"fine-tuning {a.model_id}: steps={a.steps} batch={a.batch} lr={lr} "
             f"precision={'bf16' if use_bf16 else 'fp32'}")

    def evaluate() -> float:
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, min(len(val_win), 512), a.batch):
                idx = list(range(i, min(i + a.batch, len(val_win))))
                ctx, cov, fut, fmask = _batch(val_win, idx, device, torch,
                                              linear_interpolation)
                with torch.amp.autocast(device, dtype=amp_dtype,
                                        enabled=use_bf16):
                    out = raw_decode(model, target=ctx, horizon=PRED_STEPS,
                                     past_future_covariates=cov)
                l = _pinball(torch, out[:, 0].float().sort(-1).values, fut, fmask, q)
                tot += float(l) * len(idx); n += len(idx)
        model.train()
        return tot / max(n, 1)

    n_applied = n_skipped = 0

    def _maybe_report(step: int) -> None:
        """Fail loudly on an eval boundary that no optimiser step ever reached.

        Without this, a run where every step is skipped for a non-finite loss
        never reaches the evaluation block at all -- it just walks the whole
        schedule and saves the untouched base model, looking exactly like a
        successful fine-tune. That happened; hence the guard.
        """
        if step % a.eval_every == 0 and n_applied == 0:
            raise RuntimeError(
                f"{step} steps in and not one optimiser update has been applied "
                f"({n_skipped} skipped for non-finite loss or gradient). The "
                "model is not training -- check precision, learning rate, and "
                "that decode is being called the way inference calls it.")

    best, bad, t0 = evaluate(), 0, time.time()
    log.info(f"  step 0  val {best:.4f}")
    model.train()
    order = rng.permutation(len(train_win))
    pos = 0
    for step in range(1, a.steps + 1):
        if pos + a.batch > len(order):
            order = rng.permutation(len(train_win)); pos = 0
        idx = order[pos: pos + a.batch]; pos += a.batch
        ctx, cov, fut, fmask = _batch(train_win, idx, device, torch,
                                      linear_interpolation)
        with torch.amp.autocast(device, dtype=amp_dtype, enabled=use_bf16):
            out = raw_decode(model, target=ctx, horizon=PRED_STEPS,
                             past_future_covariates=cov)
        loss = _pinball(torch, out[:, 0].float().sort(-1).values, fut, fmask, q)
        if not torch.isfinite(loss):
            n_skipped += 1
            opt.zero_grad(set_to_none=True)
            _maybe_report(step); continue
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(train_p, 1.0)
        if not torch.isfinite(gnorm):
            n_skipped += 1
            opt.zero_grad(set_to_none=True)
            _maybe_report(step); continue
        opt.step(); opt.zero_grad(set_to_none=True); n_applied += 1

        if step % a.eval_every == 0:
            v = evaluate()
            mark = ""
            if v < best - 1e-5:
                best, bad, mark = v, 0, "  *best*"
                _save(model, run_dir / "model", revin_was)
            else:
                bad += 1
            log.info(f"  step {step}/{a.steps}  train {float(loss):.4f}  "
                     f"val {v:.4f}{mark}  [{n_applied} steps applied, "
                     f"{n_skipped} skipped]  ({(time.time()-t0)/60:.1f} min)")
            if bad >= a.patience:
                log.info(f"early stop at step {step}; best val {best:.4f}")
                break

    if not (run_dir / "model").exists():          # never improved: save anyway
        _save(model, run_dir / "model", revin_was)
    save_config(cfg, run_dir / "config.resolved.yaml")
    log.write_json("manifest.json", {
        "run_name": a.name, "fine_tune": True, "base_model": a.model_id,
        "mode": a.mode, "unfrozen": a.unfrozen, "steps": a.steps,
        "batch": a.batch, "lr": lr, "trainable_params": int(n_train),
        "best_val_pinball": float(best), "quantiles": qs,
        "n_train_windows": len(train_win), "n_val_windows": len(val_win),
        "config": cfg.to_dict(), "provenance": provenance(),
    })
    log.info(f"saved {run_dir / 'model'}  (best val pinball {best:.4f})")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
