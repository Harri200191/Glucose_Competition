"""Self-supervised pretraining for the GlucoFM encoder, on OUR corpus.

There is no public GlucoFM checkpoint. Without one, ``model/glucofm`` is a
dual-stream architecture trained end-to-end with the forecaster -- a fair test of
the inductive bias, but not a foundation model. This script supplies the missing
half: it runs the paper's two objectives over our unlabeled CGM and writes an
encoder checkpoint that ``model.encoder.fm.weights`` can load and freeze.

    python -m metabonet_forecasting.fm.glucofm_pretrain \
        --set data.store=artifacts/store/main \
        --set data.internal_holdout.enabled=true \
        --set data.internal_holdout.exclude_whole_source=true \
        --out artifacts/fm_weights/glucofm_ssl.pt --epochs 3

WHY THIS IS WORTH DOING. GlucoFM was pretrained on 109,066 hours from 477
participant/session records, mostly non-diabetic. Our training split is orders of
magnitude larger and is the actual deployment distribution. Pretraining in-domain
is the strictly better version of the same recipe.

HOLDOUT SAFETY. The loader comes from ``build_data(cfg)`` with the same config
the forecaster uses, so the train split's exclusions apply here unchanged. Run
this with the same ``data.internal_holdout.*`` settings as the runs that will
consume it: SSL pretraining is training. A representation fitted on holdout
patients leaks them into every downstream model that loads it, and no later
integrity check would catch it, because by then the weights are just a file.
The header of the checkpoint records the split sizes so the provenance survives.

THE TWO OBJECTIVES (paper section 3, lambda_MCR = lambda_TD = 1.0)

    MCR  mask 50-60% of the 24 patches, replace them with a learned token, and
         predict the EMA target encoder's latents at those positions. SmoothL1,
         weighted by each patch's observed fraction so an all-missing patch --
         which the target encoder cannot describe either -- contributes nothing.
    TD   from the unmasked context, predict the next patch's state and event
         representations. This is the part that makes the encoder model
         dynamics rather than just fill in gaps.

Predicting *latents* rather than raw glucose is the whole point: reconstructing
values rewards memorising sensor noise, and the JEPA family's result is that the
latent objective is what produces representations that survive a cohort change.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.config import config_from_args
from ..data.dataset import build_data
from ..models.fm_encoders import mask_aware_instance_norm
from ..models.glucofm_net import GlucoFMNet, GlucoFMPredictor, patchify


def _mask_patches(B: int, N: int, lo: float, hi: float, device) -> torch.Tensor:
    """[B, N] with 1 = hidden; each row hides the same count, drawn in [lo, hi]."""
    k = max(1, min(N - 1, int(round(N * float(torch.empty(1).uniform_(lo, hi))))))
    order = torch.rand(B, N, device=device).argsort(dim=1)
    m = torch.zeros(B, N, device=device)
    return m.scatter_(1, order[:, :k], 1.0)


def _cgm(batch, device):
    x = batch["x_cgm"].to(device, non_blocking=True)
    g = x[..., 0]
    m = batch.get("m_cgm")
    obs = (m[..., 0].to(device) > 0.5).float() if m is not None else \
          ((x[..., 1] > 0.5).float() if x.shape[-1] > 1 else torch.ones_like(g))
    return mask_aware_instance_norm(g, obs) * obs, obs


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="base")
    ap.add_argument("--set", dest="sets", action="append", default=[])
    ap.add_argument("--out", required=True, help="checkpoint path to write")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=0, help="0 = a full epoch")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--mask-lo", type=float, default=0.5)
    ap.add_argument("--mask-hi", type=float, default=0.6)
    ap.add_argument("--ema", type=float, default=0.996,
                    help="target-encoder momentum at step 0; annealed to 1.0")
    ap.add_argument("--d-fm", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--ffn", type=int, default=256)
    ap.add_argument("--patch", type=int, default=12)
    ap.add_argument("--sigma-min", type=float, default=2.0)
    ap.add_argument("--sigma-max", type=float, default=12.0)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)

    cfg = config_from_args(a.config, a.sets)
    bundle = build_data(cfg)
    counts = {k: len(v) for k, v in bundle.datasets.items()}
    print(f"[glucofm-ssl] windows: {json.dumps(counts)}")
    print(f"[glucofm-ssl] pretraining on the TRAIN split only -- the same split, "
          f"with the same exclusions, that the forecaster uses.")
    loader = bundle.loader("train", cfg, shuffle=True)

    dev = torch.device(a.device)
    net = GlucoFMNet(d_fm=a.d_fm, n_heads=a.n_heads, n_layers=a.n_layers, ffn=a.ffn,
                     patch=a.patch, sigma_min=a.sigma_min, sigma_max=a.sigma_max).to(dev)
    target = copy.deepcopy(net).to(dev).eval()
    for p in target.parameters():
        p.requires_grad_(False)
    predictor = GlucoFMPredictor(d_fm=a.d_fm, n_heads=a.n_heads, ffn=a.ffn).to(dev)

    params = list(net.parameters()) + list(predictor.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.weight_decay)
    total = (a.max_steps or len(loader)) * a.epochs
    print(f"[glucofm-ssl] encoder {sum(p.numel() for p in net.parameters()):,} params, "
          f"{total:,} planned steps on {dev}")

    step, t0 = 0, time.time()
    run = {"mcr": 0.0, "td": 0.0, "n": 0}
    for epoch in range(a.epochs):
        for batch in loader:
            if a.max_steps and step >= a.max_steps * (epoch + 1):
                break
            g, obs = _cgm(batch, dev)
            B, T = g.shape
            if T % a.patch:                      # left-trim to a whole number of patches
                g, obs = g[:, T % a.patch:], obs[:, T % a.patch:]
            N = g.shape[1] // a.patch

            for gp in opt.param_groups:          # linear warmup, cosine decay
                gp["lr"] = a.lr * (min(1.0, (step + 1) / max(1, a.warmup))
                                   * (0.5 * (1 + math.cos(math.pi * step / max(1, total)))))

            with torch.no_grad():
                z_tgt = target(g, obs)                        # [B, N, d]
            w = patchify(obs, a.patch).mean(dim=-1)           # observed fraction per patch

            # ---- MCR: masked latent prediction ----------------------------
            pm = _mask_patches(B, N, a.mask_lo, a.mask_hi, dev)
            p_mcr = predictor(net(g, obs, None, pm))
            wm = (w * pm).unsqueeze(-1)
            l_mcr = (F.smooth_l1_loss(p_mcr, z_tgt, reduction="none") * wm).sum() \
                    / wm.sum().clamp_min(1e-6) / z_tgt.shape[-1]

            # ---- TD: next-patch state and event dynamics -------------------
            h = predictor(net(g, obs))
            wt = (w[:, 1:] * w[:, :-1]).unsqueeze(-1)
            l_td = 0.0
            for read in ("to_state", "to_event"):
                pred = getattr(net, read)(h[:, :-1])
                tgt = getattr(target, read)(z_tgt[:, 1:]).detach()
                l_td = l_td + (F.smooth_l1_loss(pred, tgt, reduction="none") * wt).sum() \
                       / wt.sum().clamp_min(1e-6) / tgt.shape[-1]
            l_td = l_td / 2.0

            loss = l_mcr + l_td
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            # EMA target, momentum annealed toward 1.0 (BYOL/I-JEPA schedule)
            with torch.no_grad():
                mom = 1.0 - (1.0 - a.ema) * (math.cos(math.pi * step / max(1, total)) + 1) / 2
                for pt, ps in zip(target.parameters(), net.parameters()):
                    pt.mul_(mom).add_(ps.detach(), alpha=1.0 - mom)
                for bt, bs in zip(target.buffers(), net.buffers()):
                    bt.copy_(bs)

            run["mcr"] += float(l_mcr); run["td"] += float(l_td); run["n"] += 1
            step += 1
            if step % a.log_every == 0:
                n = max(1, run["n"])
                print(f"[glucofm-ssl] step {step:>7,}/{total:,}  "
                      f"MCR {run['mcr']/n:.4f}  TD {run['td']/n:.4f}  "
                      f"sigma {float(net.filter.sigma()):.2f} steps  "
                      f"lr {opt.param_groups[0]['lr']:.2e}  "
                      f"{(time.time()-t0)/step:.3f}s/step", flush=True)
                run = {"mcr": 0.0, "td": 0.0, "n": 0}

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"encoder": net.state_dict(),
                "meta": {"steps": step, "epochs": a.epochs,
                         "arch": {"d_fm": a.d_fm, "n_heads": a.n_heads,
                                  "n_layers": a.n_layers, "ffn": a.ffn,
                                  "patch": a.patch, "sigma_min": a.sigma_min,
                                  "sigma_max": a.sigma_max},
                         "sigma_learned": float(net.filter.sigma()),
                         "window_counts": counts,
                         "config": cfg.to_dict()}}, out)
    print(f"[glucofm-ssl] wrote {out} after {step:,} steps "
          f"({(time.time()-t0)/60:.1f} min); learned sigma = "
          f"{float(net.filter.sigma()):.2f} steps ({float(net.filter.sigma())*5:.0f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
