"""GlucoFM (Google Research / UNSW, arXiv:2605.30865) -- architecture, built here.

THERE ARE NO PUBLIC WEIGHTS. The paper says "we will release the code and
reproducibility scripts"; as of this writing nothing is on GitHub or HuggingFace
and the blog post links no checkpoint. So unlike CGM-JEPA, which we load, this
module is the *architecture* re-built from the paper's description. Anything
that calls it must say "GlucoFM-arch", never "GlucoFM", unless it has first
loaded weights produced by ``fm/glucofm_pretrain.py`` on our own corpus.

That is not a consolation prize. GlucoFM's claim is an inductive bias -- split
the trace into a slow physiological state and fast events, keep the missingness
mask -- and an SSL recipe. Both are reproducible without their checkpoint, and
our unlabeled CGM corpus is ~100x the 109,066 hours they pretrained on.

The paper's specification, as implemented below:

    grid          24 h at 5 min  -> L = 288, observation mask M in {0,1}^L
    normalise     mask-aware, over observed entries only
    state         X_state = Filter(X, M), causal Gaussian, LEARNABLE bandwidth
                  sigma in [2.0, 12.0] grid steps
    event         X_event = (X - X_state) * M, plus rate-of-change statistics
    encoder       3-layer transformer, d 128, 4 heads, FFN 256 (mlp_ratio 2)
    predictor     1-layer transformer (pretraining only)
    objectives    MCR (mask 50-60%, SmoothL1 on latents) + TD (next-patch
                  state/event), lambda_MCR = lambda_TD = 1.0, EMA target encoder

Two deliberate deviations, both recorded because they change what the numbers mean:

* the causal Gaussian filter is normalised by the mask-weighted tap sum, so a
  gap does not drag the trend toward zero. The paper says "Filter(X, M)" without
  giving the form; this is the only reading under which M does anything.
* TD is read out with two linear heads on the fused representation rather than
  running two separate towers. The paper describes per-stream transitions; one
  shared trunk with per-stream read-outs is the cheap equivalent and keeps the
  encoder identical between pretraining and fine-tuning.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cgm_jepa_net import Block


class CausalGaussianFilter(nn.Module):
    """Mask-aware causal Gaussian smoother with a learnable bandwidth.

        state[t] = sum_k w_k m[t-k] x[t-k] / sum_k w_k m[t-k],
        w_k = exp(-k^2 / 2 sigma^2),  k = 0 .. taps-1,  sigma in [lo, hi]

    Left padding only, so ``state[t]`` depends on inputs at or before ``t``.
    ``sigma`` is a single scalar squashed through a sigmoid, which keeps it in
    the paper's [2.0, 12.0] range for any value the optimiser reaches.
    """

    def __init__(self, sigma_min: float = 2.0, sigma_max: float = 12.0,
                 taps: Optional[int] = None) -> None:
        super().__init__()
        self.sigma_min, self.sigma_max = float(sigma_min), float(sigma_max)
        # 3 sigma of the widest kernel; beyond that the taps are numerically zero.
        self.taps = int(taps or math.ceil(3.0 * self.sigma_max) + 1)
        # sigmoid(0) = 0.5 -> starts at the midpoint of the range (7.0 steps = 35 min)
        self.raw_sigma = nn.Parameter(torch.zeros(()))
        self.register_buffer("_k", torch.arange(self.taps).float())

    def sigma(self) -> torch.Tensor:
        return self.sigma_min + (self.sigma_max - self.sigma_min) * torch.sigmoid(self.raw_sigma)

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """x, m: [B, T] -> [B, T]."""
        w = torch.exp(-(self._k ** 2) / (2.0 * self.sigma() ** 2))
        # conv1d correlates, so flipping puts tap 0 (the current step) at the right edge
        k = w.flip(0).view(1, 1, -1).to(x.dtype)
        xm = (x * m).unsqueeze(1)
        mm = m.unsqueeze(1)
        pad = (self.taps - 1, 0)
        num = F.conv1d(F.pad(xm, pad), k).squeeze(1)
        den = F.conv1d(F.pad(mm, pad), k).squeeze(1)
        return num / den.clamp_min(1e-4)


def patchify(x: torch.Tensor, patch: int) -> torch.Tensor:
    """[B, T] -> [B, T // patch, patch]. T must divide exactly."""
    B, T = x.shape
    return x.view(B, T // patch, patch)


class GlucoFMNet(nn.Module):
    """Dual-stream GlucoFM encoder: [B, T] glucose + mask -> [B, N, d_fm].

    ``forward`` optionally takes ``patch_mask`` ([B, N] with 1 = hide) so the
    same module serves masked-latent pretraining and downstream use without a
    second code path.
    """

    def __init__(self, d_fm: int = 128, n_heads: int = 4, n_layers: int = 3,
                 ffn: int = 256, patch: int = 12, dropout: float = 0.0,
                 sigma_min: float = 2.0, sigma_max: float = 12.0,
                 taps: Optional[int] = None, time_inp_dim: int = 5,
                 max_patches: int = 512) -> None:
        super().__init__()
        self.d_fm, self.patch = d_fm, patch
        self.time_inp_dim = time_inp_dim
        self.filter = CausalGaussianFilter(sigma_min, sigma_max, taps)

        self.state_embed = nn.Linear(patch, d_fm)
        self.event_embed = nn.Linear(2 * patch, d_fm)   # event values + rate of change
        self.miss_embed = nn.Linear(patch, d_fm)
        self.time_embed = nn.Linear(time_inp_dim, d_fm)
        self.pos = nn.Parameter(torch.zeros(1, max_patches, d_fm))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_fm))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.in_norm = nn.LayerNorm(d_fm)

        mlp_ratio = float(ffn) / float(d_fm)
        self.blocks = nn.ModuleList([
            Block(d_fm, n_heads, mlp_ratio=mlp_ratio, qkv_bias=True,
                  drop=dropout, attn_drop=dropout)
            for _ in range(n_layers)])
        self.out_norm = nn.LayerNorm(d_fm)
        # per-stream read-outs, used by the TD objective during pretraining
        self.to_state = nn.Linear(d_fm, d_fm)
        self.to_event = nn.Linear(d_fm, d_fm)

    # ------------------------------------------------------------ streams
    def streams(self, g: torch.Tensor, obs: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Slow state, fast event, and rate of change -- all [B, T]."""
        state = self.filter(g, obs)
        event = (g - state) * obs
        roc = torch.zeros_like(g)
        both = obs[:, 1:] * obs[:, :-1]
        roc[:, 1:] = (g[:, 1:] - g[:, :-1]) * both
        return state, event, roc

    def forward(self, g: torch.Tensor, obs: torch.Tensor,
                x_mark: Optional[torch.Tensor] = None,
                patch_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """g, obs: [B, T]; x_mark: [B, N, patch, time_inp_dim]; -> [B, N, d_fm]."""
        state, event, roc = self.streams(g, obs)
        p = self.patch
        z = (self.state_embed(patchify(state, p))
             + self.event_embed(torch.cat([patchify(event, p), patchify(roc, p)], dim=-1))
             + self.miss_embed(patchify(obs, p)))
        if x_mark is not None:
            B, N, P, D = x_mark.shape
            z = z + self.time_embed(x_mark.reshape(-1, D)).view(B, N, P, -1).mean(dim=2)
        z = self.in_norm(z)
        if patch_mask is not None:
            z = torch.where(patch_mask.unsqueeze(-1).bool(),
                            self.mask_token.to(z.dtype).expand_as(z), z)
        z = z + self.pos[:, :z.size(1)]
        for blk in self.blocks:
            z = blk(z)
        return self.out_norm(z)


class GlucoFMPredictor(nn.Module):
    """The 1-layer transformer used only while pretraining (MCR and TD)."""

    def __init__(self, d_fm: int = 128, n_heads: int = 4, ffn: int = 256,
                 dropout: float = 0.0, max_patches: int = 512) -> None:
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, max_patches, d_fm))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.block = Block(d_fm, n_heads, mlp_ratio=float(ffn) / float(d_fm),
                           qkv_bias=True, drop=dropout, attn_drop=dropout)
        self.norm = nn.LayerNorm(d_fm)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.norm(self.block(z + self.pos[:, :z.size(1)]))
