"""xLSTM backbone (Beck et al., NeurIPS 2024) — mLSTM blocks, parallel form.

Tests: *does exponential-gated matrix-memory recurrence — the modern LSTM that
matches transformers on language and tops recent zero-shot forecasting models
(TiRex is built from it) — beat Mamba-2's SSD recurrence here?*

This implements the mLSTM in its **parallel (attention-like) form**, which is
exact — the same mathematics as the recurrent form, computed with one masked
matrix product instead of a step loop (paper appendix A.3):

    D̃[i, j] = (F[i] − F[j]) + ĩ[j]   for j ≤ i, else −inf
    where F[t] = Σ_{s≤t} log σ(f̃[s]) is the cumulative log forget gate
    S = (Q Kᵀ / √d) ∘ exp(D̃ − m),  m[i] = max_j D̃[i, j]     (stabiliser)
    H[i] = Σ_j S[i, j] V[j] / max(|Σ_j S[i, j]|, exp(−m[i]))

Strict causality is the ``j ≤ i`` mask, and the causality test covers it like
every other backbone. The block follows the paper's pre-LayerNorm residual
design: up-projection ×2, a short causal convolution feeding Q/K, per-head
GroupNorm on the memory output, a SiLU-gated side path, down-projection.

The gate/score computation runs in fp32 under AMP for the same reason Mamba-2
does: a product of hundreds of forget gates is exactly what fp16 loses.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.registry import register_backbone
from .base import TemporalBackbone
from .encoders import CausalConv1d


class MLSTMLayer(nn.Module):
    """The parallel mLSTM memory: [B, L, d_in] -> [B, L, d_in]."""

    def __init__(self, d_in: int, n_heads: int = 4, d_conv: int = 4) -> None:
        super().__init__()
        if d_in % n_heads:
            raise ValueError(f"d_in {d_in} not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.d_head = d_in // n_heads
        self.conv = CausalConv1d(d_in, d_in, d_conv)
        self.q_proj = nn.Linear(d_in, d_in, bias=False)
        self.k_proj = nn.Linear(d_in, d_in, bias=False)
        self.v_proj = nn.Linear(d_in, d_in, bias=False)
        # Scalar input/forget gate per head per step. Forget bias starts positive
        # so the initial memory horizon is long, per the paper's grid.
        self.gate_proj = nn.Linear(d_in, 2 * n_heads)
        nn.init.zeros_(self.gate_proj.weight)
        with torch.no_grad():
            self.gate_proj.bias.copy_(torch.cat([
                torch.zeros(n_heads),          # input gate ĩ
                torch.full((n_heads,), 3.0),   # forget gate f̃ -> σ ≈ 0.95
            ]))
        self.out_norm = nn.GroupNorm(n_heads, d_in)
        self.out_proj = nn.Linear(d_in, d_in, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, d = x.shape
        h, dh = self.n_heads, self.d_head
        conv_x = F.silu(self.conv(x))
        q = self.q_proj(conv_x).view(B, L, h, dh).transpose(1, 2)   # [B, h, L, dh]
        k = self.k_proj(conv_x).view(B, L, h, dh).transpose(1, 2)
        v = self.v_proj(x).view(B, L, h, dh).transpose(1, 2)
        gates = self.gate_proj(x)                                    # [B, L, 2h]
        i_gate = gates[..., :h].transpose(1, 2)                      # [B, h, L]
        f_gate = gates[..., h:].transpose(1, 2)

        # Exponential gating is numerically alive only in fp32.
        with torch.autocast(device_type=x.device.type, enabled=False):
            q32, k32, v32 = q.float(), k.float(), v.float()
            log_f = F.logsigmoid(f_gate.float())                     # [B, h, L]
            F_cum = torch.cumsum(log_f, dim=-1)
            # D̃[i, j] = F[i] - F[j] + ĩ[j], causal-masked.
            d_mat = F_cum.unsqueeze(-1) - F_cum.unsqueeze(-2) \
                + i_gate.float().unsqueeze(-2)                       # [B, h, L, L]
            causal = torch.ones(L, L, dtype=torch.bool, device=x.device).tril()
            d_mat = d_mat.masked_fill(~causal, float("-inf"))
            m = d_mat.max(dim=-1, keepdim=True).values               # [B, h, L, 1]
            decay = torch.exp(d_mat - m)
            scores = (q32 @ k32.transpose(-1, -2)) / math.sqrt(dh) * decay
            norm = scores.sum(dim=-1, keepdim=True).abs().clamp_min(
                torch.exp(-m))
            out = (scores / norm) @ v32                              # [B, h, L, dh]

        out = out.to(x.dtype).transpose(1, 2).reshape(B, L, d)
        # Headwise norm per time step: GroupNorm over [N, C] sees one position at
        # a time. Feeding it [B, C, L] instead would pool statistics across the
        # whole window and leak the future into the past.
        out = self.out_norm(out.reshape(B * L, d)).reshape(B, L, d)
        return self.out_proj(out)


class XLSTMBlock(nn.Module):
    """Pre-norm residual mLSTM block with the paper's SiLU-gated side path."""

    def __init__(self, d_model: int, n_heads: int = 4, expand: int = 2,
                 d_conv: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        d_inner = expand * d_model
        self.norm = nn.LayerNorm(d_model)
        self.up_proj = nn.Linear(d_model, 2 * d_inner)
        self.mlstm = MLSTMLayer(d_inner, n_heads=n_heads, d_conv=d_conv)
        self.down_proj = nn.Linear(d_inner, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, z = self.up_proj(self.norm(x)).chunk(2, dim=-1)
        h = self.mlstm(u) * F.silu(z)
        return x + self.drop(self.down_proj(h))


@register_backbone("xlstm")
class XLSTMBackbone(TemporalBackbone):
    def __init__(self, d_model: int, n_layers: int = 4, n_heads: int = 4,
                 expand: int = 2, d_conv: int = 4, dropout: float = 0.1,
                 context: Optional[int] = None) -> None:
        super().__init__(d_model)
        self.blocks = nn.ModuleList([
            XLSTMBlock(d_model, n_heads=n_heads, expand=expand, d_conv=d_conv,
                       dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.norm(x)
