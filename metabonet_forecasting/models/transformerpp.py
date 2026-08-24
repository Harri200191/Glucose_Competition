"""Modern decoder transformer backbone (LLaMA-style: RoPE + RMSNorm + SwiGLU).

Tests: *does the current-generation attention stack — the one every 2024-25
frontier sequence model actually uses — beat both the dated PatchTST design and
the recurrent state-space models on this task?*

The transformers already in this study are period pieces: PatchTST is a 2022
encoder with absolute learned positions and post-hoc patching, TFT is 2019. The
stack here is what replaced them everywhere else:

* **rotary position embeddings** — relative positions enter through the Q·K
  product itself, so the model is translation-equivariant over the window
  instead of memorising absolute slot indices;
* **RMSNorm, pre-norm, no biases** — the stability recipe of every modern LM;
* **SwiGLU FFN** — the gated FFN that consistently beats GELU MLPs at equal
  parameter count;
* **strict causal masking** via ``scaled_dot_product_attention(is_causal=True)``,
  which also picks the fastest available attention kernel for the hardware.

Per-step attention over the raw 5-minute lattice (no patching): at context 288
the quadratic cost is trivial, and the shared head reads one state per step
like every other backbone.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.registry import register_backbone
from .base import TemporalBackbone


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


def rope_cache(seq_len: int, d_head: int, device, dtype,
               base: float = 10000.0):
    """cos/sin tables [L, d_head//2]."""
    inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2, device=device,
                                            dtype=torch.float32) / d_head))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """x: [B, h, L, d_head] with d_head even."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        if self.d_head % 2:
            raise ValueError("head dimension must be even for RoPE")
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        cos, sin = rope_cache(L, self.d_head, x.device, x.dtype)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0)
        out = out.transpose(1, 2).reshape(B, L, d)
        return self.proj(out)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, ffn_ratio: float, dropout: float) -> None:
        super().__init__()
        hidden = int(ffn_ratio * d_model)
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.drop(F.silu(self.gate(x)) * self.up(x)))


class TransformerPPBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_ratio: float,
                 dropout: float) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, ffn_ratio, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


@register_backbone("transformerpp")
class TransformerPPBackbone(TemporalBackbone):
    def __init__(self, d_model: int, n_layers: int = 6, n_heads: int = 8,
                 ffn_ratio: float = 2.67, dropout: float = 0.1,
                 context: Optional[int] = None) -> None:
        super().__init__(d_model)
        self.blocks = nn.ModuleList([
            TransformerPPBlock(d_model, n_heads, ffn_ratio, dropout)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.norm(x)
