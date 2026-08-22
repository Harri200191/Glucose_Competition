"""MambaFormer: Mamba-2 recurrence interleaved with modern attention.

Tests: *does the 2025 hybrid recipe — state-space layers for cheap long-range
integration, periodic attention layers for content-based retrieval — beat the
pure Mamba-2 that currently tops this benchmark?*

The pattern follows Jamba/Samba: mostly Mamba-2 layers (the incumbent best
backbone here), with an attention layer inserted every ``attn_every`` layers.
The state-space layers carry the slow physiology (insulin decay, circadian
drift) in their state; the attention layers let the anchor step look back at
*specific* events — the last meal spike, the last correction bolus — which a
fixed-size recurrent state can only summarise.

Both halves reuse tested code: ``Mamba2Layer`` from ``mamba2.py`` (the chunked
SSD scan, verified against an explicit recurrence) and the RoPE causal
attention block from ``transformerpp.py``. Strictly causal by composition of
causal parts, and covered by the same perturbation test as every backbone.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..common.registry import register_backbone
from .base import TemporalBackbone
from .mamba2 import Mamba2Layer
from .transformerpp import TransformerPPBlock


@register_backbone("mambaformer")
class MambaFormerBackbone(TemporalBackbone):
    def __init__(self, d_model: int, n_layers: int = 6, attn_every: int = 3,
                 n_heads: int = 8, ffn_ratio: float = 2.67,
                 d_state: int = 64, d_conv: int = 4, expand: int = 2,
                 headdim: int = 32, chunk_size: int = 24,
                 dropout: float = 0.1, context: Optional[int] = None) -> None:
        super().__init__(d_model)
        mamba_kwargs = dict(d_state=d_state, d_conv=d_conv, expand=expand,
                            headdim=headdim, chunk_size=chunk_size)
        layers = []
        for i in range(n_layers):
            # Layers 1-indexed: with attn_every=3, layers 3 and 6 are attention.
            if attn_every and (i + 1) % attn_every == 0:
                layers.append(TransformerPPBlock(d_model, n_heads, ffn_ratio,
                                                 dropout))
            else:
                layers.append(Mamba2Layer(d_model, dropout=dropout,
                                          **mamba_kwargs))
        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
