"""ModernTCN backbone (Luo & Wang, ICLR 2024), causal adaptation.

Tests: *do large-kernel depthwise convolutions — the design that put convnets
back on top of the long-horizon forecasting benchmarks — beat the dilated TCN
and Mamba-2 here?*

The original ModernTCN interleaves a depthwise large-kernel convolution with two
pointwise ConvFFNs (one mixing channels-per-variable, one mixing variables).
This pipeline hands the backbone one fused latent sequence rather than separate
variables, so the variable-mixing FFN collapses into the ordinary pointwise FFN
— the structure that remains is exactly ConvNeXt-for-time-series: depthwise
large kernel, then an inverted-bottleneck MLP, both residual.

Two deviations from the paper, both deliberate:

* **strictly causal** — the paper pads symmetrically; here every depthwise
  convolution is a ``CausalConv1d`` (left padding only) so output *t* depends on
  inputs ``<= t``, matching every other backbone in the study and the causality
  tests.
* **no patch-embedding downsampling** — the paper stem halves resolution; at 288
  steps the sequence is short enough to keep full resolution, and the shared
  head expects one output per input step.

Depth is auto-sized so the receptive field covers the context, mirroring the
TCN's ``auto_levels``.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from ..common.registry import register_backbone
from .base import TemporalBackbone
from .encoders import CausalConv1d


class ModernTCNBlock(nn.Module):
    """Depthwise large-kernel causal conv + inverted-bottleneck FFN, residual."""

    def __init__(self, d_model: int, kernel_size: int, ffn_ratio: int = 2,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.dwconv = CausalConv1d(d_model, d_model, kernel_size, groups=d_model)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = ffn_ratio * d_model
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.dwconv(self.norm1(x)))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


@register_backbone("moderntcn")
class ModernTCNBackbone(TemporalBackbone):
    def __init__(self, d_model: int, kernel_size: int = 51, n_blocks: int = 6,
                 ffn_ratio: int = 2, dropout: float = 0.1,
                 context: Optional[int] = None, auto_blocks: bool = True) -> None:
        super().__init__(d_model)
        if auto_blocks and context:
            n_blocks = self.blocks_for_context(context, kernel_size)
        self.kernel_size = kernel_size
        self.n_blocks = n_blocks
        self.blocks = nn.ModuleList([
            ModernTCNBlock(d_model, kernel_size, ffn_ratio, dropout)
            for _ in range(n_blocks)
        ])
        self.norm = nn.LayerNorm(d_model)

    @staticmethod
    def blocks_for_context(context: int, kernel_size: int) -> int:
        """Smallest depth whose receptive field covers ``context`` steps."""
        return max(1, math.ceil((context - 1) / (kernel_size - 1)))

    def receptive_field(self) -> int:
        return 1 + (self.kernel_size - 1) * self.n_blocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.norm(x)
