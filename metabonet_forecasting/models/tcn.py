"""Temporal convolutional backbone (Bai, Kolter & Koltun 2018).

Tests: *are multiscale local/medium-range temporal patterns sufficient?*

Two properties matter for a fair comparison and are enforced here rather than
assumed:

* **strictly causal** -- every convolution is left-padded only (``CausalConv1d``),
  so output t depends on inputs <= t;
* **receptive field matched to the context** -- ``receptive_field()`` reports
  1 + 2*(k-1)*(2^n_levels - 1) and ``auto_levels`` picks the depth that covers the
  configured window. A TCN that physically cannot see 24 h would otherwise be
  quietly entered into the long-context study.
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


class TCNBlock(nn.Module):
    """Two dilated causal convolutions with a residual connection."""

    def __init__(self, d_model: int, kernel_size: int, dilation: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(d_model, d_model, kernel_size, dilation)
        self.conv2 = CausalConv1d(d_model, d_model, kernel_size, dilation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(F.gelu(self.conv1(self.norm1(x))))
        h = self.drop(F.gelu(self.conv2(self.norm2(h))))
        return x + h


@register_backbone("tcn")
class TCNBackbone(TemporalBackbone):
    def __init__(self, d_model: int, n_levels: int = 6, kernel_size: int = 3,
                 dropout: float = 0.1, context: Optional[int] = None,
                 auto_levels: bool = True) -> None:
        super().__init__(d_model)
        if auto_levels and context:
            n_levels = self.levels_for_context(context, kernel_size)
        self.kernel_size = kernel_size
        self.n_levels = n_levels
        self.blocks = nn.ModuleList([
            TCNBlock(d_model, kernel_size, dilation=2 ** i, dropout=dropout)
            for i in range(n_levels)
        ])
        self.norm = nn.LayerNorm(d_model)

    @staticmethod
    def levels_for_context(context: int, kernel_size: int) -> int:
        """Smallest depth whose receptive field covers ``context`` steps."""
        n = 1
        while 1 + 2 * (kernel_size - 1) * (2 ** n - 1) < context and n < 16:
            n += 1
        return n

    def receptive_field(self) -> int:
        return 1 + 2 * (self.kernel_size - 1) * (2 ** self.n_levels - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.norm(x)
