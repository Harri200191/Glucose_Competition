"""Recurrent backbone -- the "simple LSTM" reference of plan stage B.

Not a contender for the headline table: it is the control that says how much of
any backbone's advantage is just "a sequence model was used at all".
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..common.registry import register_backbone
from .base import TemporalBackbone


@register_backbone("lstm")
class LSTMBackbone(TemporalBackbone):
    def __init__(self, d_model: int, n_layers: int = 2, dropout: float = 0.1,
                 context: Optional[int] = None) -> None:
        super().__init__(d_model)
        self.rnn = nn.LSTM(d_model, d_model, num_layers=n_layers, batch_first=True,
                           dropout=dropout if n_layers > 1 else 0.0)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)          # unidirectional: causal by construction
        return self.norm(out)


@register_backbone("gru")
class GRUBackbone(TemporalBackbone):
    def __init__(self, d_model: int, n_layers: int = 2, dropout: float = 0.1,
                 context: Optional[int] = None) -> None:
        super().__init__(d_model)
        self.rnn = nn.GRU(d_model, d_model, num_layers=n_layers, batch_first=True,
                          dropout=dropout if n_layers > 1 else 0.0)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.norm(out)
