"""Forecast heads (plan stage 5): one shared design for every backbone.

The head is shared on purpose. If PatchTST used a flatten-all-patches head and
the TCN used a last-step head, part of any measured difference would be the head.
Here every backbone hands over ``[B, L, d_model]`` and the head pools it the same
way.

``pooling``:
    last       the final step -- the natural "now" representation
    mean       average over the window
    last_mean  both, concatenated (default): the last step carries the current
               state, the mean carries the recent regime
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn


class MultiHorizonHead(nn.Module):
    """[B, L, d] -> [B, H] point forecasts, optionally [B, H, Q] quantiles."""

    def __init__(self, d_model: int, n_horizons: int, hidden: Optional[int] = None,
                 pooling: str = "last_mean", dropout: float = 0.0,
                 quantiles: Optional[Sequence[float]] = None) -> None:
        super().__init__()
        if pooling not in ("last", "mean", "last_mean"):
            raise ValueError(f"unknown pooling {pooling!r}")
        self.pooling = pooling
        self.n_horizons = n_horizons
        self.quantiles = list(quantiles) if quantiles else []
        in_dim = d_model * (2 if pooling == "last_mean" else 1)
        hidden = hidden or max(d_model, 64)
        n_out = n_horizons * (1 + len(self.quantiles))
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_out),
        )
        # Start from "no change": with delta targets, a zero output is the
        # persistence forecast, which is a sane and honest initial state.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def pool(self, seq: torch.Tensor) -> torch.Tensor:
        if self.pooling == "last":
            return seq[:, -1]
        if self.pooling == "mean":
            return seq.mean(dim=1)
        return torch.cat([seq[:, -1], seq.mean(dim=1)], dim=-1)

    def forward(self, seq: torch.Tensor):
        out = self.net(self.pool(seq))
        point = out[:, :self.n_horizons]
        if not self.quantiles:
            return point, None
        q = out[:, self.n_horizons:].reshape(-1, self.n_horizons, len(self.quantiles))
        # Monotone quantiles by construction: cumulative softplus offsets around
        # the point forecast, so the 90th can never fall below the 10th.
        offsets = torch.cumsum(torch.nn.functional.softplus(q), dim=-1)
        centre = offsets[..., len(self.quantiles) // 2: len(self.quantiles) // 2 + 1]
        return point, point.unsqueeze(-1) + (offsets - centre)
