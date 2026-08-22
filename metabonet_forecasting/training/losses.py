"""Losses.

All of them are **masked**: the four horizons have slightly different sample
counts (a window can have a real reading at +30 min and none at +120), so a loss
that ignored the mask would train on fabricated zeros.

Losses operate in model space (standardised delta or absolute), which puts the
horizons on a comparable scale -- the standard deviation of the 120-minute change
is roughly three times that of the 30-minute change, so an unscaled multi-horizon
loss would silently become a 120-minute loss.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum().clamp(min=1.0)
    return (x * mask).sum() / denom


class MaskedHuber(nn.Module):
    """Huber on standardised targets. Robust to the tail of sensor artefacts."""

    def __init__(self, delta: float = 1.0,
                 horizon_weights: Optional[Sequence[float]] = None) -> None:
        super().__init__()
        self.delta = delta
        self.register_buffer(
            "w", torch.tensor(list(horizon_weights), dtype=torch.float32)
            if horizon_weights else torch.ones(1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        loss = F.huber_loss(pred, target, reduction="none", delta=self.delta)
        w = self.w.to(pred.device)
        if w.numel() == pred.shape[1]:
            loss = loss * w.reshape(1, -1)
        return masked_mean(loss, mask)


class MaskedMSE(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        return masked_mean((pred - target) ** 2, mask)


class MaskedMAE(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        return masked_mean((pred - target).abs(), mask)


class QuantileLoss(nn.Module):
    """Pinball loss for the optional probabilistic head (plan stage 5, later phase)."""

    def __init__(self, quantiles: Sequence[float]) -> None:
        super().__init__()
        self.register_buffer("q", torch.tensor(list(quantiles), dtype=torch.float32))

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        err = target.unsqueeze(-1) - pred
        q = self.q.to(pred.device).reshape(1, 1, -1)
        loss = torch.maximum(q * err, (q - 1.0) * err).mean(dim=-1)
        return masked_mean(loss, mask)


class HybridLoss(nn.Module):
    """Huber on the point forecast + pinball on the quantiles.

    Receives the quantile tensor ``[B, H, Q]``. The head constructs quantiles as
    monotone offsets around the point forecast with the centre offset zero, so
    the middle column *is* the point forecast — Huber on it trains the point
    exactly as the plain loss would, while the pinball term shapes the rest of
    the distribution. ``alpha`` weights the pinball term.
    """

    def __init__(self, quantiles: Sequence[float], delta: float = 1.0,
                 alpha: float = 1.0) -> None:
        super().__init__()
        self.point = MaskedHuber(delta)
        self.pinball = QuantileLoss(quantiles)
        self.mid = len(list(quantiles)) // 2
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        return self.point(pred[..., self.mid], target, mask) \
            + self.alpha * self.pinball(pred, target, mask)


def build_loss(cfg: Dict) -> nn.Module:
    name = str(cfg.get("name", "huber")).lower()
    if name == "huber":
        return MaskedHuber(float(cfg.get("delta", 1.0)), cfg.get("horizon_weights"))
    if name == "mse":
        return MaskedMSE()
    if name == "mae":
        return MaskedMAE()
    if name == "quantile":
        return QuantileLoss(cfg.get("quantiles", [0.1, 0.5, 0.9]))
    if name == "hybrid":
        return HybridLoss(cfg.get("quantiles", [0.1, 0.5, 0.9]),
                          float(cfg.get("delta", 1.0)),
                          float(cfg.get("alpha", 1.0)))
    raise ValueError(f"unknown loss {name!r}")
