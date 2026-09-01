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


def _zone_surrogate(torch, pred, target, mask, halfwidth, tau):
    """Smooth stand-in for "is this prediction inside zone A?".

    Zone A is |log(pred/true)| <= ~0.2027 -- a hard 0/1 indicator with no
    gradient anywhere. The standard remedy is to replace the indicator with a
    sigmoid (arXiv 2305.02024), giving soft_hit = sigmoid((h - |resid|) / tau),
    and to minimise 1 - soft_hit.

    ``halfwidth`` is in MODEL space, so the log-space window has to be divided
    by the target standardiser's scale; the Trainer does that and passes it in.
    Only meaningful under a log target mode, where the model-space residual is
    proportional to log(pred/true).

    Blended with the ordinary loss, never used alone: on its own the surrogate
    is flat wherever a prediction sits far outside the band, so it supplies no
    gradient in exactly the cases that are worst.
    """
    resid = (pred - target).abs()
    soft_hit = torch.sigmoid((halfwidth - resid) / max(tau, 1e-6))
    m = mask.float()
    return ((1.0 - soft_hit) * m).sum() / m.sum().clamp_min(1.0)


def _move_weights(target: torch.Tensor, strength: float) -> Optional[torch.Tensor]:
    """Upweight anchors where glucose actually moves a long way.

    Zone-A failures concentrate in the fast-changing anchors: measured on the
    holdout at h=120, the flat third of anchors score 78.2% while the
    fastest-falling tenth score 65.2%. An unweighted loss spends the same
    capacity on a flat night-time stretch as on a post-meal rise, so the cases
    that decide the score are the ones it invests least in.

    ``target`` is already in model space (a delta or a log-ratio), so its
    magnitude *is* how far the glucose travels. The weight is renormalised to
    mean 1 so the loss keeps its scale and the learning rate still means what it
    meant before. strength=0 disables it exactly.
    """
    if strength <= 0:
        return None
    w = 1.0 + strength * target.detach().abs()
    return w / w.mean().clamp_min(1e-6)


class MaskedHuber(nn.Module):
    """Huber on standardised targets. Robust to the tail of sensor artefacts."""

    def __init__(self, delta: float = 1.0,
                 horizon_weights: Optional[Sequence[float]] = None,
                 move_weight: float = 0.0) -> None:
        super().__init__()
        self.delta = delta
        self.move_weight = float(move_weight)
        self.register_buffer(
            "w", torch.tensor(list(horizon_weights), dtype=torch.float32)
            if horizon_weights else torch.ones(1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        loss = F.huber_loss(pred, target, reduction="none", delta=self.delta)
        w = self.w.to(pred.device)
        if w.numel() == pred.shape[1]:
            loss = loss * w.reshape(1, -1)
        mw = _move_weights(target, self.move_weight)
        if mw is not None:
            loss = loss * mw
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

    def __init__(self, quantiles: Sequence[float],
                 horizon_weights: Optional[Sequence[float]] = None,
                 move_weight: float = 0.0) -> None:
        super().__init__()
        self.move_weight = float(move_weight)
        self.register_buffer("q", torch.tensor(list(quantiles), dtype=torch.float32))
        self.register_buffer(
            "w", torch.tensor(list(horizon_weights), dtype=torch.float32)
            if horizon_weights else torch.ones(1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        err = target.unsqueeze(-1) - pred
        q = self.q.to(pred.device).reshape(1, 1, -1)
        loss = torch.maximum(q * err, (q - 1.0) * err).mean(dim=-1)
        w = self.w.to(pred.device)
        if w.numel() == loss.shape[1]:
            loss = loss * w.reshape(1, -1)
        mw = _move_weights(target, self.move_weight)
        if mw is not None:
            loss = loss * mw
        return masked_mean(loss, mask)


class ZoneBlend(nn.Module):
    """base loss + zone_weight * (1 - soft zone-A hit rate)."""

    def __init__(self, base: nn.Module, halfwidth, tau: float,
                 weight: float, mid) -> None:
        super().__init__()
        self.base, self.tau, self.weight, self.mid = base, tau, weight, mid
        self.register_buffer("hw", torch.as_tensor(halfwidth, dtype=torch.float32))

    def forward(self, pred, target, mask):
        point = pred if self.mid is None else pred[..., self.mid]
        hw = self.hw.to(point.device)
        if hw.numel() == point.shape[1]:
            hw = hw.reshape(1, -1)
        return self.base(pred, target, mask) + self.weight * _zone_surrogate(
            torch, point, target, mask, hw, self.tau)


class HybridLoss(nn.Module):
    """Huber on the point forecast + pinball on the quantiles.

    Receives the quantile tensor ``[B, H, Q]``. The head constructs quantiles as
    monotone offsets around the point forecast with the centre offset zero, so
    the middle column *is* the point forecast — Huber on it trains the point
    exactly as the plain loss would, while the pinball term shapes the rest of
    the distribution. ``alpha`` weights the pinball term.
    """

    def __init__(self, quantiles: Sequence[float], delta: float = 1.0,
                 alpha: float = 1.0,
                 horizon_weights: Optional[Sequence[float]] = None,
                 move_weight: float = 0.0) -> None:
        super().__init__()
        self.point = MaskedHuber(delta, horizon_weights, move_weight)
        self.pinball = QuantileLoss(quantiles, horizon_weights, move_weight)
        self.mid = len(list(quantiles)) // 2
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        return self.point(pred[..., self.mid], target, mask) \
            + self.alpha * self.pinball(pred, target, mask)


def build_loss(cfg: Dict) -> nn.Module:
    """Build the loss named in ``training.loss``.

    ``horizon_weights`` re-weights the per-horizon terms, and is the honest way
    to spend capacity where the score is. The scorer pools all four horizons, but
    30 min already sits near its ceiling while 120 min has several times the
    headroom, so an unweighted loss under-invests in the horizon that decides the
    result. Pass four numbers in the order of ``data.horizons_min``; they are
    applied before the mask, so a horizon with no target still contributes
    nothing. Now honoured by every loss, not only ``huber``.
    """
    name = str(cfg.get("name", "huber")).lower()
    hw = cfg.get("horizon_weights") or None
    if hw is not None:
        hw = [float(x) for x in hw]
    mw = float(cfg.get("move_weight", 0.0) or 0.0)
    if name == "huber":
        return MaskedHuber(float(cfg.get("delta", 1.0)), hw, mw)
    if name == "mse":
        return MaskedMSE()
    if name == "mae":
        return MaskedMAE()
    if name == "quantile":
        return QuantileLoss(cfg.get("quantiles", [0.1, 0.5, 0.9]), hw, mw)
    if name == "hybrid":
        return HybridLoss(cfg.get("quantiles", [0.1, 0.5, 0.9]),
                          float(cfg.get("delta", 1.0)),
                          float(cfg.get("alpha", 1.0)), hw, mw)
    raise ValueError(f"unknown loss {name!r}")


def build_loss_with_zone(cfg: Dict) -> nn.Module:
    """``build_loss`` plus the optional zone-A surrogate term."""
    base = build_loss(cfg)
    w = float(cfg.get("zone_weight", 0.0) or 0.0)
    if w <= 0:
        return base
    hw = cfg.get("zone_halfwidth")
    if hw is None:
        raise ValueError(
            "training.loss.zone_weight is set but zone_halfwidth is missing. It "
            "is the zone-A half-width in MODEL space, which depends on the "
            "target standardiser, so the Trainer fills it in -- and only for a "
            "log target mode. Set data.target_mode=log_ratio.")
    mid = (len(list(cfg.get("quantiles", []) or [])) // 2
           if str(cfg.get("name", "")).lower() in ("hybrid", "quantile") else None)
    return ZoneBlend(base, hw, float(cfg.get("zone_tau", 0.15)), w, mid)
