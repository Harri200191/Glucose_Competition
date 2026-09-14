"""Losses.

All of them are **masked**: the four horizons have slightly different sample
counts (a window can have a real reading at +30 min and none at +120), so a loss
that ignored the mask would train on fabricated zeros.

Losses operate in model space (standardised delta or absolute), which puts the
horizons on a comparable scale -- the standard deviation of the 120-minute change
is roughly three times that of the 30-minute change, so an unscaled multi-horizon
loss would silently become a 120-minute loss.

THE BAND, AND WHY SO MUCH OF THIS FILE IS ABOUT IT
--------------------------------------------------
The scorer's borders reduce exactly (``evaluation/metrics.py``): above a true 62.5
mg/dL, DTS zone A is ``0.8y <= p <= 1.2y``. In log space that is a fixed-width band
that is **not centred on zero**::

    log 0.8 <= log(p/y) <= log 1.2       i.e.   |log(p/y) - CENTRE| <= HALFWIDTH
    CENTRE    = -0.020411                the band's midpoint
    HALFWIDTH = +0.202733

Two consequences drive everything below.

**The band is off-centre.** For any symmetric unimodal predictive law the
zone-A-optimal point forecast is the conditional log-median plus ``CENTRE``,
*independent of the spread*. Measured: the closed form ``exp(CENTRE - mu)`` matches
a grid search for the best multiplier to 0.001-0.004 at every horizon
(``scripts/analysis/_objective_probe.py``). ``band_offset`` puts that constant into
training instead of correcting for it afterwards, which matters because the
post-hoc fit does not transfer -- +0.225 on test, **-0.106** on the 89 unseen
holdout patients, where the model's bias flips sign.

**The band is bounded.** Once a prediction is outside it, being further outside
costs the metric nothing, but a squared or linear loss keeps paying attention in
proportion to the error. Share of total gradient landing on anchors already outside
zone A, measured on ``runs/ss_ctrl_s0`` at h=120::

    squared error  65.6%      Huber delta=0.5  51.2%   <- ox_half, +0.28 measured
    abs error      30.6%      Welsch 1.0R      33.5%
    Huber delta=1  62.4%      Tukey  2.0R      24.3%

``ox_half`` is the only loss change that ever beat the noise floor in this project
and it sits in the *middle* of that ladder, which is why ``Welsch`` and ``Tukey``
exist here. The obvious objection -- that a bounded loss abandons the hard cases --
is answered by the distances: at h=120, 46% of the failing anchors are within half
a band-width of passing and only 8.4% are beyond two band-widths.

Everything band-aware needs ``r`` and ``c`` in MODEL space, which depends on the
fitted standardiser, so the Trainer derives them once via ``band_constants`` and
injects them. Only a log target mode makes any of it meaningful.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from ..common import studentt as _t

# Scorer geometry, derived rather than typed. Same constants as
# submission/recentre.py:CENTRE -- a property of the grid, not a fitted quantity,
# which is why they are module constants and not config keys.
ZONE_A_LOG_HALFWIDTH: float = (math.log(1.2) - math.log(0.8)) / 2.0   # 0.202733
ZONE_A_LOG_CENTRE: float = (math.log(1.2) + math.log(0.8)) / 2.0      # -0.020411

LOG_TARGET_MODES = ("log_ratio", "log")
# Losses that need the band in model space, so the Trainer knows what to inject
# without learning anything about individual loss classes.
_BAND_LOSSES = ("welsch", "tukey", "zone_t")
# What each loss wants handed to it. The Trainer keys off this instead of the
# architecture or the loss's name.
DIST_KEYS = ("dist_point", "dist_mu", "dist_sigma", "dist_nu", "dist_log_pi")


def band_constants(target_scale, target_mode: str) -> Dict[str, list]:
    """``HALFWIDTH`` and ``CENTRE`` per horizon, in model space. The ONE derivation.

    Under a log target mode ``log(p/y) = target_scale * (pred - target)`` exactly:
    ``target_center`` cancels between the two, which is why it does not appear
    here. That is deliberate, not an omission.

    With the measured ``target_scale`` of [0.195, 0.298, 0.361, 0.403] the band
    half-width in standardised units is [1.04, 0.68, 0.56, 0.50] -- so a single
    Huber ``delta=1.0`` sits exactly on the boundary at 30 min and twice beyond it
    at 120 min, which is the mismatch this whole module is about.
    """
    if target_mode not in LOG_TARGET_MODES:
        raise ValueError(
            f"a band-aware loss needs a log target mode (got "
            f"data.target_mode={target_mode!r}): zone A is a RELATIVE band, so it "
            "is only a constant-width window on the model's output in log space. "
            "Set data.target_mode=log_ratio.")
    s = np.maximum(np.asarray(target_scale, dtype=np.float64), 1e-6)
    return {"band_halfwidth": (ZONE_A_LOG_HALFWIDTH / s).tolist(),
            "band_centre": (ZONE_A_LOG_CENTRE / s).tolist()}


def needs_band(cfg: Dict) -> bool:
    """Does this loss config need the model-space band injected?"""
    name = str(cfg.get("name", "huber")).lower()
    return (name in _BAND_LOSSES
            or bool(cfg.get("band_offset", False))
            or float(cfg.get("zone_weight", 0.0) or 0.0) > 0)


def loss_inputs(criterion: nn.Module, out) -> object:
    """``ForecastOutput`` -> whatever this criterion consumes.

    Keys off the loss's declared ``input_kind``, never off which architecture
    produced the output -- that is the invariant the Trainer has to keep.
    """
    kind = getattr(criterion, "input_kind", "point")
    if kind == "point":
        return out.pred_norm
    if kind == "quantiles":
        if out.quantiles is None:
            raise ValueError(
                "this loss consumes the quantile tensor but the model emitted "
                "none; set model.head.quantiles to match training.loss.quantiles")
        return out.quantiles
    if kind == "dist":
        missing = [k for k in DIST_KEYS if k not in (out.aux or {})]
        if missing:
            raise ValueError(
                f"training.loss.name=zone_t consumes a predictive law and the "
                f"model did not emit {missing}. Set model.head.law.name=student_t "
                "(a FrontEndForecaster head; native models do not carry a law).")
        return {k: out.aux[k] for k in DIST_KEYS}
    raise ValueError(f"unknown loss input_kind {kind!r}")


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum().clamp(min=1.0)
    return (x * mask).sum() / denom


def _zone_surrogate(torch, pred, target, mask, halfwidth, tau, centre=None):
    """Smooth stand-in for "is this prediction inside zone A?".

    Zone A is |log(pred/true) - CENTRE| <= HALFWIDTH -- a hard 0/1 indicator with
    no gradient anywhere. The standard remedy is to replace the indicator with a
    sigmoid (arXiv 2305.02024), giving soft_hit = sigmoid((h - |resid|) / tau),
    and to minimise 1 - soft_hit.

    ``halfwidth`` and ``centre`` are in MODEL space, so the log-space window has to
    be divided by the target standardiser's scale; the Trainer does that and passes
    them in. ``centre`` is opt-in via ``band_offset`` so that an existing
    ``zone_weight`` run reproduces: before this module it was implicitly zero, i.e.
    the surrogate stood in for a band centred 2% off the real one.

    Blended with the ordinary loss, never used alone: on its own the surrogate
    is flat wherever a prediction sits far outside the band, so it supplies no
    gradient in exactly the cases that are worst. ``ZoneStudentT`` is the version
    that does not have that problem, because a Student-t tail decays polynomially
    rather than exponentially.
    """
    resid = pred - target
    if centre is not None:
        resid = resid - centre
    soft_hit = torch.sigmoid((halfwidth - resid.abs()) / max(tau, 1e-6))
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

    MEASURED NEGATIVE and kept only so the result stays reproducible: -0.71 at
    strength 0.5 and -1.83 at 1.5 (`results/subsample/`). It buys the tails
    (big fall +4.95, big rise +1.26 at h=120) by selling the flat middle (-3.41),
    and the middle is 42% of anchors against the tails' 25%.
    """
    if strength <= 0:
        return None
    w = 1.0 + strength * target.detach().abs()
    return w / w.mean().clamp_min(1e-6)


def _huber(err: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Huber with a PER-HORIZON delta. ``delta`` broadcasts against ``err``."""
    a = err.abs()
    return torch.where(a <= delta, 0.5 * err * err, delta * (a - 0.5 * delta))


class _PointLoss(nn.Module):
    """Shared plumbing for every loss that scores a point forecast.

    Order of operations is fixed here so the arms differ in the penalty shape and
    in nothing else: residual -> band offset -> penalty -> horizon weights ->
    move weights -> masked mean.
    """

    input_kind = "point"

    def __init__(self, horizon_weights: Optional[Sequence[float]] = None,
                 move_weight: float = 0.0,
                 band_centre: Optional[Sequence[float]] = None) -> None:
        super().__init__()
        self.move_weight = float(move_weight)
        self.register_buffer(
            "w", torch.tensor(list(horizon_weights), dtype=torch.float32)
            if horizon_weights else torch.ones(1))
        self.register_buffer(
            "centre", torch.tensor(list(band_centre), dtype=torch.float32)
            if band_centre is not None else torch.zeros(0))

    def _residual(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        err = pred - target
        if self.centre.numel() == pred.shape[1]:
            err = err - self.centre.to(pred.device).reshape(1, -1)
        return err

    def _reduce(self, loss: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        w = self.w.to(loss.device)
        if w.numel() == loss.shape[1]:
            loss = loss * w.reshape(1, -1)
        mw = _move_weights(target, self.move_weight)
        if mw is not None:
            loss = loss * mw
        return masked_mean(loss, mask)

    def penalty(self, err: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        return self._reduce(self.penalty(self._residual(pred, target)), target, mask)


class MaskedHuber(_PointLoss):
    """Huber on standardised targets. Robust to the tail of sensor artefacts.

    ``delta`` accepts a list, one value per horizon, and that is not cosmetic.
    Targets are standardised per horizon (``target_scale`` = 0.20/0.30/0.36/0.40
    for 30/60/90/120 min), while zone A is a fixed RELATIVE band: accepted
    predictions are 0.8x to 1.2x the truth, i.e. |log ratio| <= ~0.2 whatever the
    horizon. Divide one by the other and the metric's decision boundary sits at

        1.04 / 0.68 / 0.56 / 0.50   standardised units, for 30/60/90/120 min

    so a single delta=1.0 lands exactly on the boundary at 30 min and twice
    beyond it at 120 min. Past the boundary the anchor has already left zone A
    and further error costs the metric nothing -- but a quadratic loss keeps
    paying attention proportional to the error. Measured on c96_mamba2's test
    predictions: at 120 min the 23.9% of anchors already outside zone A absorb
    **85.4%** of the squared error, and at 30 min -- the horizon that is already
    near its ceiling -- only 61%.

    Setting delta to the per-horizon boundary aligns where the loss stops growing
    with where the metric stops caring. Measured: the band vector itself gained
    +0.148 (inside noise), half the band +0.280, a plain scalar 0.5 +0.195. The
    per-horizon alignment is NOT distinguishable from simply using a smaller
    number; ``[0.52, 0.34, 0.28, 0.25]`` is the best-measured value and no claim
    is made about the mechanism. Huber still grows linearly past delta, which is
    what ``Welsch`` and ``Tukey`` change.
    """

    def __init__(self, delta: float = 1.0,
                 horizon_weights: Optional[Sequence[float]] = None,
                 move_weight: float = 0.0,
                 band_centre: Optional[Sequence[float]] = None) -> None:
        super().__init__(horizon_weights, move_weight, band_centre)
        d = [float(x) for x in delta] if isinstance(delta, (list, tuple)) else [float(delta)]
        self.register_buffer("delta", torch.tensor(d, dtype=torch.float32))

    def penalty(self, err: torch.Tensor) -> torch.Tensor:
        d = self.delta.to(err.device)
        d = d.reshape(1, -1) if d.numel() == err.shape[1] else d.reshape(1, 1)
        return _huber(err, d)


class MaskedMSE(_PointLoss):
    def penalty(self, err: torch.Tensor) -> torch.Tensor:
        return err ** 2


class MaskedMAE(_PointLoss):
    def penalty(self, err: torch.Tensor) -> torch.Tensor:
        return err.abs()


class Welsch(_PointLoss):
    """Redescending (correntropy) loss: ``s^2 (1 - exp(-e^2 / 2 s^2))``.

    The next rung past Huber on the ladder in the module docstring. Huber *caps*
    the gradient past ``delta``; Welsch drives it back to zero, so an anchor that
    has already left zone A stops competing for capacity with one that has not.
    At h=120 it spends 33.5% of its gradient on already-lost anchors against
    Huber(0.5)'s 51.2% and squared error's 65.6%.

    ``bandwidth`` is a MULTIPLE of the zone-A half-width in model space, so the
    knob means the same thing at every horizon even though the standardiser's
    scale does not. bandwidth=1.0 puts the inflection exactly on the band edge.

    rho(e) -> e^2/2 as e -> 0, matching MSE's curvature at the origin, so the
    learning rate keeps meaning what it meant.
    """

    def __init__(self, band_halfwidth: Sequence[float], bandwidth: float = 1.0,
                 horizon_weights: Optional[Sequence[float]] = None,
                 move_weight: float = 0.0,
                 band_centre: Optional[Sequence[float]] = None) -> None:
        super().__init__(horizon_weights, move_weight, band_centre)
        s = torch.tensor([float(bandwidth) * float(v) for v in band_halfwidth],
                         dtype=torch.float32)
        self.register_buffer("s", s)

    def penalty(self, err: torch.Tensor) -> torch.Tensor:
        s = self.s.to(err.device)
        s = s.reshape(1, -1) if s.numel() == err.shape[1] else s.reshape(1, 1)
        return s * s * (1.0 - torch.exp(-0.5 * (err / s) ** 2))


class Tukey(_PointLoss):
    """Tukey biweight: quadratic near zero, exactly flat past ``bandwidth * r``.

    The hard-cutoff end of the ladder. Where Welsch decays smoothly, Tukey gives
    an anchor beyond the cutoff *exactly* zero gradient. At bandwidth=2.0 that
    discards the 8.4% of failing anchors that sit more than two band-widths out at
    h=120 -- which is the intent, and the number to quote if it wins or loses.
    """

    def __init__(self, band_halfwidth: Sequence[float], bandwidth: float = 2.0,
                 horizon_weights: Optional[Sequence[float]] = None,
                 move_weight: float = 0.0,
                 band_centre: Optional[Sequence[float]] = None) -> None:
        super().__init__(horizon_weights, move_weight, band_centre)
        s = torch.tensor([float(bandwidth) * float(v) for v in band_halfwidth],
                         dtype=torch.float32)
        self.register_buffer("s", s)

    def penalty(self, err: torch.Tensor) -> torch.Tensor:
        s = self.s.to(err.device)
        s = s.reshape(1, -1) if s.numel() == err.shape[1] else s.reshape(1, 1)
        u = (err / s).abs().clamp(max=1.0)
        return (s * s / 6.0) * (1.0 - (1.0 - u * u) ** 3)


class QuantileLoss(nn.Module):
    """Pinball loss for the optional probabilistic head (plan stage 5, later phase).

    Deliberately NOT band-offset: pinball at level q targets the conditional
    q-quantile, and shifting its target would simply make it estimate a different
    quantile. The band offset belongs on the point/decision, which is what
    ``MaskedHuber`` and ``ZoneStudentT`` carry.
    """

    input_kind = "quantiles"

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

    input_kind = "point"

    def __init__(self, base: nn.Module, halfwidth, tau: float,
                 weight: float, mid, centre=None) -> None:
        super().__init__()
        self.base, self.tau, self.weight, self.mid = base, tau, weight, mid
        self.input_kind = getattr(base, "input_kind", "point")
        self.register_buffer("hw", torch.as_tensor(halfwidth, dtype=torch.float32))
        self.register_buffer(
            "centre", torch.as_tensor(centre, dtype=torch.float32)
            if centre is not None else torch.zeros(0))

    def forward(self, pred, target, mask):
        point = pred if self.mid is None else pred[..., self.mid]
        hw = self.hw.to(point.device)
        if hw.numel() == point.shape[1]:
            hw = hw.reshape(1, -1)
        c = None
        if self.centre.numel() == point.shape[1]:
            c = self.centre.to(point.device).reshape(1, -1)
        return self.base(pred, target, mask) + self.weight * _zone_surrogate(
            torch, point, target, mask, hw, self.tau, c)


class HybridLoss(nn.Module):
    """Huber on the point forecast + pinball on the quantiles.

    Receives the quantile tensor ``[B, H, Q]``. The head constructs quantiles as
    monotone offsets around the point forecast with the centre offset zero, so
    the middle column *is* the point forecast -- Huber on it trains the point
    exactly as the plain loss would, while the pinball term shapes the rest of
    the distribution. ``alpha`` weights the pinball term.

    ONE TENSION, STATED. With ``band_offset`` on, the Huber term pulls the middle
    column to ``median + CENTRE`` while the pinball's q50 term pulls it to the
    median: the same column has two targets 2% apart. Huber wins by roughly 28:1
    on gradient magnitude (delta up to 1.0 against alpha/Q * 0.5 = 0.036), so the
    column lands essentially where the band wants it -- but this is a compromise,
    not a design. ``ZoneStudentT`` resolves it properly by emitting the decision
    and the law as separate outputs, which is why ``lo_band`` is the cheap
    approximation of ``lo_t`` rather than a rival to it.
    """

    input_kind = "quantiles"

    def __init__(self, quantiles: Sequence[float], delta=1.0,
                 alpha: float = 1.0,
                 horizon_weights: Optional[Sequence[float]] = None,
                 move_weight: float = 0.0,
                 band_centre: Optional[Sequence[float]] = None) -> None:
        super().__init__()
        self.point = MaskedHuber(delta, horizon_weights, move_weight, band_centre)
        self.pinball = QuantileLoss(quantiles, horizon_weights, move_weight)
        self.mid = len(list(quantiles)) // 2
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        return self.point(pred[..., self.mid], target, mask) \
            + self.alpha * self.pinball(pred, target, mask)


class ZoneStudentT(nn.Module):
    """The exact expected-zone-A objective, under a Student-t predictive law.

    Zone A is a 0/1 indicator: piecewise constant, gradient zero almost
    everywhere, undefined on the boundary. It cannot be optimised. Its
    **expectation under a predictive law** can, and for a t it is closed form.

    In model space, with ``R = HALFWIDTH/scale``, ``C = CENTRE/scale`` and a
    decision ``v``, the anchor scores iff the truth lands in ``[v-C-R, v-C+R]``, so

        P(zone A) = SUM_k pi_k * [ F_nu_k((v-C+R - mu_k)/sigma_k)
                                 - F_nu_k((v-C-R - mu_k)/sigma_k) ]

    which is smooth in every parameter. No surrogate, no temperature. The t is not
    an arbitrary family: fitted to this project's own log-residual it reproduces
    the observed zone-A rate to 0.27-1.17 points at every horizon, where a Gaussian
    is wrong by up to 8.71 (see ``common/studentt.py``).

    TWO DESIGN DECISIONS THAT ARE EASY TO GET WRONG.

    ``zone_obj='prob'`` (minimise ``1 - P``) is the default, not ``'logprob'``.
    ``-log P`` grows like ``nu*log|e|`` in the tail, so its gradient decays only as
    ``1/|e|`` and it spends 59.4% of its gradient on already-lost anchors at h=120
    -- *worse* than plain Huber's 51.2%, and the opposite of the point. ``1 - P``
    decays with the tail density and spends 36.3%. ``logprob`` is kept as an arm
    and is predicted to lose, which is a falsifiable claim rather than a hedge.

    ``nll_weight`` is load-bearing and defaults to 1.0. The band term alone is
    degenerate: it drives ``sigma -> 0`` wherever the model is right and
    ``sigma -> |residual|`` wherever it is wrong, which destroys the quantile
    ladder that ``submission/decide.py`` and ``scripts/analysis/_calibration.py``
    consume. A proper scoring rule has to anchor it. ``nll_weight=0`` exists only
    as a diagnostic arm -- and there ``val_loss`` reads directly as the miss rate,
    since ``mean(1-P)`` is an unbiased estimate of ``1 - zone_A/100``.
    """

    input_kind = "dist"

    def __init__(self, band_halfwidth: Sequence[float],
                 band_centre: Sequence[float],
                 nll_weight: float = 1.0, zone_obj: str = "prob",
                 zone_obj_weight: float = 1.0, point_weight: float = 0.0,
                 sharpness_weight: float = 0.0, delta=1.0, quad_nodes: int = 32,
                 horizon_weights: Optional[Sequence[float]] = None,
                 move_weight: float = 0.0) -> None:
        super().__init__()
        if zone_obj not in ("prob", "logprob"):
            raise ValueError(f"training.loss.zone_obj must be prob|logprob, got {zone_obj!r}")
        self.zone_obj = zone_obj
        self.nll_weight = float(nll_weight)
        self.zone_obj_weight = float(zone_obj_weight)
        self.point_weight = float(point_weight)
        self.sharpness_weight = float(sharpness_weight)
        self.move_weight = float(move_weight)
        self.register_buffer("R", torch.tensor([float(v) for v in band_halfwidth],
                                               dtype=torch.float32))
        self.register_buffer("C", torch.tensor([float(v) for v in band_centre],
                                               dtype=torch.float32))
        self.register_buffer(
            "w", torch.tensor(list(horizon_weights), dtype=torch.float32)
            if horizon_weights else torch.ones(1))
        x, wq = _t.quadrature(int(quad_nodes), dtype=torch.float32)
        self.register_buffer("qx", x)
        self.register_buffer("qw", wq)
        self.point = MaskedHuber(delta, horizon_weights, move_weight,
                                 band_centre) if point_weight > 0 else None

    def _reduce(self, per_horizon: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        w = self.w.to(per_horizon.device)
        if w.numel() == per_horizon.shape[1]:
            per_horizon = per_horizon * w.reshape(1, -1)
        mw = _move_weights(target, self.move_weight)
        if mw is not None:
            per_horizon = per_horizon * mw
        return masked_mean(per_horizon, mask)

    def forward(self, dist: Dict[str, torch.Tensor], target: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        # lgamma/log/exp are not on the autocast allow-list and the tail terms are
        # where precision matters, so the whole objective runs in fp32.
        with torch.amp.autocast(target.device.type, enabled=False):
            v = dist["dist_point"].float()                  # [B,H] the decision
            mu = dist["dist_mu"].float()                    # [B,H,K]
            sigma = dist["dist_sigma"].float().clamp_min(1e-4)
            nu = dist["dist_nu"].float()
            log_pi = dist["dist_log_pi"].float()
            y = target.float().unsqueeze(-1)
            R = self.R.to(v.device).reshape(1, -1, 1)
            C = self.C.to(v.device).reshape(1, -1, 1)
            total = target.new_zeros(())

            if self.nll_weight:
                z = (y - mu) / sigma
                lp = torch.logsumexp(log_pi + _t.log_pdf(z, nu) - torch.log(sigma), -1)
                total = total + self.nll_weight * self._reduce(-lp, target, mask)

            if self.zone_obj_weight:
                lo = (v.unsqueeze(-1) - C - R - mu) / sigma
                hi = (v.unsqueeze(-1) - C + R - mu) / sigma
                pk = _t.band_prob(lo, hi, nu, self.qx.to(v.device), self.qw.to(v.device))
                p = (log_pi.exp() * pk).sum(-1).clamp(1e-7, 1.0)
                obj = (1.0 - p) if self.zone_obj == "prob" else -torch.log(p)
                total = total + self.zone_obj_weight * self._reduce(obj, target, mask)

            if self.sharpness_weight:
                # Explicit sharpness/calibration trade, with one interpretable
                # coefficient, instead of hoping weight decay does it indirectly.
                s = torch.logsumexp(log_pi + torch.log(sigma), -1)
                total = total + self.sharpness_weight * self._reduce(s, target, mask)

            if self.point is not None:
                total = total + self.point_weight * self.point(v, target, mask)
        return total


def build_loss(cfg: Dict) -> nn.Module:
    """Build the loss named in ``training.loss``.

    ``horizon_weights`` re-weights the per-horizon terms, and is the honest way
    to spend capacity where the score is. The scorer pools all four horizons, but
    30 min already sits near its ceiling while 120 min has several times the
    headroom, so an unweighted loss under-invests in the horizon that decides the
    result. Pass four numbers in the order of ``data.horizons_min``; they are
    applied before the mask, so a horizon with no target still contributes
    nothing. Honoured by every loss -- which was previously claimed but not true
    of ``mse`` and ``mae``.

    ``band_halfwidth`` / ``band_centre`` are injected by the Trainer, never by a
    config file: they depend on the fitted standardiser.
    """
    name = str(cfg.get("name", "huber")).lower()
    hw = cfg.get("horizon_weights") or None
    if hw is not None:
        hw = [float(x) for x in hw]
    mw = float(cfg.get("move_weight", 0.0) or 0.0)
    # delta may be a scalar or one value per horizon; MaskedHuber documents why.
    _d = cfg.get("delta", 1.0)
    delta = [float(x) for x in _d] if isinstance(_d, (list, tuple)) else float(_d)
    centre = cfg.get("band_centre") if cfg.get("band_offset", False) else None
    if cfg.get("band_offset", False) and centre is None:
        raise ValueError(
            "training.loss.band_offset is set but band_centre is missing. It is "
            "the zone-A band's midpoint in MODEL space, which depends on the "
            "target standardiser, so the Trainer fills it in -- and only for a "
            "log target mode. Set data.target_mode=log_ratio.")

    def _band(key: str):
        v = cfg.get(key)
        if v is None:
            raise ValueError(
                f"training.loss.name={name!r} needs {key}, the zone-A band in "
                "MODEL space. The Trainer injects it from the fitted "
                "standardiser, and only for a log target mode. Set "
                "data.target_mode=log_ratio.")
        return v

    if name == "huber":
        return MaskedHuber(delta, hw, mw, centre)
    if name == "mse":
        return MaskedMSE(hw, mw, centre)
    if name == "mae":
        return MaskedMAE(hw, mw, centre)
    if name == "welsch":
        return Welsch(_band("band_halfwidth"), float(cfg.get("bandwidth", 1.0)),
                      hw, mw, centre)
    if name == "tukey":
        return Tukey(_band("band_halfwidth"), float(cfg.get("bandwidth", 2.0)),
                     hw, mw, centre)
    if name == "quantile":
        return QuantileLoss(cfg.get("quantiles", [0.1, 0.5, 0.9]), hw, mw)
    if name == "hybrid":
        return HybridLoss(cfg.get("quantiles", [0.1, 0.5, 0.9]), delta,
                          float(cfg.get("alpha", 1.0)), hw, mw, centre)
    if name == "zone_t":
        return ZoneStudentT(
            _band("band_halfwidth"), _band("band_centre"),
            nll_weight=float(cfg.get("nll_weight", 1.0)),
            zone_obj=str(cfg.get("zone_obj", "prob")),
            zone_obj_weight=float(cfg.get("zone_obj_weight", 1.0)),
            point_weight=float(cfg.get("point_weight", 0.0)),
            sharpness_weight=float(cfg.get("sharpness_weight", 0.0)),
            delta=delta, quad_nodes=int(cfg.get("quad_nodes", 32)),
            horizon_weights=hw, move_weight=mw)
    raise ValueError(f"unknown loss {name!r}")


def build_loss_with_zone(cfg: Dict) -> nn.Module:
    """``build_loss`` plus the optional zone-A sigmoid surrogate term."""
    base = build_loss(cfg)
    w = float(cfg.get("zone_weight", 0.0) or 0.0)
    if w <= 0:
        return base
    hw = cfg.get("band_halfwidth")
    if hw is None:
        raise ValueError(
            "training.loss.zone_weight is set but band_halfwidth is missing. It "
            "is the zone-A half-width in MODEL space, which depends on the "
            "target standardiser, so the Trainer fills it in -- and only for a "
            "log target mode. Set data.target_mode=log_ratio.")
    if getattr(base, "input_kind", "point") == "dist":
        raise ValueError(
            "training.loss.zone_weight blends a SIGMOID stand-in for zone A onto "
            "a base loss. training.loss.name=zone_t already optimises the exact "
            "expected zone A in closed form, so the surrogate is strictly worse "
            "and would double-count. Set zone_weight=0.")
    mid = (len(list(cfg.get("quantiles", []) or [])) // 2
           if str(cfg.get("name", "")).lower() in ("hybrid", "quantile") else None)
    centre = cfg.get("band_centre") if cfg.get("band_offset", False) else None
    return ZoneBlend(base, hw, float(cfg.get("zone_tau", 0.15)), w, mid, centre)
