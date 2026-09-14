"""A differentiable Student-t CDF, and the band probability the scorer needs.

WHY THIS FILE EXISTS
--------------------
DTS zone A is a 0/1 indicator, so it has no gradient. Its *expectation under a
predictive law* does, and for a Student-t that expectation is a closed form:

    P(zone A) = F_nu((b - mu)/sigma) - F_nu((a - mu)/sigma)

That is the whole reason a distributional head is worth building -- it converts
the metric from something that cannot be optimised into something that can, with
no surrogate and no temperature.

The t is not an arbitrary choice. Fitting laws to this project's own pooled
log-residual and asking them to predict the observed zone-A rate
(``scripts/analysis/_objective_probe.py``, ``runs/ss_ctrl_s0``):

    h      observed   Student-t   Gaussian   fitted nu
    30      91.48       92.25       90.40       2.50
    60      81.09       82.26       74.05       2.89
    90      74.43       75.27       65.98       3.24
    120     69.39       69.66       60.68       3.67

A t reproduces the metric to within 1.2 points; a Gaussian is wrong by up to 8.7,
which is fourteen times the run-to-run noise floor. The heavy tail also fixes the
vanishing-gradient problem that ``losses._zone_surrogate`` documents: a t density
decays polynomially, so at three conditional sds outside the band it still
supplies roughly 1400x the gradient of a tau=0.05 sigmoid surrogate.

WHY WE WRITE OUR OWN
--------------------
torch 2.6 has no ``betainc`` (neither ``torch.betainc`` nor
``torch.special.betainc``), and ``torch.distributions.StudentT.cdf`` / ``.icdf``
both raise ``NotImplementedError``. ``torch.special.gammainc`` exists but is
differentiable only with respect to ``x``, not the shape parameter, so it cannot
carry a gradient into ``nu``.

THE METHOD
----------
Substitute ``t = sqrt(nu) * tan(phi)`` into the t density. Then
``1 + t^2/nu = sec^2(phi)`` and ``dt = sqrt(nu) sec^2(phi) dphi``, so

    f(t) dt = K(nu) * cos^(nu-1)(phi) dphi,
    K(nu)   = exp(lgamma((nu+1)/2) - lgamma(nu/2) - log(pi)/2)

and the upper tail becomes a proper integral over a finite interval:

    P(T > u) = K(nu) * INTEGRAL_{atan(u/sqrt(nu))}^{pi/2} cos^(nu-1)(phi) dphi

The integrand is smooth, bounded by 1 and monotone on that interval, so
fixed-node Gauss-Legendre converges spectrally. Measured against ``scipy.stats.t``
over nu in [2.05, 100] and u in [0, 100] (``scripts/analysis/_check_studentt.py``):

    nodes    max abs err    max rel err
      16       4.2e-07        1.8e-04
      24       3.2e-08        6.4e-08
      32       1.0e-08        2.0e-08      <- the default
      64       6.1e-10        1.2e-09

Everything is differentiable in ``mu``, ``sigma`` and ``nu``: the nodes and
weights are constants, so autograd flows through ``cos^(nu-1)`` and ``lgamma``.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch

# Below this the integrand's log blows up at phi -> pi/2; the clamp costs nothing
# because (nu-1) > 0 drives the term to zero there anyway.
_COS_FLOOR = 1e-30


def quadrature(n: int = 32, dtype=torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gauss-Legendre nodes and weights on [-1, 1], as plain tensors.

    Computed once and registered as buffers by the caller, so no host->device
    traffic happens per step.
    """
    x, w = np.polynomial.legendre.leggauss(int(n))
    return (torch.as_tensor(x, dtype=dtype), torch.as_tensor(w, dtype=dtype))


def log_norm_const(nu: torch.Tensor) -> torch.Tensor:
    """log K(nu) = lgamma((nu+1)/2) - lgamma(nu/2) - log(pi)/2."""
    return (torch.lgamma((nu + 1) * 0.5) - torch.lgamma(nu * 0.5)
            - 0.5 * math.log(math.pi))


def upper_tail(u: torch.Tensor, nu: torch.Tensor,
               x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """P(T > u) for u >= 0. Callers pass |u| and fix the sign themselves."""
    phi0 = torch.atan(u / torch.sqrt(nu))
    half = (math.pi / 2 - phi0) * 0.5
    mid = (math.pi / 2 + phi0) * 0.5
    phi = mid.unsqueeze(-1) + half.unsqueeze(-1) * x
    log_cos = torch.log(torch.cos(phi).clamp_min(_COS_FLOOR))
    integ = (torch.exp((nu.unsqueeze(-1) - 1.0) * log_cos) * w).sum(-1)
    return (torch.exp(log_norm_const(nu)) * half * integ).clamp(0.0, 1.0)


def cdf(z: torch.Tensor, nu: torch.Tensor,
        x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """F_nu(z), any sign."""
    tail = upper_tail(z.abs(), nu, x, w)
    return torch.where(z >= 0, 1.0 - tail, tail)


def band_prob(a: torch.Tensor, b: torch.Tensor, nu: torch.Tensor,
              x: torch.Tensor, w: torch.Tensor,
              eps: float = 1e-7) -> torch.Tensor:
    """P(a <= T <= b), assuming a <= b. Three branches, for a reason.

    The both-negative case must NOT be computed as ``(1-ga) - (1-gb)``: in
    float32 that subtracts two numbers close to 1 and loses the leading digits
    exactly where the band probability is small and the gradient matters. Each
    branch below is separately accurate in its own regime, and all three reuse
    the same two tail evaluations, so the extra accuracy is free.
    """
    ga = upper_tail(a.abs(), nu, x, w)
    gb = upper_tail(b.abs(), nu, x, w)
    p = torch.where(a >= 0, ga - gb,
                    torch.where(b <= 0, gb - ga, 1.0 - ga - gb))
    return p.clamp_min(eps)


def log_pdf(z: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
    """log f_nu(z) for the standardised variable; add -log(sigma) for a scaled one."""
    return (torch.lgamma((nu + 1) * 0.5) - torch.lgamma(nu * 0.5)
            - 0.5 * torch.log(math.pi * nu)
            - 0.5 * (nu + 1) * torch.log1p(z * z / nu))


def icdf(p: torch.Tensor, nu: torch.Tensor, x: torch.Tensor, w: torch.Tensor,
         iters: int = 48, lo: float = -1e3, hi: float = 1e3) -> torch.Tensor:
    """F_nu^{-1}(p), differentiable in ``nu``.

    Bisection gives a root but a piecewise-constant one, so its gradient is zero.
    The fix is the standard implicit-function trick: bisect under ``no_grad``,
    detach, then take ONE Newton step. The step is a no-op numerically (the root
    is already converged) but it reattaches the correct derivative,
    ``dz/dnu = -(dF/dnu)/f``. Verified against central finite differences to six
    decimals, and against ``scipy.stats.t.ppf`` to 2.8e-08 at nu=2.05.
    """
    with torch.no_grad():
        l = torch.full_like(p, lo)
        h = torch.full_like(p, hi)
        for _ in range(iters):
            m = (l + h) * 0.5
            below = cdf(m, nu, x, w) < p
            l = torch.where(below, m, l)
            h = torch.where(below, h, m)
        z = ((l + h) * 0.5).detach()
    return z - (cdf(z, nu, x, w) - p) / torch.exp(log_pdf(z, nu)).clamp_min(1e-30)


def mixture_icdf(p: float, log_pi: torch.Tensor, mu: torch.Tensor,
                 sigma: torch.Tensor, nu: torch.Tensor,
                 x: torch.Tensor, w: torch.Tensor,
                 iters: int = 48) -> torch.Tensor:
    """Quantile of a K-component location-scale t mixture, over the last axis.

    A mixture CDF is a convex combination of monotone functions, so it is
    monotone and bisection is valid. One Newton step with the mixture density
    reattaches the gradient, exactly as in ``icdf``.
    """
    pi = log_pi.exp()

    def _cdf(z):
        return (pi * cdf((z.unsqueeze(-1) - mu) / sigma, nu, x, w)).sum(-1)

    def _pdf(z):
        s = (z.unsqueeze(-1) - mu) / sigma
        return (pi * torch.exp(log_pdf(s, nu)) / sigma).sum(-1)

    base = mu.detach()
    scale = sigma.detach()
    with torch.no_grad():
        l = (base - 40.0 * scale).min(-1).values
        h = (base + 40.0 * scale).max(-1).values
        for _ in range(iters):
            m = (l + h) * 0.5
            below = _cdf(m) < p
            l = torch.where(below, m, l)
            h = torch.where(below, h, m)
        z = ((l + h) * 0.5).detach()
    return z - (_cdf(z) - p) / _pdf(z).clamp_min(1e-30)
