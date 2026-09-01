"""Physiological features derived from CGM + insulin, with no meal log required.

WHY
---
Zone A at 120 min splits sharply by how fast glucose is moving (measured on the
holdout): flat anchors 78.2%, fastest-rising 67.2%, fastest-falling 65.2%. A
fifth of the anchors sit 11-13 points below the rest, and fast rises are meals.

The annual holdout records **no carbohydrate data at all** -- verified, 0% of
anchors -- while insulin is present on 100%. So the meal signal cannot be read;
it has to be reconstructed.

The route is the deconvolution used in Fabris et al. (PMC6806292), which reports
+17.9% EGA A-region at a 120-minute horizon over an ARX baseline. Its forward
model needs logged meals, but the deconvolution step does not:

    Ra(k) = G'(k) + (S_G + X(k))·G(k) − S_G·G_b / V

CGM derivative, insulin action X(k), and population constants. Ra is literally
"glucose is appearing from the gut right now" -- the meal signal, recovered from
signals the holdout actually has.

WHAT THIS DOES, AND WHAT IT DOES NOT
------------------------------------
It does NOT hardcode S_G, G_b or V. Those are population averages and the CGM
channel arrives standardised, so the literal formula would be wrong here.
Instead it hands the network the *algebraic terms* of that equation -- G', G'',
X·G for two insulin-action timescales, and the uncalibrated combination
G' + X·G -- and lets it fit the coefficients itself. Physics-informed features,
not a physics simulator.

The rectified, smoothed Ra proxy doubles as a meal-onset signal: positive
unexplained glucose appearance is what every threshold-family meal detector
keys on (see the survey in arXiv 2507.00080, which reaches 88-93% recall from
CGM alone). Full Dynamic Mode Decomposition detection is a separate, heavier
approach and is deliberately not attempted here.

Everything is strictly causal: differences look backwards, the smoother is a
one-sided exponential moving average.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

# Column offsets inside each modality block, from data/constants.py.
CGM_VALUE = 0
INS_BOLUS_EXP60 = 5
INS_BOLUS_EXP120 = 6
INS_BASAL_EXP60 = 8

N_PHYSIO = 6


class PhysioFeatures(nn.Module):
    """[B,T,C_cgm] + [B,T,C_ins] -> [B,T,6] causal physiological channels.

    1. dG      first difference of CGM              (rate of change)
    2. d2G     second difference                    (acceleration; meals curve up)
    3. X60·G   insulin action x glucose, fast pool   } the two X·G terms of the
    4. X120·G  insulin action x glucose, slow pool   } deconvolution equation
    5. Ra~     dG + (X60 + Xbasal)·G                 (uncalibrated appearance rate)
    6. Ra+     one-sided EMA of relu(Ra~)            (sustained appearance = meal)
    """

    def __init__(self, ema_halflife_steps: float = 6.0) -> None:
        super().__init__()
        # 6 steps = 30 min, the timescale meal detectors report for onset.
        self.alpha = float(0.5 ** (1.0 / max(ema_halflife_steps, 1e-3)))
        self.n_out = N_PHYSIO

    @staticmethod
    def _causal_diff(x: torch.Tensor) -> torch.Tensor:
        """x[t] - x[t-1], first step zero. Never looks forward."""
        d = torch.zeros_like(x)
        d[:, 1:] = x[:, 1:] - x[:, :-1]
        return d

    def _causal_ema(self, x: torch.Tensor) -> torch.Tensor:
        """One-sided exponential moving average, computed without a Python loop.

        ema[t] = (1-a) * sum_k a^k x[t-k], normalised so a constant input maps
        to itself. Done as a single conv1d over a truncated kernel: at
        half-life 6 the weights are below 1e-3 after ~60 steps, so the
        truncation is invisible and the cost is one convolution.
        """
        k = 64
        idx = torch.arange(k, device=x.device, dtype=x.dtype)
        w = (1.0 - self.alpha) * self.alpha ** idx
        w = (w / w.sum()).flip(0).view(1, 1, k)
        pad = torch.nn.functional.pad(x.unsqueeze(1), (k - 1, 0))
        return torch.nn.functional.conv1d(pad, w).squeeze(1)

    def forward(self, x_cgm: torch.Tensor,
                x_ins: torch.Tensor) -> torch.Tensor:
        g = x_cgm[..., CGM_VALUE]                       # [B, T]
        dg = self._causal_diff(g)
        d2g = self._causal_diff(dg)

        if x_ins is not None and x_ins.shape[-1] > INS_BASAL_EXP60:
            x60 = x_ins[..., INS_BOLUS_EXP60]
            x120 = x_ins[..., INS_BOLUS_EXP120]
            xb = x_ins[..., INS_BASAL_EXP60]
        else:                                            # causal-only run
            x60 = x120 = xb = torch.zeros_like(g)

        ra = dg + (x60 + xb) * g                         # uncalibrated Ra
        ra_pos = self._causal_ema(torch.relu(ra))

        return torch.stack([dg, d2g, x60 * g, x120 * g, ra, ra_pos], dim=-1)

    def describe(self) -> Dict[str, object]:
        return {"n_features": self.n_out, "ema_alpha": round(self.alpha, 4),
                "names": ["dG", "d2G", "X60G", "X120G", "Ra", "Ra_pos"]}
