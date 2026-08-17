"""Fusion strategies (plan section 5) -- the core methodological contribution.

Four settings, deliberately nested so the ablation is a ladder rather than four
unrelated models:

``raw_concat``      normalised channels concatenated per time step -> linear -> backbone.
                    The required baseline: it answers whether any of the rest is needed.
``concat``          per-modality shallow encoders, concatenated -> linear.
                    Isolates "modality-specific encoding" from "learned mixing".
``gated``           per-modality gates g_m(t) = sigma(W_m E_m(t) + b_m), then
                    concatenate the gated embeddings and project. The network can
                    suppress a modality when it is uninformative.
``film``            CGM-centred conditioning. Drivers form a context c(t); the
                    context generates feature-wise scale and shift that modulate
                    the CGM state embedding:  E_G_bar = gamma(c) * E_G + beta(c).

``film`` states the structural hypothesis explicitly: CGM is the endogenous state
and insulin/carbohydrate/activity are interventions that modify how that state
evolves. It is an inductive bias, not a pharmacokinetic model, and the ablation
ladder is what decides whether it earns its place.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.registry import register_fusion


class FusionBase(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model

    def forward(self, encoded: Dict[str, torch.Tensor],
                static: Optional[torch.Tensor] = None,
                time: Optional[torch.Tensor] = None,
                masks: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        raise NotImplementedError


def _broadcast_static(static: Optional[torch.Tensor], length: int) -> Optional[torch.Tensor]:
    if static is None:
        return None
    return static.unsqueeze(1).expand(-1, length, -1)


@register_fusion("concat")
class ConcatFusion(FusionBase):
    """Separate encoders, concatenation, one projection. No learned mixing."""

    def __init__(self, d_model: int, modalities: List[str], d_static: int = 0,
                 d_time: int = 0, dropout: float = 0.0) -> None:
        super().__init__(d_model)
        in_dim = d_model * len(modalities) + d_static + d_time
        self.proj = nn.Sequential(nn.Linear(in_dim, d_model), nn.Dropout(dropout))
        self.norm = nn.LayerNorm(d_model)
        self.modalities = list(modalities)

    def forward(self, encoded, static=None, time=None, masks=None):
        length = encoded[self.modalities[0]].shape[1]
        parts = [encoded[m] for m in self.modalities]
        st = _broadcast_static(static, length)
        if st is not None:
            parts.append(st)
        if time is not None:
            parts.append(time)
        return self.norm(self.proj(torch.cat(parts, dim=-1)))


@register_fusion("gated")
class GatedFusion(FusionBase):
    """z(t) = W_f [ g_G*E_G ; g_I*E_I ; g_C*E_C ; g_A*E_A ; z_static ]."""

    def __init__(self, d_model: int, modalities: List[str], d_static: int = 0,
                 d_time: int = 0, dropout: float = 0.0) -> None:
        super().__init__(d_model)
        self.modalities = list(modalities)
        self.gates = nn.ModuleDict({m: nn.Linear(d_model, d_model) for m in self.modalities})
        in_dim = d_model * len(self.modalities) + d_static + d_time
        self.proj = nn.Sequential(nn.Linear(in_dim, d_model), nn.Dropout(dropout))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, encoded, static=None, time=None, masks=None):
        length = encoded[self.modalities[0]].shape[1]
        parts: List[torch.Tensor] = []
        gate_values: Dict[str, torch.Tensor] = {}
        for m in self.modalities:
            e = encoded[m]
            g = torch.sigmoid(self.gates[m](e))
            gate_values[m] = g
            parts.append(g * e)
        st = _broadcast_static(static, length)
        if st is not None:
            parts.append(st)
        if time is not None:
            parts.append(time)
        out = self.norm(self.proj(torch.cat(parts, dim=-1)))
        # Gate activations are kept for interpretation: "which modality is the
        # model listening to around a meal?" is a question the plan asks (E6).
        self.last_gates = {m: g.detach().mean(dim=-1) for m, g in gate_values.items()}
        return out


@register_fusion("film")
class FiLMFusion(FusionBase):
    """CGM-centred FiLM conditioning: interventions modulate the state embedding.

    ``driver_concat`` keeps a compact projection of the driver context alongside
    the modulated state, as the plan allows: FiLM alone can only scale and shift,
    so a driver signal with no multiplicative interpretation would otherwise be
    lost. Setting it to False gives the pure form.
    """

    def __init__(self, d_model: int, modalities: List[str], d_static: int = 0,
                 d_time: int = 0, dropout: float = 0.0, d_context: Optional[int] = None,
                 driver_concat: bool = True, driver_dim: Optional[int] = None) -> None:
        super().__init__(d_model)
        self.modalities = list(modalities)
        self.drivers = [m for m in self.modalities if m != "cgm"]
        d_context = d_context or d_model
        self.driver_concat = driver_concat and bool(self.drivers)
        self.driver_dim = driver_dim or max(d_model // 4, 8)

        ctx_in = d_model * len(self.drivers) + d_static + d_time
        if ctx_in == 0:                      # CGM-only ablation: FiLM degenerates
            ctx_in = d_model                 # to a learned affine, which is fine
        self.context = nn.Sequential(
            nn.Linear(ctx_in, d_context), nn.GELU(), nn.Dropout(dropout),
            nn.LayerNorm(d_context),
        )
        # gamma is parameterised as 1 + delta so the identity is the initial state:
        # at initialisation FiLM leaves the CGM embedding untouched, and any
        # modulation the model learns is a departure it had to pay for.
        self.to_gamma = nn.Linear(d_context, d_model)
        self.to_beta = nn.Linear(d_context, d_model)
        nn.init.zeros_(self.to_gamma.weight); nn.init.zeros_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight); nn.init.zeros_(self.to_beta.bias)

        self.driver_proj = nn.Linear(d_context, self.driver_dim) if self.driver_concat \
            else None
        out_dim = d_model + (self.driver_dim if self.driver_concat else 0)
        self.proj = nn.Sequential(nn.Linear(out_dim, d_model), nn.Dropout(dropout))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, encoded, static=None, time=None, masks=None):
        state = encoded["cgm"]
        length = state.shape[1]
        parts: List[torch.Tensor] = [encoded[m] for m in self.drivers]
        st = _broadcast_static(static, length)
        if st is not None:
            parts.append(st)
        if time is not None:
            parts.append(time)
        ctx_in = torch.cat(parts, dim=-1) if parts else state
        c = self.context(ctx_in)

        gamma = 1.0 + self.to_gamma(c)
        beta = self.to_beta(c)
        conditioned = gamma * state + beta

        if self.driver_concat:
            conditioned = torch.cat([conditioned, self.driver_proj(c)], dim=-1)
        self.last_gamma = gamma.detach()
        return self.norm(self.proj(conditioned))


def build_fusion(name: str, d_model: int, modalities: List[str], d_static: int,
                 d_time: int, cfg: Dict) -> FusionBase:
    from ..common.registry import FUSIONS
    kwargs = dict(d_model=d_model, modalities=modalities, d_static=d_static,
                  d_time=d_time, dropout=float(cfg.get("dropout", 0.0)))
    if name == "film":
        kwargs["driver_concat"] = bool(cfg.get("driver_concat", True))
        kwargs["d_context"] = cfg.get("d_context")
        kwargs["driver_dim"] = cfg.get("driver_dim")
    return FUSIONS.build(name, **kwargs)
