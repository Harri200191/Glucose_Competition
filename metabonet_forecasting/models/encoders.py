"""Modality-specific shallow encoders (plan 4.1-4.6).

Design rule for all of them: stay shallow. The scientific question is whether
*representation and fusion* help, and a deep encoder would quietly take over the
sequence modelling that the backbone comparison is supposed to measure.

Causality rule: no operation may read a future sample. The CGM encoder therefore
uses left-padded convolutions, and every normalisation is per-feature
(LayerNorm over channels at a time step), never over time.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """1-D convolution that can only see the present and the past.

    Left-pads by ``(kernel_size - 1) * dilation`` and applies no right padding, so
    output ``t`` depends on inputs ``<= t``. ``tests/test_leakage.py`` verifies
    this numerically by perturbing future inputs and checking the output does not
    move.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1,
                 groups: int = 1) -> None:
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, groups=groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, C_in] -> [B, T, C_out]"""
        y = x.transpose(1, 2)
        if self.pad:
            y = F.pad(y, (self.pad, 0))
        return self.conv(y).transpose(1, 2)


class CGMEncoder(nn.Module):
    """The state encoder: two causal convolutions (kernels 5 and 3) + GELU.

    Optional explicit first and second differences are an *ablation*, not a
    mandatory input: the convolutions can learn them, and the plan asks that we
    check whether handing them over explicitly helps.

    ``missing_embedding`` implements plan 4.6: where the observation mask says the
    reading is absent, a learned vector is added rather than leaving the model to
    infer absence from a zero that could equally be a real mid-range value.
    """

    def __init__(self, n_channels: int, d_out: int, hidden: Optional[int] = None,
                 use_differences: bool = False, missing_embedding: bool = False,
                 dropout: float = 0.0) -> None:
        super().__init__()
        hidden = hidden or d_out
        self.use_differences = use_differences
        self.missing_embedding = missing_embedding
        in_ch = n_channels + (2 if use_differences else 0)
        self.conv1 = CausalConv1d(in_ch, hidden, kernel_size=5)
        self.conv2 = CausalConv1d(hidden, d_out, kernel_size=3)
        self.norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(dropout)
        self.missing = nn.Parameter(torch.zeros(d_out)) if missing_embedding else None

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: [B, T, C] normalised CGM channels; mask: [B, T, 1] observation flag."""
        if self.use_differences:
            d1 = torch.zeros_like(x[..., :1])
            d1[:, 1:] = x[:, 1:, :1] - x[:, :-1, :1]
            d2 = torch.zeros_like(d1)
            d2[:, 1:] = d1[:, 1:] - d1[:, :-1]
            x = torch.cat([x, d1, d2], dim=-1)
        h = F.gelu(self.conv1(x))
        h = self.drop(h)
        h = self.conv2(h)
        if self.missing is not None and mask is not None:
            absent = (1.0 - mask[..., :1])
            h = h + absent * self.missing
        return self.norm(h)


class DriverEncoder(nn.Module):
    """Shared shape for insulin / carbohydrate / activity encoders.

    A per-time-step MLP over the modality's channels, followed by an optional
    short causal convolution that lets a dose spike smear forward a little before
    fusion. The exposure kernels already carry most of that structure; the
    convolution is there so the model is not forced to rely on them.
    """

    def __init__(self, n_channels: int, d_out: int, hidden: Optional[int] = None,
                 kernel_size: int = 3, dropout: float = 0.0,
                 missing_embedding: bool = False) -> None:
        super().__init__()
        hidden = hidden or max(d_out, 32)
        self.mlp = nn.Sequential(
            nn.Linear(n_channels, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, d_out),
        )
        self.conv = CausalConv1d(d_out, d_out, kernel_size=kernel_size) \
            if kernel_size > 1 else None
        self.norm = nn.LayerNorm(d_out)
        self.missing = nn.Parameter(torch.zeros(d_out)) if missing_embedding else None

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.mlp(x)
        if self.conv is not None:
            h = h + F.gelu(self.conv(h))
        if self.missing is not None and mask is not None:
            # "this modality is not recorded for this patient at this time"
            absent = (1.0 - mask.amax(dim=-1, keepdim=True))
            h = h + absent * self.missing
        return self.norm(h)


class StaticEncoder(nn.Module):
    """Small MLP over numeric statics plus embeddings for categorical ones.

    Kept small on purpose (plan 4.5, section 2.1): static characteristics should
    modulate the baseline response, not dominate the temporal representation.
    Source-dataset identity is excluded unless a config explicitly turns it on for
    the shortcut sensitivity analysis.
    """

    def __init__(self, n_numeric: int, cardinalities: Sequence[int], d_out: int,
                 emb_dim: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(c, emb_dim) for c in cardinalities])
        in_dim = n_numeric + emb_dim * len(cardinalities)
        self.enabled = in_dim > 0
        self.mlp = nn.Sequential(
            nn.Linear(max(in_dim, 1), d_out), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_out, d_out), nn.LayerNorm(d_out),
        )
        self.d_out = d_out

    def forward(self, numeric: torch.Tensor, categorical: torch.Tensor) -> torch.Tensor:
        parts: List[torch.Tensor] = []
        if numeric.numel():
            parts.append(numeric)
        for i, emb in enumerate(self.embeddings):
            parts.append(emb(categorical[:, i].clamp(min=0, max=emb.num_embeddings - 1)))
        if not parts:
            return torch.zeros(categorical.shape[0], self.d_out, device=categorical.device)
        return self.mlp(torch.cat(parts, dim=-1))


class TimeEncoder(nn.Module):
    """Cyclic calendar features -> a small per-step embedding.

    Absolute calendar time is never an input: timestamps are shifted per study and
    the corpus maximum is in the future, so cross-cohort calendar alignment is
    fiction. Time-of-day is real physiology (a 15-25 mg/dL diurnal swing) and is
    encoded cyclically so 23:55 sits next to 00:00.
    """

    def __init__(self, n_features: int, d_out: int) -> None:
        super().__init__()
        self.enabled = n_features > 0
        self.proj = nn.Linear(n_features, d_out) if self.enabled else None
        self.d_out = d_out

    def forward(self, time: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if not self.enabled or time is None:
            return None
        return self.proj(time)


class FutureActionEncoder(nn.Module):
    """Encodes the planned-intervention block that follows the anchor.

    Only constructed when a run explicitly opts into future covariates; the
    default benchmark never builds it, so the causal model is bit-for-bit what it
    was before the option existed.

    The block is flattened rather than pooled: *when* a dose lands inside the
    forecast span matters as much as how much, and a sum over the window would
    throw that away. At 24 steps and a handful of channels this is a small
    matrix, so there is no reason to compress it.
    """

    def __init__(self, n_steps: int, n_channels: int, d_out: int,
                 hidden: Optional[int] = None, dropout: float = 0.0) -> None:
        super().__init__()
        self.n_steps, self.n_channels = n_steps, n_channels
        hidden = hidden or max(d_out, 64)
        self.net = nn.Sequential(
            nn.Linear(n_steps * n_channels, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, d_out), nn.LayerNorm(d_out),
        )
        self.d_out = d_out

    def forward(self, x_future: torch.Tensor) -> torch.Tensor:
        """[B, F, C] -> [B, d_out]"""
        return self.net(x_future.reshape(x_future.shape[0], -1))


def build_modality_encoders(channel_counts: Dict[str, int], d_model: int,
                            cfg: Dict) -> nn.ModuleDict:
    """One encoder per present modality, sized from the config."""
    enc = nn.ModuleDict()
    dropout = float(cfg.get("dropout", 0.0))
    missing_emb = bool(cfg.get("missing_embedding", False))
    for name, n_ch in channel_counts.items():
        if n_ch == 0:
            continue
        if name == "cgm":
            local = dict(hidden=cfg.get("cgm_hidden"),
                         use_differences=bool(cfg.get("cgm_differences", False)),
                         missing_embedding=missing_emb)
            kind = str(cfg.get("cgm_encoder", "default") or "default")
            if kind == "default":
                enc[name] = CGMEncoder(n_ch, d_model, dropout=dropout, **local)
            else:
                # A CGM foundation model (CGM-JEPA / GlucoFM) stands in for the
                # convolutional encoder. Imported lazily so a run that does not
                # ask for one never pays for torch-hub/safetensors imports.
                from .fm_encoders import build_fm_cgm_encoder
                enc[name] = build_fm_cgm_encoder(
                    kind, n_ch, d_model, dict(cfg.get("fm", {}) or {}),
                    dropout, local_kwargs=local)
        else:
            enc[name] = DriverEncoder(n_ch, d_model, hidden=cfg.get("driver_hidden"),
                                      kernel_size=int(cfg.get("driver_kernel", 3)),
                                      dropout=dropout, missing_embedding=missing_emb)
    return enc
