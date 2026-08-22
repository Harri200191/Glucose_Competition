"""Forecast heads (plan stage 5): one shared design for every backbone,
plus an opt-in cross-attention decoder for the future-covariate condition.

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
                 quantiles: Optional[Sequence[float]] = None,
                 d_extra: int = 0) -> None:
        super().__init__()
        if pooling not in ("last", "mean", "last_mean"):
            raise ValueError(f"unknown pooling {pooling!r}")
        self.pooling = pooling
        self.n_horizons = n_horizons
        self.quantiles = list(quantiles) if quantiles else []
        # d_extra is the planned-action embedding, when a run opted into one. It
        # enters here, at the single point every architecture shares, so that the
        # future-covariate comparison is not confounded by *where* the
        # information was injected.
        self.d_extra = d_extra
        in_dim = d_model * (2 if pooling == "last_mean" else 1) + d_extra
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

    def forward(self, seq: torch.Tensor, extra: Optional[torch.Tensor] = None):
        pooled = self.pool(seq)
        if self.d_extra:
            if extra is None:
                raise ValueError("head was built with d_extra but got no extra input")
            pooled = torch.cat([pooled, extra], dim=-1)
        out = self.net(pooled)
        point = out[:, :self.n_horizons]
        if not self.quantiles:
            return point, None
        q = out[:, self.n_horizons:].reshape(-1, self.n_horizons, len(self.quantiles))
        return point, _monotone_quantiles(point, q, len(self.quantiles))


def _monotone_quantiles(point: torch.Tensor, raw: torch.Tensor,
                        n_q: int) -> torch.Tensor:
    """Monotone quantiles by construction: cumulative softplus offsets around
    the point forecast, so the 90th can never fall below the 10th."""
    offsets = torch.cumsum(torch.nn.functional.softplus(raw), dim=-1)
    centre = offsets[..., n_q // 2: n_q // 2 + 1]
    return point.unsqueeze(-1) + (offsets - centre)


class HorizonCrossHead(nn.Module):
    """Per-horizon query tokens cross-attending over history and planned doses.

    The pooled head compresses 24 h of context into one vector and appends the
    future-insulin block as one flat embedding — the single most informative
    signal in the competition condition enters the network after all sequence
    processing is over. Here each horizon owns a query token, initialised from
    the anchor state, that attends over (a) every step of the backbone's output
    sequence and (b) one token per future 5-minute slot of the planned doses.
    A 60-minute query can weigh the bolus landing at +25 min differently from
    the 120-minute one — position-resolved credit assignment the flat embedding
    cannot express.

    Output contract matches ``MultiHorizonHead``: ``[B, H]`` point (zero-init,
    so training starts at persistence) plus optional monotone quantiles.
    """

    def __init__(self, d_model: int, n_horizons: int,
                 future_steps: int = 0, future_channels: int = 0,
                 n_heads: int = 4, n_layers: int = 2, dropout: float = 0.0,
                 quantiles: Optional[Sequence[float]] = None) -> None:
        super().__init__()
        self.n_horizons = n_horizons
        self.quantiles = list(quantiles) if quantiles else []
        self.future_steps = future_steps
        self.queries = nn.Parameter(torch.randn(n_horizons, d_model) * 0.02)
        self.anchor_proj = nn.Linear(d_model, d_model)
        if future_steps and future_channels:
            self.future_proj = nn.Linear(future_channels, d_model)
            self.future_pos = nn.Parameter(
                torch.randn(future_steps, d_model) * 0.02)
        else:
            self.future_proj = None
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(nn.ModuleDict({
                "ln_q": nn.LayerNorm(d_model),
                "ln_m": nn.LayerNorm(d_model),
                "attn": nn.MultiheadAttention(d_model, n_heads,
                                              dropout=dropout, batch_first=True),
                "ln_f": nn.LayerNorm(d_model),
                "ffn": nn.Sequential(
                    nn.Linear(d_model, 2 * d_model), nn.GELU(),
                    nn.Dropout(dropout), nn.Linear(2 * d_model, d_model)),
            }))
        self.out_norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, 1 + len(self.quantiles))
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, seq: torch.Tensor, future_block: Optional[torch.Tensor] = None):
        B = seq.shape[0]
        memory = seq
        if self.future_proj is not None:
            if future_block is None:
                raise ValueError("head was built for future covariates but got none")
            fut = self.future_proj(future_block) + self.future_pos.unsqueeze(0)
            memory = torch.cat([memory, fut], dim=1)
        q = self.queries.unsqueeze(0).expand(B, -1, -1) \
            + self.anchor_proj(seq[:, -1]).unsqueeze(1)
        for blk in self.blocks:
            attn_out, _ = blk["attn"](blk["ln_q"](q), blk["ln_m"](memory),
                                      blk["ln_m"](memory), need_weights=False)
            q = q + attn_out
            q = q + blk["ffn"](blk["ln_f"](q))
        out = self.out(self.out_norm(q))                     # [B, H, 1+Q]
        point = out[..., 0]
        if not self.quantiles:
            return point, None
        return point, _monotone_quantiles(point, out[..., 1:],
                                          len(self.quantiles))
