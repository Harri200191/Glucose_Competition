"""Near-canonical PatchTST (Nie et al., 2023), as a *native* model.

Why this exists separately from ``models/patchtst.py``
-----------------------------------------------------
The backbone in ``patchtst.py`` patches the **fused 128-wide latent** produced by
the shared front end. That is what the controlled comparison asked for, but it
discards the paper's central claim -- *channel independence* -- because the
channels have already been mixed before PatchTST ever runs. It also uses
non-overlapping patches and the project's shared last+mean readout.

This module implements the paper's own arrangement, so that "PatchTST is worse"
can be tested against the real method rather than against an adaptation:

1. **Channel independence.** Each input channel is patched, embedded and passed
   through the transformer *on its own*, with weights shared across channels.
   Channels never see each other until the head. Implemented by folding the
   channel axis into the batch: ``[B, L, C] -> [B*C, L, 1]``.
2. **Overlapping patches.** ``patch_len 16 / stride 8`` by default, giving 50%
   overlap, as opposed to the non-overlapping ``16/16`` of the fused variant.
3. **Flatten-all-patches head.** Every patch token is flattened into the head,
   rather than the shared ``last + mean`` pooling.
4. **Full attention over the lookback.** The paper does not mask attention inside
   the lookback window, and it does not need to: every position in the window is
   at or before the anchor, so attending across them cannot see the future. This
   is *less* restrictive than the fused variant's causal mask and is the correct
   native behaviour. Causality is preserved by the window boundary, not by a mask.

Statics and calendar features have no native place in PatchTST, so they enter at
the head -- the same place the optional future-action embedding enters for every
other architecture, which keeps that comparison unconfounded.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from .base import ForecastModel, ForecastOutput
from .encoders import StaticEncoder
from .frontend import _build_future_encoder


class PatchTSTNativeForecaster(ForecastModel):
    def __init__(self, shapes: Dict, model_cfg: Dict) -> None:
        super().__init__(n_horizons=int(shapes["n_horizons"]))
        cfg = dict(model_cfg.get("backbone", {}) or {})
        d_model = int(model_cfg.get("d_model", 128))
        self.patch_len = int(cfg.get("patch_len", 16))
        self.stride = int(cfg.get("stride", 8))
        n_layers = int(cfg.get("n_layers", 3))
        n_heads = int(cfg.get("n_heads", 8))
        dropout = float(cfg.get("dropout", 0.1))
        d_head = int(cfg.get("d_head", 32))
        self.instance_norm = bool(cfg.get("instance_norm", False))
        self.context = int(shapes["context"])

        self.modalities: List[str] = [m for m, n in shapes["channel_counts"].items() if n]
        self.n_channels = sum(shapes["channel_counts"][m] for m in self.modalities)
        self.n_patches = self._n_patches(self.context)
        self.d_model = d_model

        # --- shared across every channel: this IS channel independence ---------
        self.patch_embed = nn.Linear(self.patch_len, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.embed_drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        # --- native flatten head, applied per channel with shared weights ------
        self.flatten_head = nn.Sequential(
            nn.Flatten(start_dim=-2),                       # [B*C, P*d_model]
            nn.Linear(self.n_patches * d_model, d_head),
            nn.GELU(), nn.Dropout(dropout),
        )

        d_static = int(model_cfg.get("d_static", 32)) if (
            shapes["n_static_numeric"] or shapes["static_cardinalities"]) else 0
        self.static_encoder = StaticEncoder(
            int(shapes["n_static_numeric"]), list(shapes["static_cardinalities"]),
            d_static, emb_dim=int(model_cfg.get("static_emb_dim", 8)),
            dropout=dropout) if d_static else None
        self.n_time = int(shapes["n_time_features"])

        self.future_encoder = _build_future_encoder(shapes, model_cfg)
        d_future = self.future_encoder.d_out if self.future_encoder is not None else 0

        mix_in = self.n_channels * d_head + d_static + self.n_time + d_future
        hidden = int(cfg.get("d_mix", max(d_model, 128)))
        self.mix = nn.Sequential(
            nn.LayerNorm(mix_in),
            nn.Linear(mix_in, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, self.n_horizons),
        )
        # Same convention as the shared head: zero-init the last layer so an
        # untrained model predicts a delta of zero, i.e. persistence.
        nn.init.zeros_(self.mix[-1].weight)
        nn.init.zeros_(self.mix[-1].bias)

    def _n_patches(self, length: int) -> int:
        return max(1, 1 + (length - self.patch_len) // self.stride)

    def forward(self, batch: Dict[str, torch.Tensor]) -> ForecastOutput:
        x = torch.cat([batch[f"x_{m}"] for m in self.modalities], dim=-1)   # [B,L,C]
        b, l, c = x.shape

        # Trailing-aligned: the final patch must end exactly at the anchor, so any
        # remainder is dropped from the *start* of the window, never the end.
        p = self._n_patches(l)
        used = self.patch_len + (p - 1) * self.stride
        if used < l:
            x = x[:, l - used:]

        x = x.permute(0, 2, 1).reshape(b * c, -1)                     # [B*C, L']
        if self.instance_norm:
            mu = x.mean(dim=-1, keepdim=True)
            sd = x.std(dim=-1, keepdim=True).clamp(min=1e-5)
            x = (x - mu) / sd
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        h = self.embed_drop(self.patch_embed(patches) + self.pos[:, :p])
        h = self.norm(self.encoder(h))                                # [B*C, P, d]

        z = self.flatten_head(h).reshape(b, -1)                       # [B, C*d_head]
        parts = [z]
        if self.static_encoder is not None:
            parts.append(self.static_encoder(batch["static_num"], batch["static_cat"]))
        if self.n_time and "time" in batch and batch["time"].numel():
            parts.append(batch["time"][:, -1])          # calendar state at the anchor
        if self.future_encoder is not None:
            parts.append(self.future_encoder(batch["x_future"]))
        return ForecastOutput(pred_norm=self.mix(torch.cat(parts, dim=-1)))

    def describe(self) -> Dict[str, object]:
        d = super().describe()
        d.update({"backbone": "patchtst_native", "fusion": "native_channel_independent",
                  "context": self.context, "patch_len": self.patch_len,
                  "stride": self.stride, "n_patches": self.n_patches,
                  "n_channels": self.n_channels,
                  "future_covariates": self.future_encoder is not None})
        return d
