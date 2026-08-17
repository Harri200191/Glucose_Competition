"""TiDE -- Time-series Dense Encoder (Das et al., 2023).

Tests: *can a simpler nonlinear dense encoder-decoder match the sequence models?*
Treated throughout as the efficiency/complexity control: if TiDE matches Mamba-2
at a tenth of the cost, that is a result about the task, not a disappointment.

Native structure, kept: per-step covariate projection -> flatten -> dense encoder
of residual blocks -> dense decoder -> temporal decoder per horizon -> plus a
**global linear residual** straight from the lookback CGM to the forecasts. That
residual is why TiDE is a strong baseline: the linear part gets persistence and
trend for free, and the dense part only has to model the correction.

Deviation, same reason as TFT: the paper's future covariates do not exist here, so
the temporal decoder is conditioned on a learned per-horizon embedding instead.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ForecastModel, ForecastOutput
from .encoders import StaticEncoder
from .frontend import _build_future_encoder


class ResidualBlock(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int, dropout: float = 0.1,
                 layer_norm: bool = True) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_out),
            nn.Dropout(dropout))
        self.skip = nn.Linear(d_in, d_out, bias=False)
        self.norm = nn.LayerNorm(d_out) if layer_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.net(x) + self.skip(x))


class TiDEForecaster(ForecastModel):
    def __init__(self, shapes: Dict, model_cfg: Dict) -> None:
        super().__init__(n_horizons=int(shapes["n_horizons"]))
        cfg = dict(model_cfg.get("backbone", {}) or {})
        d_hidden = int(cfg.get("d_hidden", 256))
        d_decoder = int(cfg.get("d_decoder", 32))
        n_enc = int(cfg.get("n_encoder_layers", 2))
        n_dec = int(cfg.get("n_decoder_layers", 2))
        r = int(cfg.get("temporal_width", 4))
        dropout = float(cfg.get("dropout", 0.1))
        self.context = int(shapes["context"])
        self.p = d_decoder

        self.modalities: List[str] = [m for m, n in shapes["channel_counts"].items() if n]
        n_cov = sum(shapes["channel_counts"][m] for m in self.modalities) \
            + int(shapes["n_time_features"])
        self.n_cov = n_cov

        d_static = int(model_cfg.get("d_static", 32))
        self.static_encoder = StaticEncoder(
            int(shapes["n_static_numeric"]), list(shapes["static_cardinalities"]),
            d_static, emb_dim=int(model_cfg.get("static_emb_dim", 8)), dropout=dropout)

        # TiDE natively takes future covariates into the dense encoder, so when a
        # run opts in they go there rather than through the shared head -- the
        # one architecture where the native placement and the shared placement
        # coincide.
        self.future_encoder = _build_future_encoder(shapes, model_cfg)
        d_future = self.future_encoder.d_out if self.future_encoder is not None else 0

        self.feature_proj = ResidualBlock(n_cov, d_hidden, r, dropout)
        enc_in = self.context * r + d_static + self.context + d_future  # + lookback CGM
        layers = [ResidualBlock(enc_in, d_hidden, d_hidden, dropout)]
        layers += [ResidualBlock(d_hidden, d_hidden, d_hidden, dropout)
                   for _ in range(n_enc - 1)]
        self.encoder = nn.Sequential(*layers)

        dec_layers = [ResidualBlock(d_hidden, d_hidden, d_hidden, dropout)
                      for _ in range(n_dec - 1)]
        dec_layers.append(ResidualBlock(d_hidden, d_hidden, self.n_horizons * d_decoder,
                                        dropout))
        self.decoder = nn.Sequential(*dec_layers)

        self.horizon_embedding = nn.Parameter(torch.zeros(self.n_horizons, r))
        nn.init.trunc_normal_(self.horizon_embedding, std=0.02)
        self.temporal_decoder = ResidualBlock(d_decoder + r, d_hidden, 1, dropout,
                                              layer_norm=False)
        self.global_residual = nn.Linear(self.context, self.n_horizons)
        nn.init.zeros_(self.global_residual.weight)
        nn.init.zeros_(self.global_residual.bias)

    def forward(self, batch: Dict[str, torch.Tensor]) -> ForecastOutput:
        parts = [batch[f"x_{m}"] for m in self.modalities]
        if "time" in batch and batch["time"].numel():
            parts.append(batch["time"])
        cov = torch.cat(parts, dim=-1)                       # [B, L, n_cov]
        b = cov.shape[0]

        lookback = batch["x_cgm"][..., 0]                    # normalised CGM series
        proj = self.feature_proj(cov).reshape(b, -1)         # [B, L*r]
        s = self.static_encoder(batch["static_num"], batch["static_cat"])
        parts_enc = [lookback, proj, s]
        if self.future_encoder is not None:
            parts_enc.append(self.future_encoder(batch["x_future"]))
        e = self.encoder(torch.cat(parts_enc, dim=-1))
        g = self.decoder(e).reshape(b, self.n_horizons, self.p)

        h_emb = self.horizon_embedding.unsqueeze(0).expand(b, -1, -1)
        point = self.temporal_decoder(torch.cat([g, h_emb], dim=-1)).squeeze(-1)
        point = point + self.global_residual(lookback)
        return ForecastOutput(pred_norm=point)

    def describe(self) -> Dict[str, object]:
        d = super().describe()
        d.update({"backbone": "tide", "fusion": "native_dense", "context": self.context,
                  "future_covariates": self.future_encoder is not None})
        return d
