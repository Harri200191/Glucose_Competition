"""Temporal Fusion Transformer (Lim et al., 2021), in native form.

Tests: *does native variable selection / gating / static enrichment beat a
purpose-built front end?*

The plan is explicit that TFT must **not** be given the common fusion module:
variable selection and gating are the architecture's own answer to the question
this project is asking, so wrapping it in another fusion module would test
nothing. It therefore consumes the raw normalised channels as separate variables
and builds its own representation:

    per-variable linear -> variable selection network (softmax over variables,
    conditioned on statics) -> LSTM encoder -> static enrichment GRN ->
    interpretable multi-head attention (causally masked) -> position-wise GRN ->
    multi-horizon output.

Documented deviation: the original model has a decoder over *known future*
covariates. In glucose forecasting nothing about the future is known at inference
time -- future insulin and meals are exactly what is unknown -- so the decoder is
dropped and the horizons are read off the final encoder position. This is stated
here and in the run manifest because it changes what the comparison means.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ForecastModel, ForecastOutput
from .encoders import StaticEncoder
from .frontend import _build_future_encoder
from .heads import MultiHorizonHead


class GatedLinearUnit(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.fc = nn.Linear(d, 2 * d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.fc(x).chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class GRN(nn.Module):
    """Gated residual network: the TFT's universal building block."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int, dropout: float = 0.1,
                 d_context: int = 0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.ctx = nn.Linear(d_context, d_hidden, bias=False) if d_context else None
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.drop = nn.Dropout(dropout)
        self.glu = GatedLinearUnit(d_out)
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else None
        self.norm = nn.LayerNorm(d_out)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        h = self.fc1(x)
        if self.ctx is not None and context is not None:
            # A static context arrives as [B, d] and is broadcast over time: the
            # same patient-level vector conditions every step of the window.
            if context.dim() == x.dim() - 1:
                context = context.unsqueeze(1)
            h = h + self.ctx(context)
        h = self.fc2(self.drop(F.elu(h)))
        res = x if self.skip is None else self.skip(x)
        return self.norm(res + self.glu(h))


class VariableSelection(nn.Module):
    """Softmax weights over variables, conditioned on the static context."""

    def __init__(self, n_vars: int, d_model: int, dropout: float = 0.1,
                 d_context: int = 0) -> None:
        super().__init__()
        self.n_vars = n_vars
        self.transforms = nn.ModuleList(
            [GRN(1, d_model, d_model, dropout) for _ in range(n_vars)])
        self.weight_grn = GRN(n_vars, d_model, n_vars, dropout, d_context=d_context)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None):
        """x: [B, T, V] -> ([B, T, d], weights [B, T, V])."""
        transformed = torch.stack(
            [self.transforms[v](x[..., v:v + 1]) for v in range(self.n_vars)], dim=-2)
        ctx = context.unsqueeze(1).expand(-1, x.shape[1], -1) if context is not None else None
        weights = torch.softmax(self.weight_grn(x, ctx), dim=-1).unsqueeze(-1)
        return (transformed * weights).sum(dim=-2), weights.squeeze(-1)


class InterpretableMultiHeadAttention(nn.Module):
    """Shared value projection across heads, as in the TFT paper."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, self.d_head)      # shared across heads
        self.out = nn.Linear(self.d_head, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        b, t, _ = x.shape
        q = self.q(x).reshape(b, t, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k(x).reshape(b, t, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v(x).unsqueeze(1)                     # [B, 1, T, d_head]
        scores = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)
        causal = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), 1)
        scores = scores.masked_fill(causal, float("-inf"))
        attn = self.drop(torch.softmax(scores, dim=-1))
        ctx = (attn @ v.expand(-1, self.n_heads, -1, -1)).mean(dim=1)
        return self.out(ctx), attn.mean(dim=1)


class TFTForecaster(ForecastModel):
    def __init__(self, shapes: Dict, model_cfg: Dict) -> None:
        super().__init__(n_horizons=int(shapes["n_horizons"]))
        d_model = int(model_cfg.get("d_model", 128))
        cfg = dict(model_cfg.get("backbone", {}) or {})
        dropout = float(cfg.get("dropout", 0.1))
        n_heads = int(cfg.get("n_heads", 4))
        self.modalities: List[str] = [m for m, n in shapes["channel_counts"].items() if n]
        n_vars = sum(shapes["channel_counts"][m] for m in self.modalities) \
            + int(shapes["n_time_features"])
        self.n_vars = n_vars

        d_static = int(model_cfg.get("d_static", 32))
        self.static_encoder = StaticEncoder(
            int(shapes["n_static_numeric"]), list(shapes["static_cardinalities"]),
            d_static, emb_dim=int(model_cfg.get("static_emb_dim", 8)), dropout=dropout)
        # The four static context vectors of the paper: variable selection,
        # enrichment, and the LSTM's initial cell and hidden state.
        self.ctx_selection = GRN(d_static, d_model, d_model, dropout)
        self.ctx_enrichment = GRN(d_static, d_model, d_model, dropout)
        self.ctx_hidden = GRN(d_static, d_model, d_model, dropout)
        self.ctx_cell = GRN(d_static, d_model, d_model, dropout)

        self.vsn = VariableSelection(n_vars, d_model, dropout, d_context=d_model)
        self.lstm = nn.LSTM(d_model, d_model, num_layers=1, batch_first=True)
        self.post_lstm_gate = GatedLinearUnit(d_model)
        self.post_lstm_norm = nn.LayerNorm(d_model)
        self.enrichment = GRN(d_model, d_model, d_model, dropout, d_context=d_model)
        self.attention = InterpretableMultiHeadAttention(d_model, n_heads, dropout)
        self.post_attn_norm = nn.LayerNorm(d_model)
        self.position_wise = GRN(d_model, d_model, d_model, dropout)
        self.future_encoder = _build_future_encoder(shapes, model_cfg)
        self.head = MultiHorizonHead(
            d_model, self.n_horizons,
            pooling=str(model_cfg.get("head", {}).get("pooling", "last")),
            dropout=dropout,
            d_extra=self.future_encoder.d_out if self.future_encoder is not None else 0)
        self.last_variable_weights: Optional[torch.Tensor] = None

    def forward(self, batch: Dict[str, torch.Tensor]) -> ForecastOutput:
        parts = [batch[f"x_{m}"] for m in self.modalities]
        if "time" in batch and batch["time"].numel():
            parts.append(batch["time"])
        x = torch.cat(parts, dim=-1)

        s = self.static_encoder(batch["static_num"], batch["static_cat"])
        c_sel, c_enr = self.ctx_selection(s), self.ctx_enrichment(s)
        h0 = self.ctx_hidden(s).unsqueeze(0).contiguous()
        c0 = self.ctx_cell(s).unsqueeze(0).contiguous()

        z, weights = self.vsn(x, c_sel)
        self.last_variable_weights = weights.detach().mean(dim=1)

        lstm_out, _ = self.lstm(z, (h0, c0))
        h = self.post_lstm_norm(z + self.post_lstm_gate(lstm_out))
        h = self.enrichment(h, c_enr)
        attn_out, _ = self.attention(h)
        h = self.post_attn_norm(h + attn_out)
        h = self.position_wise(h)
        future = (self.future_encoder(batch["x_future"])
                  if self.future_encoder is not None else None)
        point, quantiles = self.head(h, future)
        return ForecastOutput(pred_norm=point, quantiles=quantiles,
                              aux={"variable_weights": self.last_variable_weights})

    def describe(self) -> Dict[str, object]:
        d = super().describe()
        d.update({"backbone": "tft", "fusion": "native_variable_selection",
                  "n_variables": self.n_vars,
                  "future_covariates": self.future_encoder is not None})
        return d
