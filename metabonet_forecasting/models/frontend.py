"""The common multimodal front end (plan section 4).

    input -> [stage 1] modality encoders
          -> [stage 2] event/exposure features (built in prepare.py, selected here)
          -> [stage 3] CGM-centred gated / FiLM fusion
          -> [stage 4] temporal backbone (TCN | PatchTST | Mamba-2 | LSTM)
          -> [stage 5] shared multi-horizon head

TCN, PatchTST and Mamba-2 receive *this same object* with only ``backbone.name``
changed, which is the entire point: a difference between them is a difference
between backbones, not between input pipelines.

``fusion.name: raw_concat`` bypasses stages 1 and 3 and feeds normalised channels
straight to a single linear projection. That is the required baseline of plan 5.1
and the starting rung of the stage-D ablation ladder.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from ..common.registry import BACKBONES
from .base import ForecastModel, ForecastOutput
from .encoders import (FeatureEncoder, FutureActionEncoder, StaticEncoder, TimeEncoder,
                       build_modality_encoders)
from .fusion import build_fusion
from .physio import PhysioFeatures
from .heads import HorizonCrossHead, MultiHorizonHead, build_law


def _build_future_encoder(shapes: Dict, model_cfg: Dict,
                          encoder_cfg: Optional[Dict] = None):
    """The planned-action encoder, or None when the run is strictly causal.

    Shared by every architecture so that "with future covariates" means the same
    thing in each of them.
    """
    steps = int(shapes.get("future_steps", 0) or 0)
    channels = int(shapes.get("future_channels", 0) or 0)
    if steps <= 0 or channels <= 0:
        return None
    encoder_cfg = encoder_cfg or {}
    return FutureActionEncoder(
        steps, channels, int(model_cfg.get("d_future", 32)),
        hidden=model_cfg.get("future_hidden"),
        dropout=float(encoder_cfg.get("dropout", 0.0)))


class FrontEndForecaster(ForecastModel):
    def __init__(self, shapes: Dict, model_cfg: Dict) -> None:
        super().__init__(n_horizons=int(shapes["n_horizons"]))
        d_model = int(model_cfg.get("d_model", 128))
        fusion_cfg = dict(model_cfg.get("fusion", {}) or {})
        encoder_cfg = dict(model_cfg.get("encoder", {}) or {})
        backbone_cfg = dict(model_cfg.get("backbone", {}) or {})
        head_cfg = dict(model_cfg.get("head", {}) or {})

        self.fusion_name = str(fusion_cfg.get("name", "film"))
        self.d_model = d_model
        self.channel_counts: Dict[str, int] = dict(shapes["channel_counts"])
        # Physiological channels derived from CGM + insulin, no meal log needed.
        # Appended to the CGM block before its encoder, so the encoder just sees
        # a wider input and nothing downstream changes.
        physio_cfg = dict(model_cfg.get("physio", {}) or {})
        self.physio = (PhysioFeatures(
            float(physio_cfg.get("ema_halflife_steps", 6.0)))
            if physio_cfg.get("enabled") else None)
        if self.physio is not None:
            self.channel_counts["cgm"] = \
                int(self.channel_counts.get("cgm", 0)) + self.physio.n_out
        self.modalities: List[str] = [m for m in self.channel_counts if
                                      self.channel_counts[m] > 0]
        self.context = int(shapes["context"])

        # --- statics and calendar: identical in every fusion mode -------------
        d_static = int(model_cfg.get("d_static", 32)) if shapes["n_static_numeric"] \
            or shapes["static_cardinalities"] else 0
        self.static_encoder = StaticEncoder(
            int(shapes["n_static_numeric"]), list(shapes["static_cardinalities"]),
            d_static, emb_dim=int(model_cfg.get("static_emb_dim", 8)),
            dropout=float(encoder_cfg.get("dropout", 0.0))) if d_static else None
        d_time = int(model_cfg.get("d_time", 16)) if shapes["n_time_features"] else 0
        self.time_encoder = TimeEncoder(int(shapes["n_time_features"]), d_time) \
            if d_time else None

        # Engineered per-anchor features ride the STATIC conditioning path: no
        # fusion inspects what `static` means, only how wide it is, so widening it
        # is transparent to raw_concat / concat / gated / film alike and to both
        # heads. Under FiLM it also puts the features where they can modulate the
        # CGM embedding at every step -- the strongest available coupling, and so
        # the fairest test of whether they carry anything the window does not.
        n_feat = int(shapes.get("n_extra_features", 0) or 0)
        d_feat = int(model_cfg.get("d_feat", 64)) if n_feat else 0
        self.feature_encoder = FeatureEncoder(
            n_feat, d_feat, hidden=model_cfg.get("feat_hidden"),
            dropout=float(encoder_cfg.get("dropout", 0.0))) if d_feat else None
        self.n_extra_features = n_feat
        d_static_total = d_static + d_feat

        # --- stages 1 and 3 ---------------------------------------------------
        if self.fusion_name == "raw_concat":
            n_raw = sum(self.channel_counts[m] for m in self.modalities)
            self.encoders = None
            self.fusion = None
            self.input_proj = nn.Sequential(
                nn.Linear(n_raw + d_static_total + d_time, d_model),
                nn.LayerNorm(d_model),
            )
        else:
            self.encoders = build_modality_encoders(
                {m: self.channel_counts[m] for m in self.modalities}, d_model, encoder_cfg)
            self.fusion = build_fusion(self.fusion_name, d_model, self.modalities,
                                       d_static_total, d_time, fusion_cfg)
            self.input_proj = None

        # --- stages 4 and 5 ---------------------------------------------------
        backbone_name = str(backbone_cfg.pop("name", "tcn"))
        backbone_cfg.setdefault("context", self.context)
        self.backbone = BACKBONES.build(backbone_name, d_model=d_model, **backbone_cfg)
        self.backbone_name = backbone_name

        self.head_name = str(head_cfg.get("name", "pooled"))
        # The predictive LAW is orthogonal to the head's topology, so it is one
        # shared object handed to whichever head is built rather than a second
        # dimension of head classes.
        law = build_law(head_cfg, self.n_horizons)
        self.law_name = None if law is None else f"student_t_k{law.k}"
        if self.head_name == "horizon_cross":
            # The cross-attention decoder consumes the raw future block itself
            # (one token per future slot), so no flat future encoder is built.
            self.future_encoder = None
            self.head = HorizonCrossHead(
                d_model, self.n_horizons,
                future_steps=int(shapes.get("future_steps", 0) or 0),
                future_channels=int(shapes.get("future_channels", 0) or 0),
                n_heads=int(head_cfg.get("n_heads", 4)),
                n_layers=int(head_cfg.get("n_layers", 2)),
                dropout=float(head_cfg.get("dropout", 0.0)),
                quantiles=head_cfg.get("quantiles"),
                # `--set model.head.chain=none` is coerced to Python None by
                # the config parser, which is the same thing the user meant.
                chain=str(head_cfg.get("chain") or "none"), law=law)
        else:
            # Planned future actions, only if this run asked for them.
            self.future_encoder = _build_future_encoder(shapes, model_cfg, encoder_cfg)
            d_extra = self.future_encoder.d_out if self.future_encoder is not None else 0
            self.head = MultiHorizonHead(
                d_model, self.n_horizons, hidden=head_cfg.get("hidden"),
                pooling=str(head_cfg.get("pooling", "last_mean")),
                dropout=float(head_cfg.get("dropout", 0.0)),
                quantiles=head_cfg.get("quantiles"), d_extra=d_extra, law=law)

    # ---------------------------------------------------------------- forward
    def _cgm_in(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """The CGM block, widened with physiological channels when enabled."""
        x = batch["x_cgm"]
        if self.physio is None:
            return x
        return torch.cat([x, self.physio(x, batch.get("x_insulin"))], dim=-1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> ForecastOutput:
        static = None
        if self.static_encoder is not None:
            static = self.static_encoder(batch["static_num"], batch["static_cat"])
        if self.feature_encoder is not None:
            feat = self.feature_encoder(batch["x_feat"])
            static = feat if static is None else torch.cat([static, feat], dim=-1)
        time = None
        if self.time_encoder is not None:
            time = self.time_encoder(batch.get("time"))

        if self.fusion is None:
            parts = [self._cgm_in(batch) if m == "cgm" else batch[f"x_{m}"]
                     for m in self.modalities]
            length = parts[0].shape[1]
            if static is not None:
                parts.append(static.unsqueeze(1).expand(-1, length, -1))
            if time is not None:
                parts.append(time)
            z = self.input_proj(torch.cat(parts, dim=-1))
        else:
            # A CGM foundation-model encoder rebuilds calendar features itself
            # (its pretrained projection expects its own layout), so it is handed
            # the raw time block; every other encoder keeps the two-argument
            # signature untouched.
            encoded = {}
            for m in self.modalities:
                enc = self.encoders[m]
                xm = self._cgm_in(batch) if m == "cgm" else batch[f"x_{m}"]
                if getattr(enc, "wants_time", False):
                    encoded[m] = enc(xm, batch.get(f"m_{m}"), batch.get("time"))
                else:
                    encoded[m] = enc(xm, batch.get(f"m_{m}"))
            z = self.fusion(encoded, static=static, time=time)

        seq = self.backbone(z)
        if self.head_name == "horizon_cross":
            point, quantiles, aux = self.head(seq, batch.get("x_future"))
        else:
            future = (self.future_encoder(batch["x_future"])
                      if self.future_encoder is not None else None)
            point, quantiles, aux = self.head(seq, future)
        return ForecastOutput(pred_norm=point, quantiles=quantiles, aux=aux)

    # ------------------------------------------------------------------ meta
    def receptive_field(self) -> Optional[int]:
        return self.backbone.receptive_field()

    def describe(self) -> Dict[str, object]:
        d = super().describe()
        d.update({"physio": self.physio.describe() if self.physio else None,
                  "backbone": self.backbone_name, "fusion": self.fusion_name,
                  "head": self.head_name, "law": self.law_name,
                  "n_extra_features": self.n_extra_features,
                  "d_model": self.d_model, "modalities": self.modalities,
                  "context": self.context,
                  "future_covariates": self.future_encoder is not None
                  or (self.head_name == "horizon_cross"
                      and getattr(self.head, "future_proj", None) is not None)})
        if self.encoders is not None:
            for m, enc in self.encoders.items():
                if hasattr(enc, "fm_summary"):
                    d[f"fm_{m}"] = enc.fm_summary()
        return d
