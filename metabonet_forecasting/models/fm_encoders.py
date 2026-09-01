"""CGM foundation models as drop-in replacements for the CGM modality encoder.

WHERE THEY PLUG IN, AND WHY THERE

Both CGM-JEPA and GlucoFM are *encoders*. Neither has a forecasting head: they
were evaluated by freezing the encoder and fitting logistic regression on the
embeddings for AUROC/PR-AUC on metabolic-phenotype labels. Nothing in either
paper predicts a glucose value 120 minutes out. So "add them to the model pool"
cannot mean "run them as forecasters" -- there is nothing to run.

What they can do is supply the *representation*. This repository already has an
exactly-shaped slot for that: every modality encoder is

    (x [B, T, C], mask [B, T, 1]) -> [B, T, d_model]

and the CGM one is currently two causal convolutions. Swapping the FM in there
keeps the entire rest of the pipeline -- FiLM fusion, the chosen backbone,
Tayyib's horizon_cross head, statics, future insulin, the loss, the decision
layer -- bit-for-bit identical, so a difference in score is a difference in CGM
representation and nothing else. It also means each FM composes with every
backbone we already have, rather than replacing one.

THE INPUT GRID LINES UP EXACTLY. Both models are specified on a 24 h, 5-minute
grid: 288 steps, which is this repo's ``data.context`` already, and for CGM-JEPA
288 = 24 patches x 12, its native patch layout. No resampling, no interpolation.

MODES

    mode=hybrid   (default) keep the convolutional CGM encoder and ADD the FM
                  through a ZERO-INITIALISED projection. At step 0 the model is
                  therefore bit-for-bit the baseline, and the projection's norm
                  is a direct read-out of how much the FM was actually worth.
                  (A zero-initialised *scalar gate* would look equivalent and is
                  not: with the gate at exactly 0 the projection behind it
                  receives exactly zero gradient, so the branch never starts
                  learning. Zeroing the projection instead starts at the same
                  place while keeping its gradient alive from the first step.)
                  A patch embedding is a 60-minute summary; the 30-minute
                  forecast needs the last few readings at full resolution, and
                  the local convolutions are what supply them. Trust this mode.
    mode=replace  the FM is the only CGM pathway. Cleaner as an experiment,
                  weaker as a forecaster, and the honest way to ask "how much
                  does the pretrained representation carry on its own".

NORMALISATION. ``input_norm=instance`` z-scores each window over its OBSERVED
steps only. This matters more than it looks: our channel is already scaled by
the corpus median/IQR, and mask-aware instance normalisation is invariant to any
affine rescaling, so the FM receives *exactly* what it would from raw mg/dL. It
also absorbs the domain shift -- both models were pretrained on largely
non-diabetic cohorts (mean ~100 mg/dL), and feeding T1D values through a
healthy-cohort scale would put every window far outside the pretraining support.
Absolute glucose level is not lost: it still reaches the model through the
convolutional path and the fusion, and the head predicts a delta from the last
reading anyway.

CAUSALITY. Both encoders attend bidirectionally across the 288-step window.
That is not a leak and does not weaken the causal protocol: every step in the
window is at or before the forecast anchor. It is a genuine deviation from the
strictly-causal backbones, and it is recorded here so nobody has to rediscover
it from the attention pattern.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn

from .encoders import CGMEncoder

_EPS = 1e-5


def mask_aware_instance_norm(g: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    """Z-score each [B, T] window over its observed steps only.

    Invariant to any affine rescaling of ``g``, which is why the FM sees exactly
    what it would have seen from raw mg/dL even though our channel arrives
    median/IQR-scaled. Shared with the GlucoFM pretrainer so the SSL objective
    and the fine-tuned encoder normalise identically.
    """
    n = obs.sum(dim=-1, keepdim=True).clamp_min(1.0)
    mu = (g * obs).sum(dim=-1, keepdim=True) / n
    var = (((g - mu) ** 2) * obs).sum(dim=-1, keepdim=True) / n
    return (g - mu) / torch.sqrt(var + _EPS)


def informer_time_features(time: torch.Tensor) -> torch.Tensor:
    """Our 4-d sin/cos calendar -> the 5-d vector CGM-JEPA's projection expects.

    Their config says ``time_inp_dim=5``, which is the Informer ``freq='t'`` map
    ``[month, day, weekday, hour, minute//15]``. We can recover weekday, hour and
    minute exactly by inverting the sin/cos pair with atan2; month and
    day-of-month are not in our store and are emitted as zeros.

    That is less lossy than it sounds. Their own preprocessing assigns SYNTHETIC
    dates to the pretraining corpus (``base_date = 2000-01-{day}``), so the
    month/day inputs the projection was fitted against were fabricated too.
    Which is exactly why ``use_time`` defaults to False: time-of-day already
    enters this pipeline through the shared TimeEncoder in fusion, so switching
    it on here adds nothing but a second, noisier copy.
    """
    two_pi = 2.0 * math.pi
    tod = torch.atan2(time[..., 0], time[..., 1]) / two_pi % 1.0     # [0,1) of a day
    dow = torch.atan2(time[..., 2], time[..., 3]) / two_pi * 7.0 % 7.0
    minutes = tod * 1440.0
    hour = torch.floor(minutes / 60.0)
    minute15 = torch.floor((minutes % 60.0) / 15.0)
    zero = torch.zeros_like(tod)
    return torch.stack([zero, zero, torch.floor(dow), hour, minute15], dim=-1)


class _FMCGMEncoder(nn.Module):
    """Shared plumbing: normalise, patchify, run the FM, upsample, mix."""

    wants_time = True          # frontend.py passes batch["time"] to these

    def __init__(self, n_channels: int, d_out: int, *, d_fm: int, patch: int,
                 mode: str = "hybrid", freeze: bool = True, use_time: bool = False,
                 input_norm: str = "instance", missing_fill: float = 0.0,
                 dropout: float = 0.0, local_kwargs: Optional[Dict] = None) -> None:
        super().__init__()
        if mode not in ("hybrid", "replace"):
            raise ValueError(f"mode must be hybrid|replace, got {mode!r}")
        if input_norm not in ("instance", "passthrough"):
            raise ValueError(f"input_norm must be instance|passthrough, got {input_norm!r}")
        self.mode, self.patch, self.use_time = mode, int(patch), bool(use_time)
        self.input_norm, self.missing_fill = input_norm, float(missing_fill)
        self.frozen = bool(freeze)

        self.proj = nn.Linear(d_fm, d_out)
        if mode == "hybrid":
            lk = dict(local_kwargs or {})
            self.local = CGMEncoder(n_channels, d_out, dropout=dropout, **lk)
            # Zero-initialised adapter: the sum starts at exactly the baseline
            # encoder's output, and grad wrt these weights is the incoming
            # gradient times the (non-zero) FM features, so the branch trains
            # from step one. The local encoder already ends in a LayerNorm, so
            # no second norm here -- that keeps init BIT-for-bit identical.
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
            self.norm = nn.Identity()
        else:
            self.local = None
            self.norm = nn.LayerNorm(d_out)

    # -- subclasses provide the network and how to call it --------------------
    def _fm_forward(self, g: torch.Tensor, obs: torch.Tensor,
                    x_mark: Optional[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def _fm_module(self) -> nn.Module:
        raise NotImplementedError

    # ------------------------------------------------------------- freezing
    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self._fm_module().eval()      # keep dropout/BN off in a frozen tower
        return self

    def _apply_freeze(self) -> None:
        if self.frozen:
            for p in self._fm_module().parameters():
                p.requires_grad_(False)
            self._fm_module().eval()

    # -------------------------------------------------------------- helpers
    def _split(self, x: torch.Tensor, mask: Optional[torch.Tensor]):
        g = x[..., 0]
        if mask is not None and mask.shape[-1] >= 1:
            obs = (mask[..., 0] > 0.5).float()
        elif x.shape[-1] > 1:
            obs = (x[..., 1] > 0.5).float()
        else:
            obs = torch.ones_like(g)
        return g, obs

    def _normalise(self, g: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        if self.input_norm == "passthrough":
            return g
        return mask_aware_instance_norm(g, obs)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                time: Optional[torch.Tensor] = None) -> torch.Tensor:
        g, obs = self._split(x, mask)
        gn = self._normalise(g, obs)
        gn = torch.where(obs > 0.5, gn, torch.full_like(gn, self.missing_fill))

        T = gn.shape[1]
        pad = (-T) % self.patch                       # left-pad to a whole patch
        if pad:
            head = torch.full((gn.shape[0], pad), self.missing_fill,
                              dtype=gn.dtype, device=gn.device)
            gn = torch.cat([head, gn], dim=1)
            obs = torch.cat([torch.zeros_like(head), obs], dim=1)

        x_mark = None
        if self.use_time and time is not None:
            tf = informer_time_features(time)          # [B, T, 5]
            if pad:
                tf = torch.cat([tf.new_zeros(tf.shape[0], pad, tf.shape[-1]), tf], dim=1)
            B, TT, D = tf.shape
            x_mark = tf.view(B, TT // self.patch, self.patch, D)

        ctx = torch.no_grad() if self.frozen else torch.enable_grad()
        with ctx:
            feat = self._fm_forward(gn, obs, x_mark)   # [B, n_patches, d_fm]
        if self.frozen:
            feat = feat.detach()

        up = feat.repeat_interleave(self.patch, dim=1)  # [B, T + pad, d_fm]
        if pad:
            up = up[:, pad:]
        h = self.proj(up)
        if self.local is not None:
            h = self.local(x, mask) + h
        return self.norm(h)

    # ------------------------------------------------------------- reporting
    def fm_summary(self) -> Dict[str, object]:
        d: Dict[str, object] = {"mode": self.mode, "frozen": self.frozen,
                                "patch": self.patch, "use_time": self.use_time,
                                "input_norm": self.input_norm}
        if self.local is not None:
            # 0.0 at initialisation; how far it moves is how much of the
            # foundation model the forecaster decided to use.
            d["adapter_norm"] = round(float(self.proj.weight.detach().norm()), 5)
        return d


class CGMJepaEncoder(_FMCGMEncoder):
    """CGM-JEPA / X-CGM-JEPA (arXiv:2605.00933), published MIT weights."""

    def __init__(self, n_channels: int, d_out: int, *, weights: str,
                 variant: str = "cgm_jepa", **kw) -> None:
        from .cgm_jepa_net import load_pretrained
        net = load_pretrained(weights, variant)
        kw.setdefault("patch", net.patch_len)
        # they trained with -1 as the explicit missing sentinel, post-normalisation
        kw.setdefault("missing_fill", -1.0)
        super().__init__(n_channels, d_out, d_fm=net.embed_dim, **kw)
        self.net = net
        self.variant = variant
        self._apply_freeze()

    def _fm_module(self) -> nn.Module:
        return self.net

    def _fm_forward(self, g, obs, x_mark):
        B, T = g.shape
        emb, _ = self.net(g.view(B, T // self.patch, self.patch), x_mark)
        return emb

    def fm_summary(self):
        d = super().fm_summary()
        d.update({"model": "cgm_jepa", "variant": self.variant, "pretrained": True})
        return d


class GlucoFMEncoder(_FMCGMEncoder):
    """GlucoFM (arXiv:2605.30865) architecture.

    ``weights`` is optional and, when given, must be a checkpoint written by
    ``metabonet_forecasting.fm.glucofm_pretrain`` on our own corpus -- there is
    no public GlucoFM checkpoint to load. Without it this is a randomly
    initialised dual-stream encoder trained end-to-end with the forecaster,
    which is a fair architecture test but is NOT "the foundation model".
    """

    def __init__(self, n_channels: int, d_out: int, *, weights: Optional[str] = None,
                 d_fm: int = 128, n_heads: int = 4, n_layers: int = 3, ffn: int = 256,
                 sigma_min: float = 2.0, sigma_max: float = 12.0,
                 fm_dropout: float = 0.0, **kw) -> None:
        from .glucofm_net import GlucoFMNet
        kw.setdefault("patch", 12)
        kw.setdefault("freeze", False)      # nothing pretrained unless weights= given
        kw.setdefault("missing_fill", 0.0)  # mask-aware streams; 0 is the neutral value
        net = GlucoFMNet(d_fm=d_fm, n_heads=n_heads, n_layers=n_layers, ffn=ffn,
                         patch=int(kw["patch"]), dropout=fm_dropout,
                         sigma_min=sigma_min, sigma_max=sigma_max)
        super().__init__(n_channels, d_out, d_fm=d_fm, **kw)
        self.net = net
        self.pretrained = False
        if weights:
            sd = torch.load(weights, map_location="cpu")
            sd = sd.get("encoder", sd)
            missing, unexpected = self.net.load_state_dict(sd, strict=False)
            if unexpected:
                raise RuntimeError(f"unexpected tensors in {weights}: {list(unexpected)}")
            self.pretrained = True
        self._apply_freeze()

    def _fm_module(self) -> nn.Module:
        return self.net

    def _fm_forward(self, g, obs, x_mark):
        return self.net(g, obs, x_mark)

    def fm_summary(self):
        d = super().fm_summary()
        d.update({"model": "glucofm", "pretrained": self.pretrained,
                  "sigma_steps": round(float(self.net.filter.sigma().detach()), 3)})
        return d


def build_fm_cgm_encoder(kind: str, n_channels: int, d_out: int,
                         fm_cfg: Dict, dropout: float,
                         local_kwargs: Optional[Dict] = None) -> nn.Module:
    """``model.encoder.cgm_encoder`` -> the encoder object."""
    cfg = {k: v for k, v in dict(fm_cfg or {}).items() if v is not None}
    cfg.setdefault("dropout", dropout)
    cfg["local_kwargs"] = local_kwargs or {}
    if kind in ("cgm_jepa", "x_cgm_jepa"):
        cfg.setdefault("weights", "artifacts/fm_weights")
        cfg["variant"] = cfg.pop("variant", kind)
        if kind == "x_cgm_jepa":
            cfg["variant"] = "x_cgm_jepa"
        return CGMJepaEncoder(n_channels, d_out, **cfg)
    if kind == "glucofm":
        cfg.pop("variant", None)
        return GlucoFMEncoder(n_channels, d_out, **cfg)
    raise ValueError(
        f"unknown model.encoder.cgm_encoder={kind!r}; "
        f"expected default | cgm_jepa | x_cgm_jepa | glucofm")
