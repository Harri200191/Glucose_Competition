"""Normalisation, fitted on training subjects only.

Two risks this module exists to avoid (plan section 12):

* **temporal leakage** -- a statistic fitted over the whole corpus lets test-period
  information into training inputs. Every statistic here is fitted on anchors from
  the training mask alone and then frozen into a JSON file that ships with the run;
* **destroying the signal by normalising it wrongly** -- bolus and carbohydrate are
  spike trains (93% and 98% exact zeros, skew ~15), so mean/sd standardisation
  makes almost every sample identical. They get log1p and a high-quantile scale
  instead, while CGM gets a robust median/IQR.

The per-channel rule is chosen by channel kind, not by hand, so adding a channel
cannot silently get the wrong treatment.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import constants as K

# Target parameterisations that live in log space. Zone A is a +/-20% band, so a
# log target turns the competition's relative criterion into an ordinary
# absolute one -- see Standardizer.transform_targets.
LOG_TARGET_MODES = ("log_ratio", "log")
# exp() bounds, in log mg/dL. Wide enough never to bind on a sane forecast,
# tight enough that an untrained model cannot produce inf.
LOG_EXP_LO, LOG_EXP_HI = float(np.log(1.0)), float(np.log(2000.0))


def channel_kind(name: str) -> str:
    """How a channel should be scaled, from its name.

    identity   already 0/1 (observation masks)
    since      minutes-since-event; mapped to [0, 1] by the 24 h cap
    positive   non-negative spike train or exposure; log1p then quantile scale
    robust     roughly continuous physiological signal; median / IQR
    """
    if name.endswith("_mask"):
        return "identity"
    if name.endswith("_since"):
        return "since"
    if name in ("cgm", "act_heartrate"):
        return "robust"
    return "positive"


@dataclass
class ChannelStat:
    kind: str
    center: float
    scale: float


class Standardizer:
    """Frozen normalisation for channels, statics and targets."""

    def __init__(self, channels: Dict[str, ChannelStat],
                 static_center: List[float], static_scale: List[float],
                 target_center: List[float], target_scale: List[float],
                 target_mode: str, horizons_min: List[int]) -> None:
        self.channels = channels
        self.static_center = np.asarray(static_center, dtype=np.float32)
        self.static_scale = np.asarray(static_scale, dtype=np.float32)
        self.target_center = np.asarray(target_center, dtype=np.float32)
        self.target_scale = np.asarray(target_scale, dtype=np.float32)
        self.target_mode = target_mode
        self.horizons_min = list(horizons_min)

    # ------------------------------------------------------------------- apply
    def transform_channels(self, x: np.ndarray, names: Sequence[str]) -> np.ndarray:
        """``x``: [T, C] raw channel values in physical units -> normalised copy.

        NaN is preserved, not filled: filling is a modelling choice made later
        (``windows.py``) so that the zero-fill / forward-fill / learned-embedding
        ablation is a config switch rather than a rewrite.
        """
        out = np.array(x, dtype=np.float32, copy=True)
        for j, name in enumerate(names):
            st = self.channels[name]
            col = out[:, j]
            if st.kind == "identity":
                continue
            if st.kind == "since":
                out[:, j] = np.clip(col / max(st.scale, 1e-6), 0.0, 1.0)
            elif st.kind == "positive":
                out[:, j] = np.log1p(np.maximum(col, 0.0)) / max(st.scale, 1e-6)
            else:
                out[:, j] = (col - st.center) / max(st.scale, 1e-6)
        return out

    def transform_statics(self, s: np.ndarray) -> np.ndarray:
        """Numeric statics only; categorical codes are passed through separately."""
        n_num = self.static_center.size
        out = (np.asarray(s[..., :n_num], dtype=np.float32) - self.static_center) \
            / np.maximum(self.static_scale, 1e-6)
        return np.clip(np.nan_to_num(out, nan=0.0), -8.0, 8.0)

    def transform_targets(self, y: np.ndarray, cgm_now: np.ndarray) -> np.ndarray:
        """mg/dL -> model space.

        ``delta``      predict the change from the last reading (the default)
        ``absolute``   predict the level
        ``log_ratio``  predict ``log(y / cgm_now)``
        ``log``        predict ``log(y)``

        The log modes exist because the competition metric is relative, not
        absolute: DTS zone A is exactly ``0.8*y <= p <= 1.2*y``. Under
        ``log_ratio`` the model output and the target differ by exactly
        ``log(p/y)``, so zone A becomes a **constant-width band on the raw model
        output** and an ordinary absolute-error loss in this space *is* a
        relative-error loss in mg/dL. Under ``delta`` it is not: a 20 mg/dL miss
        is scored identically at a true 80 (zone B) and a true 300 (zone A),
        while the loss sees one number.

        Non-positive or missing glucose becomes NaN here rather than -inf, and
        the mask drops it exactly as it drops any other absent target.
        """
        y = np.asarray(y, dtype=np.float32)
        z = self._to_model_space(y, cgm_now)
        return (z - self.target_center) / np.maximum(self.target_scale, 1e-6)

    def _to_model_space(self, y: np.ndarray, cgm_now: np.ndarray) -> np.ndarray:
        """mg/dL -> the pre-standardisation target, per ``target_mode``."""
        now = np.asarray(cgm_now, dtype=np.float32).reshape(-1, 1)
        if self.target_mode in LOG_TARGET_MODES:
            with np.errstate(invalid="ignore", divide="ignore"):
                out = np.log(np.where(y > 0, y, np.nan))
                if self.target_mode == "log_ratio":
                    out = out - np.log(np.where(now > 0, now, np.nan))
            return out.astype(np.float32)
        base = now if self.target_mode == "delta" else 0.0
        return y - base

    def inverse_targets(self, z, cgm_now):
        """Model space -> mg/dL. Works for numpy arrays and torch tensors alike.

        The torch check is ``detach``, not ``device``: numpy 2 arrays carry a
        ``device`` attribute for array-API compliance, so the obvious test picks
        the wrong branch and fails on the dtype.
        """
        log_mode = self.target_mode in LOG_TARGET_MODES
        if hasattr(z, "detach"):          # torch tensor
            import torch
            center = torch.as_tensor(self.target_center, device=z.device, dtype=z.dtype)
            scale = torch.as_tensor(self.target_scale, device=z.device, dtype=z.dtype)
            if log_mode:
                out = z * scale + center
                if self.target_mode == "log_ratio":
                    out = out + torch.log(cgm_now.reshape(-1, 1).clamp(min=1.0))
                # Clamp the exponent, not the result: exp() of an early-training
                # outlier overflows to inf and poisons the whole batch.
                return torch.exp(out.clamp(LOG_EXP_LO, LOG_EXP_HI))
            base = cgm_now.reshape(-1, 1) if self.target_mode == "delta" else 0.0
            return z * scale + center + base
        center, scale = self.target_center, self.target_scale
        now = np.asarray(cgm_now, dtype=np.float32).reshape(-1, 1)
        if log_mode:
            out = np.asarray(z, dtype=np.float32) * scale + center
            if self.target_mode == "log_ratio":
                out = out + np.log(np.maximum(now, 1.0))
            return np.exp(np.clip(out, LOG_EXP_LO, LOG_EXP_HI)).astype(np.float32)
        base = now if self.target_mode == "delta" else 0.0
        return np.asarray(z, dtype=np.float32) * scale + center + base

    # -------------------------------------------------------------------- I/O
    def to_json(self) -> Dict:
        return {
            "channels": {k: asdict(v) for k, v in self.channels.items()},
            "static_center": self.static_center.tolist(),
            "static_scale": self.static_scale.tolist(),
            "target_center": self.target_center.tolist(),
            "target_scale": self.target_scale.tolist(),
            "target_mode": self.target_mode,
            "horizons_min": self.horizons_min,
        }

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_json(), fh, indent=2)

    @classmethod
    def load(cls, path) -> "Standardizer":
        with open(path) as fh:
            blob = json.load(fh)
        return cls.from_json(blob)

    @classmethod
    def from_json(cls, blob: Dict) -> "Standardizer":
        return cls(
            channels={k: ChannelStat(**v) for k, v in blob["channels"].items()},
            static_center=blob["static_center"], static_scale=blob["static_scale"],
            target_center=blob["target_center"], target_scale=blob["target_scale"],
            target_mode=blob["target_mode"], horizons_min=blob["horizons_min"],
        )


# --------------------------------------------------------------------- fitting
def fit_standardizer(store, train_mask: np.ndarray, channel_names: Sequence[str],
                     horizon_ids: Sequence[int], target_mode: str = "delta",
                     n_sample_anchors: int = 200_000, seed: int = 0) -> Standardizer:
    """Fit every statistic on training anchors only.

    Channel statistics are computed over the anchor rows themselves rather than
    over whole windows: an anchor row is a uniform draw from the same lattice its
    window is drawn from, and sampling 200k rows instead of 200k x 288 keeps the
    fit to a second or two.
    """
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(train_mask)
    if idx.size == 0:
        raise ValueError("cannot fit a standardizer on an empty training mask")
    if idx.size > n_sample_anchors:
        idx = np.sort(rng.choice(idx, size=n_sample_anchors, replace=False))
    rows = store.anchors["row"].to_numpy()[idx]

    ch_ids = store.channel_ids(channel_names)
    sample = np.asarray(store.channels[rows][:, ch_ids], dtype=np.float64)

    stats: Dict[str, ChannelStat] = {}
    for j, name in enumerate(channel_names):
        kind = channel_kind(name)
        col = sample[:, j]
        col = col[np.isfinite(col)]
        if kind == "identity":
            stats[name] = ChannelStat(kind, 0.0, 1.0)
        elif kind == "since":
            stats[name] = ChannelStat(kind, 0.0, float(K.SINCE_CAP_MIN))
        elif kind == "positive":
            logged = np.log1p(np.maximum(col, 0.0))
            scale = float(np.quantile(logged, 0.99)) if logged.size else 1.0
            stats[name] = ChannelStat(kind, 0.0, scale if scale > 1e-6 else 1.0)
        else:
            med = float(np.median(col)) if col.size else 0.0
            q1, q3 = (np.quantile(col, [0.25, 0.75]) if col.size else (0.0, 1.0))
            iqr = float(q3 - q1)
            stats[name] = ChannelStat(kind, med, (iqr / 1.349) if iqr > 1e-6 else 1.0)

    # statics: robust centre/scale over the *subjects* present in training
    subj = np.unique(store.anchors["subject"].to_numpy()[idx])
    n_num = len(store.manifest["static_numeric"])
    s = store.statics[subj, :n_num].astype(np.float64)
    with np.errstate(invalid="ignore"):
        centre = np.nanmedian(s, axis=0)
        q1 = np.nanquantile(s, 0.25, axis=0)
        q3 = np.nanquantile(s, 0.75, axis=0)
    scale = (q3 - q1) / 1.349
    centre = np.nan_to_num(centre, nan=0.0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)

    # targets
    y = np.asarray(store.targets[idx][:, list(horizon_ids)], dtype=np.float64)
    now = store.anchors["cgm_now"].to_numpy()[idx].reshape(-1, 1)
    if target_mode == "delta":
        y = y - now
    elif target_mode in LOG_TARGET_MODES:
        with np.errstate(invalid="ignore", divide="ignore"):
            y = np.log(np.where(y > 0, y, np.nan))
            if target_mode == "log_ratio":
                y = y - np.log(np.where(now > 0, now, np.nan))
    elif target_mode != "absolute":
        raise ValueError(
            f"unknown target_mode {target_mode!r}; expected delta, absolute, "
            f"log_ratio or log")
    with np.errstate(invalid="ignore"):
        t_centre = np.nanmean(y, axis=0)
        t_scale = np.nanstd(y, axis=0)
    t_centre = np.nan_to_num(t_centre, nan=0.0)
    t_scale = np.where(np.isfinite(t_scale) & (t_scale > 1e-6), t_scale, 1.0)

    horizons = [store.horizons[i] for i in horizon_ids]
    return Standardizer(stats, centre.tolist(), scale.tolist(),
                        t_centre.tolist(), t_scale.tolist(), target_mode, horizons)


# ------------------------------------------------------------------- NaN policy
def fill_missing(x: np.ndarray, mask: Optional[np.ndarray], policy: str) -> np.ndarray:
    """Fill non-finite entries of a normalised window.

    ``zero``   the benchmark's simple rule: missing becomes the training centre
    ``ffill``  carry the last observed value forward (causal), then zero at the head
    ``learned`` zero here; the encoder adds a learned missing-modality embedding
               where the mask says the value is absent (plan 4.6)
    """
    out = np.array(x, dtype=np.float32, copy=True)
    bad = ~np.isfinite(out)
    if not bad.any():
        return out
    if policy in ("zero", "learned"):
        out[bad] = 0.0
        return out
    if policy != "ffill":
        raise ValueError(f"unknown missing-value policy {policy!r}")
    for j in range(out.shape[1]):
        col = out[:, j]
        miss = ~np.isfinite(col)
        if not miss.any():
            continue
        idx = np.where(~miss, np.arange(col.size), 0)
        np.maximum.accumulate(idx, out=idx)
        filled = col[idx]
        filled[~np.isfinite(filled)] = 0.0
        out[:, j] = filled
    return out
