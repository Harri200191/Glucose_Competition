"""The common window dataset.

Plan stage A.4: *"create a common window dataset class that returns identical
windows to all custom models."* This is that class. TCN, PatchTST, Mamba-2, TFT,
TiDE, the LSTM and every classical baseline consume exactly these tensors, so a
difference between two runs is a difference between two models.

One item is one anchor:

    x_cgm       [L, n_cgm]      state channels    (value + observation mask)
    x_insulin   [L, n_ins]      driver channels   (basal, bolus, exposures, recency)
    x_carbs     [L, n_carb]     driver channels
    x_activity  [L, n_act]      optional contextual modifiers
    time        [L, 4]          time-of-day / day-of-week, cyclically encoded
    static_num  [S]             normalised numeric statics
    static_cat  [C]             categorical codes (0 = unknown/absent)
    y           [H]             CGM at t+h, mg/dL, NaN where unusable
    y_norm      [H]             the same in model space (delta or absolute)
    y_mask      [H]             1 where the horizon has a usable label
    cgm_now     scalar          last real CGM in mg/dL, for inverting predictions
    index       scalar          row in the store's anchor table

Causality is structural: the window is ``store.channels[row-L+1 : row+1]`` and the
label is read from ``targets.npy``, which prepare.py built from rows strictly after
``row``. No transform in this file looks forward.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from . import constants as K
from .store import SequenceStore
from .transforms import Standardizer, fill_missing

MODALITY_ORDER: List[str] = ["cgm", "insulin", "carbs", "activity"]


@dataclass
class WindowSpec:
    """Everything that decides what a window contains. Hashed into run manifests."""
    context: int = 288                       # lattice steps of history (288 = 24 h)
    horizons_min: List[int] = field(default_factory=lambda: list(K.PRIMARY_HORIZONS_MIN))
    modalities: List[str] = field(default_factory=lambda: ["cgm", "insulin", "carbs"])
    use_event_features: bool = True          # decay kernels + time-since-event (D10)
    use_masks: bool = True                   # observation masks (D11)
    use_static: bool = True
    use_source_identity: bool = False        # cohort id as a static: shortcut risk
    use_time_features: bool = True
    missing_policy: str = "zero"             # zero | ffill | learned

    def channels_for(self, modality: str) -> List[str]:
        names = list(K.MODALITY_CHANNELS[modality])
        if not self.use_event_features:
            names = [n for n in names if n not in K.EVENT_FEATURE_CHANNELS]
        if not self.use_masks:
            kept = [n for n in names if not n.endswith("_mask")]
            # CGM must keep at least one channel; if masks are off we still need
            # the value itself, which is always present.
            names = kept or names
        return names

    def all_channels(self) -> List[str]:
        out: List[str] = []
        for m in self.modalities:
            out.extend(self.channels_for(m))
        return out

    def validate(self) -> None:
        unknown = [m for m in self.modalities if m not in K.MODALITY_CHANNELS]
        if unknown:
            raise ValueError(f"unknown modalities {unknown}; "
                             f"have {sorted(K.MODALITY_CHANNELS)}")
        if "cgm" not in self.modalities:
            raise ValueError("cgm is the forecast state and cannot be ablated away")
        if self.missing_policy not in ("zero", "ffill", "learned"):
            raise ValueError(f"unknown missing_policy {self.missing_policy!r}")


class WindowDataset(Dataset):
    """Lazily gathers windows out of the memmapped store."""

    def __init__(self, store: SequenceStore, mask: np.ndarray, spec: WindowSpec,
                 standardizer: Standardizer) -> None:
        spec.validate()
        if spec.context > store.min_history_steps:
            raise ValueError(
                f"context {spec.context} exceeds the history guaranteed by this store "
                f"({store.min_history_steps}); rebuild with --min-history-steps")
        self.store = store
        self.spec = spec
        self.std = standardizer
        self.indices = np.flatnonzero(mask).astype(np.int64)

        self.modalities = [m for m in MODALITY_ORDER if m in spec.modalities]
        self.channel_names: Dict[str, List[str]] = {
            m: spec.channels_for(m) for m in self.modalities}
        self.channel_ids: Dict[str, np.ndarray] = {
            m: store.channel_ids(self.channel_names[m]) for m in self.modalities}

        self.horizon_ids = store.horizon_ids(spec.horizons_min)
        a = store.anchors
        self.rows = a["row"].to_numpy()
        self.cgm_now = a["cgm_now"].to_numpy().astype(np.float32)
        self.subject = a["subject"].to_numpy()
        self.tod_min = a["tod_min"].to_numpy().astype(np.float32)
        self.dow = a["dow"].to_numpy().astype(np.float32)

        self.static_cat_fields = list(K.STATIC_CATEGORICAL) if spec.use_source_identity \
            else list(K.STATIC_CATEGORICAL_PRIMARY)
        self.n_static_numeric = len(store.manifest["static_numeric"])
        self._cat_cols = [K.STATIC_CATEGORICAL.index(f) + self.n_static_numeric
                          for f in self.static_cat_fields]
        # +1 everywhere: code 0 is reserved for "unknown / not recorded".
        self.cat_cardinalities = [store.n_levels(f) + 1 for f in self.static_cat_fields]

        # Relative minute offsets of the window's steps from the anchor.
        self._offsets = np.arange(-(spec.context - 1), 1, dtype=np.float32) * K.GRID_MIN

    # ------------------------------------------------------------------ shapes
    def channel_counts(self) -> Dict[str, int]:
        return {m: len(self.channel_names[m]) for m in self.modalities}

    @property
    def n_time_features(self) -> int:
        return 4 if self.spec.use_time_features else 0

    @property
    def n_horizons(self) -> int:
        return len(self.spec.horizons_min)

    def __len__(self) -> int:
        return int(self.indices.size)

    # ------------------------------------------------------------------- item
    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        ai = int(self.indices[i])
        row = int(self.rows[ai])
        L = self.spec.context
        block = np.asarray(self.store.channels[row - L + 1: row + 1])

        item: Dict[str, torch.Tensor] = {}
        for m in self.modalities:
            raw = block[:, self.channel_ids[m]]
            norm = self.std.transform_channels(raw, self.channel_names[m])
            mask_cols = [j for j, n in enumerate(self.channel_names[m])
                         if n.endswith("_mask")]
            filled = fill_missing(norm, None, self.spec.missing_policy)
            item[f"x_{m}"] = torch.from_numpy(np.ascontiguousarray(filled))
            if mask_cols:
                item[f"m_{m}"] = torch.from_numpy(
                    np.ascontiguousarray(filled[:, mask_cols]))

        if self.spec.use_time_features:
            minutes = (self.tod_min[ai] + self._offsets) % 1440.0
            day = (self.dow[ai] + np.floor((self.tod_min[ai] + self._offsets) / 1440.0)) % 7.0
            tod = 2.0 * np.pi * minutes / 1440.0
            dow = 2.0 * np.pi * day / 7.0
            item["time"] = torch.from_numpy(np.stack(
                [np.sin(tod), np.cos(tod), np.sin(dow), np.cos(dow)], axis=1
            ).astype(np.float32))

        subj = int(self.subject[ai])
        stat = self.store.statics[subj]
        if self.spec.use_static:
            item["static_num"] = torch.from_numpy(
                self.std.transform_statics(stat[None, :])[0].astype(np.float32))
            cats = np.array([stat[c] for c in self._cat_cols], dtype=np.float32)
            cats = np.where(np.isfinite(cats) & (cats >= 0), cats + 1.0, 0.0)
            item["static_cat"] = torch.from_numpy(cats.astype(np.int64))
        else:
            item["static_num"] = torch.zeros(0, dtype=torch.float32)
            item["static_cat"] = torch.zeros(0, dtype=torch.int64)

        y = np.asarray(self.store.targets[ai][self.horizon_ids], dtype=np.float32)
        y_mask = np.isfinite(y).astype(np.float32)
        cgm_now = float(self.cgm_now[ai])
        y_norm = self.std.transform_targets(np.nan_to_num(y, nan=0.0)[None, :],
                                            np.array([cgm_now]))[0]
        item["y"] = torch.from_numpy(np.nan_to_num(y, nan=0.0))
        item["y_norm"] = torch.from_numpy(np.nan_to_num(y_norm, nan=0.0).astype(np.float32))
        item["y_mask"] = torch.from_numpy(y_mask)
        item["cgm_now"] = torch.tensor(cgm_now, dtype=torch.float32)
        item["index"] = torch.tensor(ai, dtype=torch.int64)
        return item


def collate(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Plain stacking; every item has the same keys and shapes by construction."""
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def batch_to_device(batch: Dict[str, torch.Tensor], device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def describe_batch(batch: Dict[str, torch.Tensor]) -> str:
    return ", ".join(f"{k}{tuple(v.shape)}" for k, v in sorted(batch.items()))
