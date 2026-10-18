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
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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

    # --- planned future actions: OFF by default, and must stay that way -------
    # Empty means the window is strictly causal, which is the benchmark. Adding
    # "insulin" and/or "carbs" lets the model read those channels from *after*
    # the anchor, which changes the task (see constants.FUTURE_MODALITY_CHANNELS)
    # and makes the result incomparable with every causal run.
    future_modalities: List[str] = field(default_factory=list)
    # How far past the anchor to read. None = out to the longest horizon, so the
    # model sees the interventions covering exactly the span it forecasts.
    future_steps: Optional[int] = None
    # Directory written by metabonet_forecasting.features.build: a per-anchor
    # engineered-feature matrix, keyed by POSITIONAL anchor index the way
    # targets.npy is. None = the feature block is off and x_feat is zero-width.
    feature_path: Optional[str] = None
    # "all" | "tier1" | comma-separated family names. The matrix is built once
    # with every family; a subset selects COLUMNS from it, so an ablation costs
    # no rebuild and every arm reads the identical file.
    feature_families: str = "all"

    def n_future_steps(self) -> int:
        if not self.future_modalities:
            return 0
        if self.future_steps is not None:
            return int(self.future_steps)
        return max(self.horizons_min) // K.GRID_MIN

    def future_channels(self) -> List[str]:
        out: List[str] = []
        for m in self.future_modalities:
            out.extend(K.FUTURE_MODALITY_CHANNELS[m])
        return out

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
        """Every channel a run touches, history and future alike.

        Normalisation statistics are fitted over this list, so a future-only
        channel still gets scaled from training data rather than left raw.
        """
        out: List[str] = []
        for m in self.modalities:
            out.extend(self.channels_for(m))
        for name in self.future_channels():
            if name not in out:
                out.append(name)
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
        unknown_future = [m for m in self.future_modalities
                          if m not in K.FUTURE_MODALITY_CHANNELS]
        if unknown_future:
            raise ValueError(f"unknown future modalities {unknown_future}; "
                             f"have {sorted(K.FUTURE_MODALITY_CHANNELS)}")
        if self.future_modalities and self.n_future_steps() <= 0:
            raise ValueError("future_steps must be positive when future "
                             "modalities are requested")
        if self.feature_path is not None:
            root = Path(self.feature_path)
            missing = [f for f in ("features.npy", "lookup.npy", "columns.json")
                       if not (root / f).exists()]
            if missing:
                raise ValueError(
                    f"data.features.path={self.feature_path!r} is not a feature "
                    f"build: missing {missing}. Build it with "
                    "python -m metabonet_forecasting.features.build --out <dir>")


class WindowDataset(Dataset):
    """Lazily gathers windows out of the memmapped store."""

    def __init__(self, store: SequenceStore, mask: np.ndarray, spec: WindowSpec,
                 standardizer: Standardizer, pad_history: bool = False) -> None:
        spec.validate()
        # pad_history exists for submissions: a competition template asks for a
        # prediction at every row it lists, including rows whose recording starts
        # less than `context` ago. Training never uses it -- the store's anchor
        # rules guarantee full history there -- so the benchmark cannot silently
        # start training on padded windows.
        self.pad_history = pad_history
        if spec.context > store.min_history_steps and not pad_history:
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

        # Engineered features. Memmapped in __init__ so forked DataLoader workers
        # inherit it, exactly as SequenceStore does with channels/targets.
        self.feats = None
        self.feat_lookup = None
        self.n_feat = 0
        if spec.feature_path is not None:
            root = Path(spec.feature_path)
            self.feats = np.load(root / "features.npy", mmap_mode="r")
            self.feat_lookup = np.load(root / "lookup.npy")
            from ..features.spec import select_columns
            self.feat_cols = select_columns(root, spec.feature_families,
                                            spec.context)
            self.n_feat = int(self.feat_cols.size)
            if self.feat_lookup.shape[0] != len(store.anchors):
                raise ValueError(
                    f"feature lookup covers {self.feat_lookup.shape[0]:,} anchors but "
                    f"the store has {len(store.anchors):,}; the features were built "
                    "against a different store")
            miss = int((self.feat_lookup[self.indices] < 0).sum())
            if miss:
                # Loudly, rather than serving some other anchor's row.
                raise ValueError(
                    f"{miss:,} of {self.indices.size:,} anchors in this split have no "
                    "row in the feature matrix. Rebuild it over this split's anchors.")

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

        # --- planned future actions, only when a run has opted in -------------
        self.future_names: List[str] = spec.future_channels()
        self.n_future = spec.n_future_steps()
        if self.future_names:
            self.future_ids = store.channel_ids(self.future_names)
            # How far the future block may legitimately run: never past the end
            # of the anchor's own segment. Beyond that the next rows belong to a
            # different recording (or a different patient), so they are zeroed
            # and flagged rather than read.
            seg_end = store.segments.set_index("seg")["end"]
            self.seg_end = seg_end.loc[a["seg"].to_numpy()].to_numpy()
        else:
            self.future_ids = np.zeros(0, dtype=np.int64)
            self.seg_end = None
        # Where each anchor's own recording begins, so a padded window can never
        # reach back into the previous patient's data.
        self.seg_start = (store.segments.set_index("seg")["start"]
                          .loc[a["seg"].to_numpy()].to_numpy() if pad_history else None)

    # ------------------------------------------------------------------ shapes
    def channel_counts(self) -> Dict[str, int]:
        return {m: len(self.channel_names[m]) for m in self.modalities}

    @property
    def n_extra_features(self) -> int:
        """Width of the engineered per-anchor block; 0 when it is off."""
        return self.n_feat

    @property
    def n_time_features(self) -> int:
        return 4 if self.spec.use_time_features else 0

    @property
    def n_horizons(self) -> int:
        return len(self.spec.horizons_min)

    @property
    def future_shape(self) -> Tuple[int, int]:
        """(steps, channels) of the planned-action block; (0, 0) when disabled."""
        return (self.n_future, len(self.future_names)) if self.future_names else (0, 0)

    def __len__(self) -> int:
        return int(self.indices.size)

    # ------------------------------------------------------------------- item
    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        ai = int(self.indices[i])
        row = int(self.rows[ai])
        L = self.spec.context
        lo = row - L + 1
        n_pad = 0
        if self.pad_history:
            lo = max(lo, int(self.seg_start[ai]), 0)
            n_pad = L - (row + 1 - lo)
        block = np.asarray(self.store.channels[lo: row + 1])

        item: Dict[str, torch.Tensor] = {}
        for m in self.modalities:
            raw = block[:, self.channel_ids[m]]
            norm = self.std.transform_channels(raw, self.channel_names[m])
            mask_cols = [j for j, n in enumerate(self.channel_names[m])
                         if n.endswith("_mask")]
            filled = fill_missing(norm, None, self.spec.missing_policy)
            if n_pad:
                # Zeros at the head: for a value channel that is the training
                # centre, and for an observation mask it is an explicit "not
                # recorded", which is exactly what a window that predates the
                # recording should say.
                filled = np.concatenate(
                    [np.zeros((n_pad, filled.shape[1]), np.float32), filled], axis=0)
            item[f"x_{m}"] = torch.from_numpy(np.ascontiguousarray(filled))
            if mask_cols:
                item[f"m_{m}"] = torch.from_numpy(
                    np.ascontiguousarray(filled[:, mask_cols]))

        if self.future_names:
            # Rows strictly after the anchor. Anything past the segment end is
            # zero-filled and marked absent in the trailing validity column, so
            # a short segment cannot leak the start of the next recording.
            n_f = self.n_future
            usable = int(min(n_f, max(int(self.seg_end[ai]) - row - 1, 0)))
            block = np.zeros((n_f, len(self.future_names) + 1), dtype=np.float32)
            if usable > 0:
                raw = np.asarray(
                    self.store.channels[row + 1: row + 1 + usable])[:, self.future_ids]
                block[:usable, :-1] = fill_missing(
                    self.std.transform_channels(raw, self.future_names),
                    None, self.spec.missing_policy)
                block[:usable, -1] = 1.0
            item["x_future"] = torch.from_numpy(block)

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

        if self.feats is not None:
            fi = int(self.feat_lookup[ai])
            row = np.asarray(self.feats[fi], dtype=np.float32)[self.feat_cols]
            item["x_feat"] = torch.from_numpy(self.std.transform_features(row))
        else:
            # Zero-width rather than absent: collate stacks a fixed key set.
            item["x_feat"] = torch.zeros(0, dtype=torch.float32)

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
