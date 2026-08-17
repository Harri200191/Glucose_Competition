"""Config -> (store, split, standardizer, dataloaders).

This is the only place that turns a YAML ``data:`` block into tensors, which is
what makes "same preprocessing, same patients, same windows, different backbone"
enforceable rather than aspirational.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from torch.utils.data import DataLoader

from ..common.config import Config
from ..common.seeding import worker_init_fn
from . import constants as K
from . import splits as S
from .store import SequenceStore
from .transforms import Standardizer, fit_standardizer
from .windows import WindowDataset, WindowSpec, collate


@dataclass
class DataBundle:
    store: SequenceStore
    spec: WindowSpec
    standardizer: Standardizer
    split: Dict[str, np.ndarray]
    strata: Dict[str, np.ndarray]
    datasets: Dict[str, WindowDataset]

    def loader(self, part: str, cfg: Config, shuffle: Optional[bool] = None) -> DataLoader:
        loader_cfg = cfg.get_path("data.loader", {})
        if shuffle is None:
            shuffle = (part == "train")
        num_workers = int(loader_cfg.get("num_workers", 8))
        kwargs = dict(
            batch_size=int(loader_cfg.get("batch_size", 256)),
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=bool(loader_cfg.get("pin_memory", True)),
            drop_last=(part == "train"),
            collate_fn=collate,
            worker_init_fn=worker_init_fn,
        )
        if num_workers > 0:
            kwargs["persistent_workers"] = bool(loader_cfg.get("persistent_workers", True))
            kwargs["prefetch_factor"] = int(loader_cfg.get("prefetch_factor", 4))
        return DataLoader(self.datasets[part], **kwargs)

    def summary(self) -> Dict[str, object]:
        return {
            "store": str(self.store.root),
            "context": self.spec.context,
            "horizons_min": list(self.spec.horizons_min),
            "modalities": list(self.spec.modalities),
            "future_modalities": list(self.spec.future_modalities),
            "future_steps": self.spec.n_future_steps(),
            "channels": {m: self.spec.channels_for(m) for m in self.spec.modalities},
            "windows": {k: len(v) for k, v in self.datasets.items()},
            "subjects": {k: int(np.unique(
                self.store.anchors["subject"].to_numpy()[v.indices]).size)
                for k, v in self.datasets.items()},
            "strata": {k: int(v.sum()) for k, v in self.strata.items()},
        }


def window_spec_from_config(cfg: Config) -> WindowSpec:
    d = cfg.get_path("data")
    return WindowSpec(
        context=int(d.get("context", 288)),
        horizons_min=list(d.get("horizons_min", K.PRIMARY_HORIZONS_MIN)),
        modalities=list(d.get("modalities", ["cgm", "insulin", "carbs"])),
        use_event_features=bool(d.get("use_event_features", True)),
        use_masks=bool(d.get("use_masks", True)),
        use_static=bool(d.get("use_static", True)),
        use_source_identity=bool(d.get("use_source_identity", False)),
        use_time_features=bool(d.get("use_time_features", True)),
        missing_policy=str(d.get("missing_policy", "zero")),
        # Empty list = strictly causal = the benchmark. See constants.py.
        future_modalities=list(d.get("future_modalities", []) or []),
        future_steps=d.get("future_steps"),
    )


def _thin(mask: np.ndarray, stride: int) -> np.ndarray:
    """Keep every ``stride``-th anchor of a mask, deterministically."""
    if stride <= 1:
        return mask
    idx = np.flatnonzero(mask)
    out = np.zeros_like(mask)
    out[idx[::stride]] = True
    return out


def build_data(cfg: Config, standardizer: Optional[Standardizer] = None) -> DataBundle:
    """Open the store, make the split, fit (or reuse) normalisation, build datasets."""
    d = cfg.get_path("data")
    store = SequenceStore(d["store"])
    spec = window_spec_from_config(cfg)
    spec.validate()

    sp = dict(d.get("split", {}))
    scheme = sp.pop("scheme", "official")
    split = S.make_split(store, scheme, **sp)
    strata = S.test_strata(store, split)

    sampling = dict(d.get("sampling", {}))
    seed = int(sampling.get("seed", 0))
    subjects = store.anchors["subject"].to_numpy()

    train = _thin(split["train"], int(sampling.get("train_stride", 1)))
    train = S.subsample_mask(train, sampling.get("max_train_windows"), seed, subjects)
    val = S.subsample_mask(split["val"], sampling.get("max_val_windows"), seed + 1, subjects)
    test = S.subsample_mask(split["test"], sampling.get("max_test_windows"), seed + 2,
                            subjects)
    masks = {"train": train, "val": val, "test": test}

    if standardizer is None:
        standardizer = fit_standardizer(
            store, masks["train"], spec.all_channels(),
            store.horizon_ids(spec.horizons_min),
            target_mode=str(d.get("target_mode", "delta")),
            n_sample_anchors=int(sampling.get("standardizer_sample", 200_000)),
            seed=seed,
        )

    datasets = {name: WindowDataset(store, m, spec, standardizer)
                for name, m in masks.items()}
    # Test strata are evaluated from the same dataset object by index, so keep the
    # full (unsubsampled) masks around for evaluation.
    return DataBundle(store=store, spec=spec, standardizer=standardizer,
                      split=split, strata=strata, datasets=datasets)


def model_input_shapes(bundle: DataBundle) -> Dict[str, object]:
    """What the model needs to know about its inputs, and nothing more."""
    ds = bundle.datasets["train"]
    future_steps, future_channels = ds.future_shape
    return {
        "channel_counts": ds.channel_counts(),
        "n_time_features": ds.n_time_features,
        "n_static_numeric": ds.n_static_numeric if bundle.spec.use_static else 0,
        "static_cardinalities": list(ds.cat_cardinalities) if bundle.spec.use_static else [],
        "context": bundle.spec.context,
        "n_horizons": ds.n_horizons,
        # (0, 0) unless the run opted into planned future actions. The +1 channel
        # is the validity flag the dataset appends.
        "future_steps": future_steps,
        "future_channels": future_channels + 1 if future_channels else 0,
    }
