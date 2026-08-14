"""Reader for the sequence store built by ``data/prepare.py``.

The store is deliberately dumb: a memmapped float32 lattice plus index tables.
All semantics -- which channels belong to which modality, which anchors belong to
which split, how values are normalised -- are applied on top, so a single build
serves every experiment in the plan.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import constants as K


class SequenceStore:
    """Memmapped access to one prepared dataset version."""

    def __init__(self, root) -> None:
        self.root = Path(root)
        if not (self.root / "manifest.json").exists():
            raise FileNotFoundError(
                f"{self.root} is not a sequence store (no manifest.json). "
                f"Build one with: python -m metabonet_forecasting.data.prepare --out {self.root}")
        with open(self.root / "manifest.json") as fh:
            self.manifest: Dict = json.load(fh)
        if self.manifest.get("schema_version") != K.SCHEMA_VERSION:
            raise RuntimeError(
                f"store schema {self.manifest.get('schema_version')} != code schema "
                f"{K.SCHEMA_VERSION}; rebuild the store")

        self.channels = np.load(self.root / "channels.npy", mmap_mode="r")
        self.targets = np.load(self.root / "targets.npy", mmap_mode="r")
        self.statics = np.load(self.root / "statics.npy")
        self.anchors = pd.read_parquet(self.root / "anchors.parquet")
        self.segments = pd.read_parquet(self.root / "segments.parquet")
        self.subjects = pd.read_parquet(self.root / "subjects.parquet")

        self.channel_names: List[str] = list(self.manifest["channels"])
        self.channel_index = {c: i for i, c in enumerate(self.channel_names)}
        self.horizons: List[int] = list(self.manifest["horizons_min"])
        self.grid_min: int = int(self.manifest["grid_min"])
        self.min_history_steps: int = int(self.manifest["config"]["min_history_steps"])
        self.static_names: List[str] = (list(self.manifest["static_numeric"])
                                        + list(self.manifest["static_categorical"]))

    # ------------------------------------------------------------------ lookups
    def channel_ids(self, names: Sequence[str]) -> np.ndarray:
        missing = [n for n in names if n not in self.channel_index]
        if missing:
            raise KeyError(f"channels {missing} not in store; have {self.channel_names}")
        return np.array([self.channel_index[n] for n in names], dtype=np.int64)

    def horizon_ids(self, horizons_min: Sequence[int]) -> np.ndarray:
        missing = [h for h in horizons_min if h not in self.horizons]
        if missing:
            raise KeyError(f"horizons {missing} not in store; have {self.horizons}")
        return np.array([self.horizons.index(h) for h in horizons_min], dtype=np.int64)

    def level_names(self, field: str) -> List[str]:
        return list(self.manifest["levels"].get(field, []))

    def n_levels(self, field: str) -> int:
        return len(self.level_names(field))

    def cohort_names(self) -> List[str]:
        return self.level_names("source_file")

    def cohort_code(self, name: str) -> int:
        names = self.cohort_names()
        if name not in names:
            raise KeyError(f"unknown cohort {name!r}; have {names}")
        return names.index(name)

    # ------------------------------------------------------------------- gather
    def window(self, row: int, context: int, channel_ids: np.ndarray) -> np.ndarray:
        """The causal window ending at (and including) ``row``: [context, n_ch]."""
        if row - context + 1 < 0:
            raise IndexError(f"window of {context} steps runs off the store at row {row}")
        return np.asarray(self.channels[row - context + 1: row + 1][:, channel_ids])

    def anchor_subjects(self) -> np.ndarray:
        return self.anchors["subject"].to_numpy()

    def describe(self) -> str:
        s = self.manifest["stats"]
        return (f"{self.root.name}: {self.manifest['n_rows']:,} rows, "
                f"{self.manifest['n_anchors']:,} anchors, "
                f"{self.manifest['n_subjects']:,} subjects, "
                f"{s.get('n_segments', 0):,} segments, "
                f"{len(self.channel_names)} channels")


def open_store(root: Optional[str] = None) -> SequenceStore:
    return SequenceStore(root or (K.ARTIFACT_DIR / "store" / "main"))
