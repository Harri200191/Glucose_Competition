"""Data: corpus -> sequence store -> common windows -> dataloaders.

Imports are lazy on purpose. ``windows`` and ``dataset`` pull in torch, and the
dataset builder (``python -m metabonet_forecasting.data.prepare``) is a pure
numpy/pyarrow job that has no use for it -- importing torch there costs seconds
of start-up and spawns thread pools sized to the machine, which on a shared box
competes with the very numpy work the build is doing.

Everything is still reachable by name; it is just resolved on first use
(PEP 562).
"""
from __future__ import annotations

from typing import Any

from . import constants

_LAZY = {
    "SequenceStore": ".store",
    "open_store": ".store",
    "WindowDataset": ".windows",
    "WindowSpec": ".windows",
    "collate": ".windows",
    "batch_to_device": ".windows",
    "DataBundle": ".dataset",
    "build_data": ".dataset",
    "model_input_shapes": ".dataset",
    "window_spec_from_config": ".dataset",
    "Standardizer": ".transforms",
    "fit_standardizer": ".transforms",
    "splits": None,
    "prepare": None,
    "store": None,
    "windows": None,
    "dataset": None,
    "transforms": None,
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    module_name = _LAZY[name]
    if module_name is None:                       # a submodule was requested
        return importlib.import_module(f"{__name__}.{name}")
    module = importlib.import_module(module_name, __name__)
    return getattr(module, name)


def __dir__():
    return sorted(set(list(globals()) + list(_LAZY)))


__all__ = ["constants"] + [k for k in _LAZY]
