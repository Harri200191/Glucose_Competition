"""Models: one API, five architectures, four fusion strategies.

``build_model(cfg, shapes)`` is the only constructor training and evaluation call.
Importing this package registers every backbone and fusion in the registries.
"""
from __future__ import annotations

from typing import Dict

from ..common.config import Config
from ..common.registry import BACKBONES, FUSIONS
from .base import ForecastModel, ForecastOutput, TemporalBackbone
from . import tcn, patchtst, mamba2, recurrent          # noqa: F401  (registration)
from . import fusion                                     # noqa: F401  (registration)
from .frontend import FrontEndForecaster
from .tft import TFTForecaster
from .tide import TiDEForecaster

# Architectures that arrange their own internals instead of using the common
# front end, because their variable handling is the thing under test.
NATIVE_MODELS = {"tft": TFTForecaster, "tide": TiDEForecaster}


def build_model(cfg: Config, shapes: Dict) -> ForecastModel:
    model_cfg = dict(cfg.get_path("model"))
    name = str(model_cfg.get("backbone", {}).get("name", "tcn"))
    if name in NATIVE_MODELS:
        fusion_name = str(model_cfg.get("fusion", {}).get("name", "native"))
        if fusion_name not in ("native", "raw_concat"):
            raise ValueError(
                f"{name} is used in native form (plan section 6); it must not be given "
                f"the common fusion module (got fusion.name={fusion_name!r})")
        return NATIVE_MODELS[name](shapes, model_cfg)
    return FrontEndForecaster(shapes, model_cfg)


def available_backbones():
    return sorted(set(BACKBONES) | set(NATIVE_MODELS))


def available_fusions():
    return sorted(set(FUSIONS) | {"raw_concat"})


__all__ = ["build_model", "ForecastModel", "ForecastOutput", "TemporalBackbone",
           "FrontEndForecaster", "TFTForecaster", "TiDEForecaster",
           "available_backbones", "available_fusions", "BACKBONES", "FUSIONS"]
