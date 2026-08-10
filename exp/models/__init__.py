"""Model registry.

`build_model(name, **overrides)` is the only entry point `train.py` uses. Names are
what you pass to `--model` on the command line.
"""
from __future__ import annotations

from .base import BaseModel, FitContext, LinearTrendModel, PersistenceModel

_LAZY = {
    # ---- baselines (no dependencies, run these first) ----------------------
    "persistence":    ("exp.models.base", "PersistenceModel", {}),
    "linear_trend":   ("exp.models.base", "LinearTrendModel", {}),

    # ---- gradient-boosted trees --------------------------------------------
    "xgboost":        ("exp.models.trees", "XGBoostModel", {}),
    "xgboost_l2":     ("exp.models.trees", "XGBoostModel", {"objective": "l2"}),
    "lightgbm":       ("exp.models.trees", "LightGBMModel", {}),
    "catboost":       ("exp.models.trees", "CatBoostModel", {}),
    "catboost_gpu":   ("exp.models.trees", "CatBoostModel", {"objective": "l2"}),
    "histgb":         ("exp.models.trees", "HistGBModel", {}),

    # ---- bagged forests -----------------------------------------------------
    "randomforest":   ("exp.models.trees", "RandomForestModel", {}),
    "extratrees":     ("exp.models.trees", "ExtraTreesModel", {}),

    # ---- prior-fitted transformer ------------------------------------------
    "tabpfn":         ("exp.models.tabpfn_model", "TabPFNModel", {}),

    # ---- neural networks ----------------------------------------------------
    "mlp":            ("exp.models.neural", "NeuralModel", {"arch": "mlp"}),
    "resmlp":         ("exp.models.neural", "NeuralModel", {"arch": "resmlp"}),
    "ft_transformer": ("exp.models.neural", "NeuralModel",
                       {"arch": "ft_transformer", "batch_size": 2048, "lr": 1e-3}),
    "tcn":            ("exp.models.neural", "NeuralModel",
                       {"arch": "tcn", "batch_size": 2048}),
    "gru":            ("exp.models.neural", "NeuralModel",
                       {"arch": "gru", "batch_size": 2048}),
}

MODEL_NAMES = list(_LAZY)
SEQUENCE_MODELS = {"tcn", "gru"}
NEURAL_MODELS = {"mlp", "resmlp", "ft_transformer", "tcn", "gru"}

# Model families, for the report tables.
FAMILY = {
    "persistence": "baseline", "linear_trend": "baseline",
    "xgboost": "gbdt", "xgboost_l2": "gbdt", "lightgbm": "gbdt",
    "catboost": "gbdt", "catboost_gpu": "gbdt", "histgb": "gbdt",
    "randomforest": "forest", "extratrees": "forest",
    "tabpfn": "pfn",
    "mlp": "nn", "resmlp": "nn", "ft_transformer": "nn", "tcn": "nn", "gru": "nn",
}


def build_model(name: str, **overrides) -> BaseModel:
    if name not in _LAZY:
        raise KeyError(f"unknown model {name!r}. Available: {', '.join(MODEL_NAMES)}")
    module_path, cls_name, defaults = _LAZY[name]
    import importlib
    cls = getattr(importlib.import_module(module_path), cls_name)
    model = cls(**{**defaults, **overrides})
    model.name = name
    return model


__all__ = ["build_model", "MODEL_NAMES", "SEQUENCE_MODELS", "NEURAL_MODELS",
           "FAMILY", "BaseModel", "FitContext", "PersistenceModel",
           "LinearTrendModel"]
