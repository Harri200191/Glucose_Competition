"""Tree ensembles: XGBoost, LightGBM, CatBoost, RandomForest, ExtraTrees.

Notes that matter for this dataset
----------------------------------
* **NaN.** XGBoost / LightGBM / CatBoost / HistGB learn a default split direction
  for missing values, which is strictly better than imputing here: EDA 3.3 showed
  missingness is cohort-structured, so any imputed value is a fabricated
  cohort-specific constant. sklearn's RandomForest and ExtraTrees cannot do this,
  so they get median imputation + missingness indicators, and say so in `notes`.
* **Objective.** The metric is MARD = mean(|p-t|/t), so absolute error on the *log*
  scale is a closer surrogate than squared error on the raw scale. Each learner
  exposes `objective="mard"` which trains on log-glucose with an L1/Huber loss and
  exponentiates on predict. Squared error remains available for comparison.
* **Hardware.** V100s are compute capability 7.0; XGBoost >= 3.1 dropped that
  target, so this repo pins xgboost 2.1.4 where `device="cuda"` still works.
"""
from __future__ import annotations

import numpy as np

from .base import BaseModel, FitContext, NanImputer


# --------------------------------------------------------------------- helpers
class _LogTargetMixin:
    """Train on log(y); predict exp(.). Makes an L1 loss approximate MARD.

    d|log p - log t| / dp ~ 1/p, which is the same 1/t weighting MARD applies, so
    the learner spends its capacity where the metric does -- on the hypoglycaemic
    end (EDA 9.2).
    """
    log_target = False

    def _fwd(self, y):
        return np.log(np.clip(y, 1.0, None)) if self.log_target else y

    def _inv(self, p):
        return np.exp(np.clip(p, 0.0, 7.0)) if self.log_target else p


# --------------------------------------------------------------------- XGBoost
class XGBoostModel(_LogTargetMixin, BaseModel):
    name = "xgboost"
    handles_nan = True
    supports_early_stopping = True

    DEFAULTS = dict(n_estimators=3000, max_depth=8, learning_rate=0.03,
                    subsample=0.8, colsample_bytree=0.6, min_child_weight=32,
                    reg_lambda=2.0, reg_alpha=0.0, max_bin=256,
                    early_stopping_rounds=100)

    def __init__(self, objective: str = "mard", **params):
        super().__init__(**params)
        self.objective = objective
        self.log_target = objective == "mard"

    def fit(self, X, y, X_val=None, y_val=None, ctx: FitContext | None = None):
        import xgboost as xgb
        p = {**self.DEFAULTS, **self.params}
        es = p.pop("early_stopping_rounds", 100)
        obj = "reg:absoluteerror" if self.objective == "mard" else "reg:squarederror"
        self.model_ = xgb.XGBRegressor(
            objective=obj, tree_method="hist",
            device=f"cuda:{ctx.gpu}" if ctx and ctx.gpu is not None else "cpu",
            n_jobs=ctx.n_jobs if ctx else -1, random_state=ctx.seed if ctx else 0,
            early_stopping_rounds=es if X_val is not None else None,
            eval_metric="mae", **p)
        evalset = [(np.asarray(X_val), self._fwd(y_val))] if X_val is not None else None
        self.model_.fit(np.asarray(X), self._fwd(y), eval_set=evalset, verbose=False)
        self.notes = {"objective": obj, "log_target": self.log_target,
                      "best_iteration": int(getattr(self.model_, "best_iteration", -1)),
                      "nan_handling": "native default-direction splits"}
        return self

    def predict(self, X) -> np.ndarray:
        return self._inv(self.model_.predict(np.asarray(X))).astype(np.float32)

    def feature_importance(self):
        return self.model_.feature_importances_


# -------------------------------------------------------------------- LightGBM
class LightGBMModel(_LogTargetMixin, BaseModel):
    name = "lightgbm"
    handles_nan = True
    supports_early_stopping = True

    DEFAULTS = dict(n_estimators=2000, num_leaves=127, learning_rate=0.04,
                    min_child_samples=64, subsample=0.8, subsample_freq=1,
                    colsample_bytree=0.6, reg_lambda=2.0, max_bin=255,
                    force_row_wise=True)
    # LightGBM scales badly past ~16 threads on data this shape: the per-thread
    # histogram slices get small and synchronisation dominates. Observed on this
    # box: 92 threads on 107k rows took >12 min for a fit XGBoost did in 6 s.
    MAX_THREADS = 16

    def __init__(self, objective: str = "mard", n_threads: int | None = None,
                 **params):
        super().__init__(**params)
        self.objective = objective
        self.log_target = objective == "mard"
        self.n_threads = n_threads

    def fit(self, X, y, X_val=None, y_val=None, ctx: FitContext | None = None):
        import os
        import lightgbm as lgb
        p = {**self.DEFAULTS, **self.params}
        obj = "l1" if self.objective == "mard" else "l2"
        avail = os.cpu_count() or 8
        req = ctx.n_jobs if (ctx and ctx.n_jobs and ctx.n_jobs > 0) else avail
        nthr = self.n_threads or min(req, self.MAX_THREADS)
        self.model_ = lgb.LGBMRegressor(
            objective=obj, n_jobs=nthr,
            random_state=ctx.seed if ctx else 0, verbose=-1, **p)
        cb = []
        evalset = None
        if X_val is not None:
            evalset = [(np.asarray(X_val), self._fwd(y_val))]
            cb = [lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)]
        self.model_.fit(np.asarray(X), self._fwd(y), eval_set=evalset,
                        categorical_feature=(ctx.categorical_idx if ctx else "auto"),
                        callbacks=cb)
        self.notes = {"objective": obj, "log_target": self.log_target,
                      "best_iteration": int(self.model_.best_iteration_ or -1),
                      "n_threads": nthr, "nan_handling": "native"}
        return self

    def predict(self, X) -> np.ndarray:
        return self._inv(self.model_.predict(np.asarray(X))).astype(np.float32)

    def feature_importance(self):
        return self.model_.feature_importances_


# -------------------------------------------------------------------- CatBoost
class CatBoostModel(_LogTargetMixin, BaseModel):
    name = "catboost"
    handles_nan = True
    supports_early_stopping = True

    DEFAULTS = dict(iterations=5000, depth=8, learning_rate=0.05,
                    l2_leaf_reg=4.0, border_count=128, random_strength=1.0)

    def __init__(self, objective: str = "mard", use_gpu: bool = True, **params):
        super().__init__(**params)
        self.objective = objective
        self.log_target = objective == "mard"
        self.use_gpu = use_gpu

    def fit(self, X, y, X_val=None, y_val=None, ctx: FitContext | None = None):
        from catboost import CatBoostRegressor, Pool
        p = {**self.DEFAULTS, **self.params}
        loss = "MAE" if self.objective == "mard" else "RMSE"
        # CatBoost's GPU trainer does not implement MAE; fall back to CPU for it.
        task = "GPU" if (self.use_gpu and ctx and ctx.gpu is not None
                         and loss != "MAE") else "CPU"
        kw = dict(loss_function=loss, eval_metric="MAE", task_type=task,
                  random_seed=ctx.seed if ctx else 0, verbose=0,
                  allow_writing_files=False, **p)
        if task == "GPU":
            kw["devices"] = str(ctx.gpu)
        else:
            kw["thread_count"] = ctx.n_jobs if (ctx and ctx.n_jobs > 0) else -1
        self.model_ = CatBoostRegressor(**kw)

        # CatBoost needs categoricals as ints/strings without NaN.
        cat_idx = list(ctx.categorical_idx) if ctx else []
        Xtr = self._prep(np.asarray(X), cat_idx)
        train_pool = Pool(Xtr, self._fwd(y), cat_features=cat_idx)
        eval_pool = None
        if X_val is not None:
            eval_pool = Pool(self._prep(np.asarray(X_val), cat_idx),
                             self._fwd(y_val), cat_features=cat_idx)
        self.model_.fit(train_pool, eval_set=eval_pool,
                        early_stopping_rounds=200 if eval_pool else None)
        self.cat_idx_ = cat_idx
        self.notes = {"loss": loss, "task_type": task, "log_target": self.log_target,
                      "best_iteration": int(self.model_.get_best_iteration() or -1),
                      "nan_handling": "native for numeric; -1 sentinel for categoricals"}
        return self

    @staticmethod
    def _prep(X: np.ndarray, cat_idx: list[int]) -> np.ndarray:
        if not cat_idx:
            return X
        X = X.copy()
        for j in cat_idx:
            col = X[:, j]
            X[:, j] = np.where(np.isfinite(col), col, -1)
        return X

    def predict(self, X) -> np.ndarray:
        Xp = self._prep(np.asarray(X), self.cat_idx_)
        return self._inv(self.model_.predict(Xp)).astype(np.float32)

    def feature_importance(self):
        return self.model_.get_feature_importance()


# ---------------------------------------------------------------- RandomForest
class _SklearnForest(_LogTargetMixin, BaseModel):
    handles_nan = False
    _cls = None

    DEFAULTS = dict(n_estimators=300, max_depth=None, min_samples_leaf=20,
                    max_features=0.3, bootstrap=True)

    def __init__(self, objective: str = "mard", add_indicators: bool = True, **params):
        super().__init__(**params)
        self.objective = objective
        self.log_target = objective == "mard"
        self.imputer = NanImputer(add_indicators=add_indicators)

    def fit(self, X, y, X_val=None, y_val=None, ctx: FitContext | None = None):
        p = {**self.DEFAULTS, **self.params}
        # sklearn forests cannot consume NaN; impute on train statistics only.
        Xtr = self.imputer.fit_transform(np.asarray(X))
        # criterion stays "squared_error": sklearn's "absolute_error" re-sorts every
        # candidate split and is not tractable at millions of rows. The log-target
        # transform supplies the MARD-like 1/t weighting instead.
        self.model_ = self._cls(
            criterion="squared_error",
            n_jobs=ctx.n_jobs if ctx else -1, random_state=ctx.seed if ctx else 0,
            **p)
        self.model_.fit(Xtr, self._fwd(y))
        self.notes = {"log_target": self.log_target,
                      "criterion": "squared_error on log-target",
                      "imputation": self.imputer.describe(
                          ctx.feature_names if ctx else [])}
        return self

    def predict(self, X) -> np.ndarray:
        Xt = self.imputer.transform(np.asarray(X))
        return self._inv(self.model_.predict(Xt)).astype(np.float32)

    def feature_importance(self):
        return self.model_.feature_importances_


class RandomForestModel(_SklearnForest):
    name = "randomforest"

    def __init__(self, **kw):
        from sklearn.ensemble import RandomForestRegressor
        self._cls = RandomForestRegressor
        super().__init__(**kw)


class ExtraTreesModel(_SklearnForest):
    name = "extratrees"
    DEFAULTS = dict(n_estimators=400, max_depth=None, min_samples_leaf=20,
                    max_features=0.5, bootstrap=False)

    def __init__(self, **kw):
        from sklearn.ensemble import ExtraTreesRegressor
        self._cls = ExtraTreesRegressor
        super().__init__(**kw)


class HistGBModel(_LogTargetMixin, BaseModel):
    """sklearn's own histogram GBM -- a NaN-native, dependency-free control."""
    name = "histgb"
    handles_nan = True
    supports_early_stopping = True

    DEFAULTS = dict(max_iter=1500, max_leaf_nodes=63, learning_rate=0.05,
                    min_samples_leaf=64, l2_regularization=1.0,
                    early_stopping=True, n_iter_no_change=50, validation_fraction=0.1)

    def __init__(self, objective: str = "mard", **params):
        super().__init__(**params)
        self.objective = objective
        self.log_target = objective == "mard"

    def fit(self, X, y, X_val=None, y_val=None, ctx: FitContext | None = None):
        from sklearn.ensemble import HistGradientBoostingRegressor
        p = {**self.DEFAULTS, **self.params}
        loss = "absolute_error" if self.objective == "mard" else "squared_error"
        self.model_ = HistGradientBoostingRegressor(
            loss=loss, random_state=ctx.seed if ctx else 0,
            categorical_features=(ctx.categorical_idx if ctx else None), **p)
        self.model_.fit(np.asarray(X), self._fwd(y))
        self.notes = {"loss": loss, "log_target": self.log_target,
                      "n_iter": int(self.model_.n_iter_), "nan_handling": "native"}
        return self

    def predict(self, X) -> np.ndarray:
        return self._inv(self.model_.predict(np.asarray(X))).astype(np.float32)
