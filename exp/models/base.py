"""Common model interface.

Every learner in this package is wrapped so `train.py` can treat them
identically. The wrapper also records *what it had to do to the data* -- most
importantly whether it imputed NaN -- because that is a modelling decision, not a
data-cleaning one, and it belongs in the run manifest rather than buried in a
pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import numpy as np


@dataclass
class FitContext:
    """Everything a model may need beyond X and y."""
    feature_names: list[str]
    categorical_idx: list[int] = field(default_factory=list)
    categorical_levels: dict[str, list] = field(default_factory=dict)
    horizon_min: int = 30
    seed: int = 0
    gpu: int | None = 0
    n_jobs: int = -1


class BaseModel:
    """Contract: fit(X, y, ...) then predict(X) -> np.ndarray[float32]."""

    name: str = "base"
    handles_nan: bool = False        # can the learner consume NaN directly?
    wants_scaling: bool = False      # does it need standardised inputs?
    supports_early_stopping: bool = False

    def __init__(self, **params):
        self.params = params
        self.notes: dict = {}        # goes into the run manifest

    # ------------------------------------------------------------------ api
    def fit(self, X, y, X_val=None, y_val=None, ctx: FitContext | None = None):
        raise NotImplementedError

    def predict(self, X) -> np.ndarray:
        raise NotImplementedError

    def feature_importance(self) -> np.ndarray | None:
        return None

    def save(self, path: Path) -> None:
        import joblib
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "BaseModel":
        import joblib
        return joblib.load(path)


class NanImputer:
    """Median imputation + optional missingness indicators.

    Used only by learners that cannot take NaN (RandomForest, ExtraTrees, the
    neural nets). Statistics are fitted on **train only** -- fitting them on the
    concatenation of train and test leaks the test marginal, which matters here
    because EDA 3.3 showed missingness is cohort-structured rather than random.
    """

    def __init__(self, add_indicators: bool = True, max_indicator_cols: int = 40):
        self.add_indicators = add_indicators
        self.max_indicator_cols = max_indicator_cols
        self.medians_: np.ndarray | None = None
        self.indicator_cols_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "NanImputer":
        self.medians_ = np.nanmedian(X, axis=0)
        self.medians_ = np.where(np.isfinite(self.medians_), self.medians_, 0.0)
        if self.add_indicators:
            frac = np.isnan(X).mean(axis=0)
            # only for columns where missingness is actually informative
            cand = np.flatnonzero((frac > 0.005) & (frac < 0.995))
            order = cand[np.argsort(-frac[cand])]
            self.indicator_cols_ = np.sort(order[:self.max_indicator_cols])
        else:
            self.indicator_cols_ = np.array([], dtype=int)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        M = np.isnan(X)
        out = np.where(M, self.medians_, X).astype(np.float32)
        if len(self.indicator_cols_):
            ind = M[:, self.indicator_cols_].astype(np.float32)
            out = np.hstack([out, ind])
        return out

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def describe(self, feature_names: list[str]) -> dict:
        return {
            "strategy": "median (fitted on train only)",
            "indicator_columns": [feature_names[i] for i in self.indicator_cols_],
            "n_indicators": int(len(self.indicator_cols_)),
        }


class Standardiser:
    """z-scoring for the neural nets. Robust: uses median / IQR, because EDA 5.2
    showed several inputs have skew in the tens to hundreds, where mean/sd is
    dominated by the tail."""

    def __init__(self, clip: float = 8.0):
        self.clip = clip
        self.center_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "Standardiser":
        self.center_ = np.nanmedian(X, axis=0)
        q1 = np.nanpercentile(X, 25, axis=0)
        q3 = np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        # fall back to sd where the IQR is degenerate (binary / near-constant cols)
        sd = np.nanstd(X, axis=0)
        scale = np.where(iqr > 1e-6, iqr / 1.349, np.where(sd > 1e-6, sd, 1.0))
        self.center_ = np.nan_to_num(self.center_)
        self.scale_ = np.nan_to_num(scale, nan=1.0)
        self.scale_[self.scale_ <= 1e-6] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.center_) / self.scale_
        return np.clip(Z, -self.clip, self.clip).astype(np.float32)


# --------------------------------------------------------------------- baselines
class PersistenceModel(BaseModel):
    """Predict 'glucose stays where it is'.

    Not a throwaway: EDA 9.6 measured this at MARD 14.03 / 22.71 / 28.41 / 32.32 %
    for the four horizons across the whole corpus. It is the number every other
    model in this repo has to beat, and it uses no features at all.
    """
    name = "persistence"
    handles_nan = True

    def __init__(self, **params):
        super().__init__(**params)
        self.col_ = None

    def fit(self, X, y, X_val=None, y_val=None, ctx: FitContext | None = None):
        self.col_ = ctx.feature_names.index("cgm_now")
        return self

    def predict(self, X) -> np.ndarray:
        return np.asarray(X[:, self.col_], dtype=np.float32)


class LinearTrendModel(BaseModel):
    """Extrapolate the last 15 minutes of slope. EDA 9.6 found this *worse* than
    persistence at every horizon -- glucose mean-reverts -- so it is here as a
    control that a real model should also beat."""
    name = "linear_trend"
    handles_nan = True

    def __init__(self, **params):
        super().__init__(**params)

    def fit(self, X, y, X_val=None, y_val=None, ctx: FitContext | None = None):
        self.now_ = ctx.feature_names.index("cgm_now")
        self.rate_ = ctx.feature_names.index("cgm_rate_3")
        self.h_ = ctx.horizon_min
        return self

    def predict(self, X) -> np.ndarray:
        now = np.asarray(X[:, self.now_], dtype=np.float64)
        rate = np.nan_to_num(np.asarray(X[:, self.rate_], dtype=np.float64))
        return (now + rate * self.h_).astype(np.float32)
