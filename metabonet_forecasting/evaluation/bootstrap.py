"""Patient-level resampling (plan 8.3).

Millions of overlapping windows are not millions of independent samples: adjacent
anchors share 23 h 55 min of history and their errors are strongly correlated.
Resampling **patients**, not windows, is what makes a confidence interval mean
something here.

Two procedures:

``bootstrap_metric``   subject-level bootstrap CI for one model's metric
``paired_bootstrap``   the distribution of the *paired* difference between two
                       models, resampling the same subjects for both. Paired is
                       essential: between-patient variance dwarfs the difference
                       between two good models, so an unpaired comparison would
                       show overlapping intervals for a difference that is
                       consistent in every single patient.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .metrics import mard, mae, rmse

METRIC_FUNCS: Dict[str, Callable] = {"rmse": rmse, "mae": mae, "mard": mard}


def _subject_groups(subjects: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
    order = np.argsort(subjects, kind="stable")
    s = subjects[order]
    bounds = np.flatnonzero(np.diff(s)) + 1
    groups = np.split(order, bounds)
    uniq = s[np.concatenate([[0], bounds])] if s.size else np.array([])
    return uniq, groups


def bootstrap_metric(y_true: np.ndarray, y_pred: np.ndarray, subjects: np.ndarray,
                     metric: str = "rmse", n_boot: int = 1000, seed: int = 0,
                     alpha: float = 0.05) -> Dict[str, float]:
    """Percentile bootstrap CI over subjects for one metric."""
    fn = METRIC_FUNCS[metric]
    valid = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0)
    y_true, y_pred, subjects = y_true[valid], y_pred[valid], subjects[valid]
    if y_true.size == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}

    uniq, groups = _subject_groups(subjects)
    rng = np.random.default_rng(seed)
    point = float(fn(y_true, y_pred))
    draws = np.empty(n_boot)
    n_groups = len(groups)
    for b in range(n_boot):
        pick = rng.integers(0, n_groups, size=n_groups)
        idx = np.concatenate([groups[i] for i in pick])
        draws[b] = fn(y_true[idx], y_pred[idx])
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return {"point": point, "lo": float(lo), "hi": float(hi),
            "se": float(draws.std(ddof=1)), "n": int(y_true.size),
            "n_subjects": int(n_groups)}


def paired_bootstrap(df_a: pd.DataFrame, df_b: pd.DataFrame, horizon: int,
                     metric: str = "rmse", n_boot: int = 1000, seed: int = 0,
                     alpha: float = 0.05) -> Dict[str, float]:
    """Distribution of metric(A) - metric(B) over the anchors both models scored.

    The join is on the anchor id, so the comparison is on identical windows with
    identical labels. A negative difference means A is better for rmse/mae/mard.
    """
    key = "anchor"
    cols = [key, "subject", f"y_true_{horizon}", f"y_pred_{horizon}"]
    a = df_a[cols].rename(columns={f"y_pred_{horizon}": "pred_a"})
    b = df_b[[key, f"y_pred_{horizon}"]].rename(columns={f"y_pred_{horizon}": "pred_b"})
    m = a.merge(b, on=key, how="inner")
    y = m[f"y_true_{horizon}"].to_numpy(dtype=np.float64)
    pa = m["pred_a"].to_numpy(dtype=np.float64)
    pb = m["pred_b"].to_numpy(dtype=np.float64)
    subjects = m["subject"].to_numpy()
    valid = np.isfinite(y) & np.isfinite(pa) & np.isfinite(pb) & (y > 0)
    y, pa, pb, subjects = y[valid], pa[valid], pb[valid], subjects[valid]
    if y.size == 0:
        return {"n": 0}

    fn = METRIC_FUNCS[metric]
    uniq, groups = _subject_groups(subjects)
    rng = np.random.default_rng(seed)
    point = float(fn(y, pa) - fn(y, pb))
    draws = np.empty(n_boot)
    n_groups = len(groups)
    for i in range(n_boot):
        pick = rng.integers(0, n_groups, size=n_groups)
        idx = np.concatenate([groups[j] for j in pick])
        draws[i] = fn(y[idx], pa[idx]) - fn(y[idx], pb[idx])
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    # Two-sided bootstrap p-value for "no difference", by the proportion of
    # resamples on the other side of zero. Reported, not thresholded.
    p_value = 2 * min((draws <= 0).mean(), (draws >= 0).mean())
    return {
        "metric": metric, "horizon": horizon, "n": int(y.size),
        "n_subjects": int(n_groups),
        "difference": point, "lo": float(lo), "hi": float(hi),
        "p_value": float(min(p_value, 1.0)),
        "favours": ("A" if point < 0 else "B") if metric in ("rmse", "mae", "mard")
                   else ("B" if point < 0 else "A"),
        "subjects_A_better": float(np.mean([
            fn(y[g], pa[g]) < fn(y[g], pb[g]) for g in groups]) * 100),
    }


def seed_variability(values: Sequence[float]) -> Dict[str, float]:
    """Mean and spread across training seeds, reported next to every neural result."""
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return {}
    return {"seeds": int(v.size), "mean": float(v.mean()),
            "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
            "min": float(v.min()), "max": float(v.max())}
