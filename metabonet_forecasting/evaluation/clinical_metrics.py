"""Clinical error analysis (plan 8.2).

Aggregate RMSE hides the cases that matter. The strata here are the ones the plan
names, and they are defined on quantities available **at forecast time** or on the
true future value, never on the prediction:

reference glycaemia   hypo L2 (<54), hypo L1 (54-70), in range (70-180),
                      hyper L1 (180-250), hyper L2 (>250) -- by the *true* value
                      at t+h, because that is the clinical situation the forecast
                      was supposed to warn about
event context         postprandial (a carbohydrate entry within 120 min before the
                      anchor), post-correction (a bolus within 120 min), quiet
                      (neither) -- from the anchor's own recency channels
dynamics              falling / steady / rising, from the 30-minute CGM change at
                      the anchor: a forecast that is only good when glucose is flat
                      is not useful

Absence of a logged event is not proof that no event occurred: carbohydrate and
activity logs are user-reported and sparse. The "quiet" stratum is therefore
"nothing was logged", not "nothing happened".
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..data import constants as K
from .metrics import point_metrics, per_subject_metrics

POSTPRANDIAL_MIN = 120.0
POST_BOLUS_MIN = 120.0
TREND_THRESHOLD = 10.0        # mg/dL over 30 min ~ 0.33 mg/dL/min


def glycemic_strata(y_true: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "hypo_L2_lt54": y_true < K.HYPO_L2,
        "hypo_L1_54_70": (y_true >= K.HYPO_L2) & (y_true < K.HYPO_L1),
        "in_range_70_180": (y_true >= K.HYPO_L1) & (y_true <= K.RANGE_HI),
        "hyper_L1_180_250": (y_true > K.RANGE_HI) & (y_true <= K.HYPER_L2),
        "hyper_L2_gt250": y_true > K.HYPER_L2,
    }


def event_strata(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    carb = df["min_since_carb"].to_numpy()
    bolus = df["min_since_bolus"].to_numpy()
    post_meal = carb <= POSTPRANDIAL_MIN
    post_bolus = bolus <= POST_BOLUS_MIN
    return {
        "postprandial": post_meal,
        "post_correction_bolus": post_bolus,
        "meal_and_bolus": post_meal & post_bolus,
        "quiet_no_logged_event": ~post_meal & ~post_bolus,
    }


def trend_strata(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    d = df["cgm_delta_30min"].to_numpy()
    return {
        "falling": d < -TREND_THRESHOLD,
        "steady": np.abs(d) <= TREND_THRESHOLD,
        "rising": d > TREND_THRESHOLD,
    }


def evaluate_horizon(df: pd.DataFrame, horizon: int, with_grids: bool = True,
                     min_n: int = 50) -> Dict[str, Dict[str, float]]:
    """Every stratum for one horizon, from a predictions frame."""
    y = df[f"y_true_{horizon}"].to_numpy(dtype=np.float64)
    p = df[f"y_pred_{horizon}"].to_numpy(dtype=np.float64)
    valid = np.isfinite(y) & np.isfinite(p)

    out: Dict[str, Dict[str, float]] = {"overall": point_metrics(y, p, grids=with_grids)}
    out["overall"].update(per_subject_metrics(y, p, df["subject"].to_numpy()))

    groups: List[Tuple[str, Dict[str, np.ndarray]]] = [
        ("glycemia", glycemic_strata(y)),
        ("event", event_strata(df)),
        ("trend", trend_strata(df)),
    ]
    for group, strata in groups:
        for name, mask in strata.items():
            m = mask & valid
            if m.sum() < min_n:
                out[f"{group}:{name}"] = {"n": int(m.sum())}
                continue
            out[f"{group}:{name}"] = point_metrics(y[m], p[m], grids=with_grids)
    return out


def evaluate_by_source(df: pd.DataFrame, horizon: int, cohort_names: List[str],
                       min_n: int = 100) -> Dict[str, Dict[str, float]]:
    """Per source dataset, for the cohort-shortcut and transportability analyses."""
    y = df[f"y_true_{horizon}"].to_numpy(dtype=np.float64)
    p = df[f"y_pred_{horizon}"].to_numpy(dtype=np.float64)
    cohort = df["cohort"].to_numpy()
    out: Dict[str, Dict[str, float]] = {}
    for code in np.unique(cohort):
        name = cohort_names[int(code)] if int(code) < len(cohort_names) else str(code)
        m = (cohort == code) & np.isfinite(y) & np.isfinite(p)
        out[name] = point_metrics(y[m], p[m], grids=False) if m.sum() >= min_n \
            else {"n": int(m.sum())}
    return out


def summarise(metrics: Dict[str, Dict[str, Dict[str, float]]],
              keys: Tuple[str, ...] = ("rmse", "mae", "mard", "dts_A")) -> pd.DataFrame:
    """Flatten the nested metric dict into a tidy frame for tables and plots."""
    rows = []
    for horizon, strata in metrics.items():
        for stratum, values in strata.items():
            row = {"horizon": horizon, "stratum": stratum, "n": values.get("n", 0)}
            row.update({k: values.get(k) for k in keys})
            rows.append(row)
    return pd.DataFrame(rows)
