"""Feature construction.

Design constraint: the corpus is 154.8 M rows. A feature must be computed over the
*whole* array (so lags are time-correct) but only ~10 M rows are ever anchors. So
every feature is materialised once as a full-length vector, immediately sampled down
to the anchor rows, and freed. Peak extra memory is one full-length float64 vector
(~1.2 GB), not the whole feature matrix (~75 GB).

Every window crossing a segment boundary is invalidated to NaN. NaN is left in place
for models that handle it natively (XGBoost, LightGBM, CatBoost, HistGB, TabPFN);
`models/` imputes only where the learner demands it (RandomForest, neural nets), and
records that it did.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from .config import GRID_MIN, FeatureConfig
from .segments import segment_bounds

_MIN = float(GRID_MIN)


# --------------------------------------------------------------------- helpers
def blocked_lag(a: np.ndarray, k: int, seg_id: np.ndarray) -> np.ndarray:
    """a[i-k], NaN where the window leaves the segment."""
    n = len(a)
    out = np.full(n, np.nan, dtype=np.float64)
    if k < n:
        out[k:] = a[:-k]
        out[k:][seg_id[k:] != seg_id[:-k]] = np.nan
    return out


def blocked_lead(a: np.ndarray, k: int, seg_id: np.ndarray) -> np.ndarray:
    """a[i+k], NaN where the window leaves the segment. Used for targets only."""
    n = len(a)
    out = np.full(n, np.nan, dtype=np.float64)
    if k < n:
        out[:n - k] = a[k:]
        out[:n - k][seg_id[k:] != seg_id[:-k]] = np.nan
    return out


def blocked_rolling(a: np.ndarray, w: int, how: str, seg_id: np.ndarray,
                    min_frac: float = 0.5) -> np.ndarray:
    """Trailing rolling aggregate over `w` steps, invalidated across segments.

    `min_frac` lets a window tolerate CGM dropout: with 0.5, half the slots may be
    NaN and the statistic is still produced. Windows that do not contain enough
    real observations return NaN.
    """
    s = pd.Series(a, dtype="float64")
    r = s.rolling(w, min_periods=max(1, int(w * min_frac)))
    out = getattr(r, how)().to_numpy()
    # invalidate any window that reaches back past the start of the segment
    n = len(a)
    if w > 1:
        bad = np.ones(n, dtype=bool)
        bad[w - 1:] = seg_id[w - 1:] != seg_id[:n - w + 1]
        out[bad] = np.nan
    return out


def blocked_window_sum(a: np.ndarray, w: int, seg_id: np.ndarray) -> np.ndarray:
    """Trailing sum treating NaN as 0.

    For an event channel (bolus, carbs) a NaN means 'nothing recorded in this slot',
    and the neutral element of a sum is 0 -- see clean.py note 7. The window is
    still invalidated at segment edges so we never sum across a wear-period break.
    """
    filled = np.nan_to_num(a, nan=0.0)
    out = pd.Series(filled).rolling(w, min_periods=1).sum().to_numpy()
    n = len(a)
    if w > 1:
        bad = np.ones(n, dtype=bool)
        bad[w - 1:] = seg_id[w - 1:] != seg_id[:n - w + 1]
        out[bad] = np.nan
    return out


def exp_decay_accumulator(a: np.ndarray, halflife_min: float,
                          seg_id: np.ndarray) -> np.ndarray:
    """Insulin-on-board / carbs-on-board proxy.

    A first-order IIR filter  y[i] = alpha*y[i-1] + a[i]  with alpha set from the
    half-life, reset at every segment start. EDA 10.2 showed the carb response peaks
    45-75 min after the event and decays over 3-4 h, which is exactly what this
    kernel encodes -- and why the instantaneous column is nearly useless on its own.

    Run per segment with `scipy.signal.lfilter`: the closed-form cumulative version
    (alpha**i * cumsum(a / alpha**i)) overflows for i in the tens of thousands.
    """
    alpha = 0.5 ** (_MIN / halflife_min)
    x = np.nan_to_num(a, nan=0.0).astype(np.float64)
    out = np.empty_like(x)
    starts, ends = segment_bounds(seg_id)
    for s, e in zip(starts, ends):
        out[s:e + 1] = lfilter([1.0], [1.0, -alpha], x[s:e + 1])
    return out


def steps_since_event(event: np.ndarray, seg_id: np.ndarray) -> np.ndarray:
    """Minutes since the most recent True in `event`, NaN if none yet this segment."""
    n = len(event)
    idx = np.arange(n, dtype=np.int64)
    last = np.maximum.accumulate(np.where(event, idx, -1))
    starts, _ = segment_bounds(seg_id)
    seg_start = starts[seg_id - seg_id.min()]
    out = np.where(last >= seg_start, (idx - last).astype(np.float64) * _MIN, np.nan)
    return out


def value_at_last_event(values: np.ndarray, event: np.ndarray,
                        seg_id: np.ndarray) -> np.ndarray:
    """Magnitude of the most recent event, NaN if none yet this segment."""
    n = len(values)
    idx = np.arange(n, dtype=np.int64)
    last = np.maximum.accumulate(np.where(event, idx, -1))
    starts, _ = segment_bounds(seg_id)
    seg_start = starts[seg_id - seg_id.min()]
    safe = np.clip(last, 0, n - 1)
    return np.where(last >= seg_start, values[safe], np.nan)


# ------------------------------------------------------------------- the recipe
def feature_plan(fc: FeatureConfig, use_cohort: bool) -> list[tuple[str, str]]:
    """(name, group) for every feature, in the exact order they are built.

    Kept separate from the builder so the plan can be printed, diffed and
    documented without touching 154 M rows.
    """
    plan: list[tuple[str, str]] = []
    A = plan.append

    A(("cgm_now", "cgm_level"))
    for k in fc.cgm_lags:
        A((f"cgm_lag_{k}", "cgm_level"))
    for k in fc.cgm_diffs:
        A((f"cgm_diff_{k}", "cgm_dynamics"))
        A((f"cgm_rate_{k}", "cgm_dynamics"))
    A(("cgm_accel", "cgm_dynamics"))
    for w in fc.windows:
        for how in ("mean", "std", "min", "max"):
            A((f"cgm_{how}_{w}", "cgm_shape"))
        A((f"cgm_range_{w}", "cgm_shape"))
    for hl in fc.ewm_halflives:
        A((f"cgm_ewm_{hl}", "cgm_shape"))
    A(("cgm_tir_frac_288", "cgm_shape"))
    A(("cgm_hypo_frac_288", "cgm_shape"))
    A(("cgm_hyper_frac_288", "cgm_shape"))

    for ch in fc.dose_channels:
        for w in fc.windows:
            A((f"{ch}_sum_{w}", "dose"))
    A(("iob", "dose"))
    A(("cob", "dose"))
    A(("basal_ob", "dose"))
    A(("min_since_bolus", "dose"))
    A(("min_since_carb", "dose"))
    A(("last_bolus_units", "dose"))
    A(("last_carb_grams", "dose"))
    A(("n_carb_events_288", "dose"))
    A(("n_bolus_events_288", "dose"))

    if fc.use_missingness_features:
        A(("cgm_missing_frac_12", "sensor_health"))
        A(("cgm_missing_frac_288", "sensor_health"))
        A(("min_since_real_cgm", "sensor_health"))
        A(("max_cgm_gap_288", "sensor_health"))

    A(("tod_sin", "calendar"))
    A(("tod_cos", "calendar"))
    A(("dow_sin", "calendar"))
    A(("dow_cos", "calendar"))
    A(("is_weekend", "calendar"))
    A(("hours_into_segment", "calendar"))

    for c in ("age", "age_of_diagnosis", "height_m", "weight_kg", "bmi",
              "diabetes_duration"):
        A((c, "static_numeric"))
    A(("gender", "static_categorical"))
    A(("insulin_delivery_modality", "static_categorical"))
    if use_cohort:
        A(("source_file", "static_categorical"))
    return plan


CATEGORICAL_FEATURES = ("gender", "insulin_delivery_modality", "source_file")


def build_feature_matrix(df: pd.DataFrame, seg_id: np.ndarray, cgm_f: np.ndarray,
                         was_interp: np.ndarray, anchor_idx: np.ndarray,
                         fc: FeatureConfig) -> tuple[np.ndarray, list[str], dict]:
    """Materialise the feature matrix at `anchor_idx`.

    `cgm_f` is the short-gap-interpolated CGM from `segments.interpolate_short_gaps`;
    the raw column is used only for the sensor-health features and the target.
    """
    plan = feature_plan(fc, fc.use_cohort_feature)
    names = [n for n, _ in plan]
    n_out, n_feat = len(anchor_idx), len(plan)
    X = np.full((n_out, n_feat), np.nan, dtype=np.float32)
    col = {n: j for j, n in enumerate(names)}

    def put(name: str, full: np.ndarray) -> None:
        X[:, col[name]] = full[anchor_idx]

    cgm_raw = df["CGM"].to_numpy(dtype=np.float64)
    bolus = df["bolus"].to_numpy(dtype=np.float64)
    basal = df["basal"].to_numpy(dtype=np.float64)
    carbs = df["carbs"].to_numpy(dtype=np.float64)

    # ---- CGM level ---------------------------------------------------------
    put("cgm_now", cgm_f.astype(np.float64))
    lags = {}
    for k in fc.cgm_lags:
        v = blocked_lag(cgm_f, k, seg_id)
        put(f"cgm_lag_{k}", v)
        if k in fc.cgm_diffs or k in (1, 2):
            lags[k] = v
        else:
            del v

    # ---- CGM dynamics ------------------------------------------------------
    for k in fc.cgm_diffs:
        d = cgm_f - lags[k]
        put(f"cgm_diff_{k}", d)
        put(f"cgm_rate_{k}", d / (k * _MIN))
        if k not in (1, 2):
            del d
    put("cgm_accel", (cgm_f - lags[1]) - (lags[1] - lags[2]))
    del lags

    # ---- CGM shape ---------------------------------------------------------
    for w in fc.windows:
        mn = blocked_rolling(cgm_f, w, "min", seg_id)
        mx = blocked_rolling(cgm_f, w, "max", seg_id)
        put(f"cgm_mean_{w}", blocked_rolling(cgm_f, w, "mean", seg_id))
        put(f"cgm_std_{w}", blocked_rolling(cgm_f, w, "std", seg_id))
        put(f"cgm_min_{w}", mn)
        put(f"cgm_max_{w}", mx)
        put(f"cgm_range_{w}", mx - mn)
        del mn, mx
    for hl in fc.ewm_halflives:
        # The EWM runs over the whole array and is only hard-reset on the first row
        # of each segment. Rows just after a boundary therefore retain a little
        # weight from the previous segment -- but with half-lives of 3-12 steps that
        # decays to nothing within ~50 rows, and an anchor is required to have 288
        # rows of in-segment history (see build_dataset), so **no anchor is ever
        # affected**. A per-segment loop would be exact and 1,909x slower for no
        # change to any value that is actually used.
        e = pd.Series(cgm_f, dtype="float64").ewm(halflife=hl, ignore_na=True).mean().to_numpy()
        bnd = np.flatnonzero(np.concatenate(([True], seg_id[1:] != seg_id[:-1])))
        e[bnd] = cgm_f[bnd]
        put(f"cgm_ewm_{hl}", e)
        del e

    w24 = 288
    in_range = ((cgm_f >= 70) & (cgm_f <= 180)).astype(np.float64)
    in_range[~np.isfinite(cgm_f)] = np.nan
    put("cgm_tir_frac_288", blocked_rolling(in_range, w24, "mean", seg_id))
    del in_range
    hypo = (cgm_f < 70).astype(np.float64); hypo[~np.isfinite(cgm_f)] = np.nan
    put("cgm_hypo_frac_288", blocked_rolling(hypo, w24, "mean", seg_id))
    del hypo
    hyper = (cgm_f > 180).astype(np.float64); hyper[~np.isfinite(cgm_f)] = np.nan
    put("cgm_hyper_frac_288", blocked_rolling(hyper, w24, "mean", seg_id))
    del hyper

    # ---- dose / carbohydrate ----------------------------------------------
    channels = {"bolus": bolus, "basal": basal, "carbs": carbs}
    for ch in fc.dose_channels:
        arr = channels[ch]
        for w in fc.windows:
            put(f"{ch}_sum_{w}", blocked_window_sum(arr, w, seg_id))

    put("iob", exp_decay_accumulator(bolus, fc.iob_halflife_min, seg_id))
    put("cob", exp_decay_accumulator(carbs, fc.cob_halflife_min, seg_id))
    put("basal_ob", exp_decay_accumulator(basal, fc.iob_halflife_min, seg_id))

    bolus_evt = np.nan_to_num(bolus, nan=0.0) > 0
    carb_evt = np.nan_to_num(carbs, nan=0.0) > 0
    put("min_since_bolus", steps_since_event(bolus_evt, seg_id))
    put("min_since_carb", steps_since_event(carb_evt, seg_id))
    put("last_bolus_units", value_at_last_event(np.nan_to_num(bolus), bolus_evt, seg_id))
    put("last_carb_grams", value_at_last_event(np.nan_to_num(carbs), carb_evt, seg_id))
    put("n_carb_events_288", blocked_window_sum(carb_evt.astype(np.float64), w24, seg_id))
    put("n_bolus_events_288", blocked_window_sum(bolus_evt.astype(np.float64), w24, seg_id))
    del bolus_evt, carb_evt

    # ---- sensor health -----------------------------------------------------
    real = np.isfinite(cgm_raw)
    if fc.use_missingness_features:
        miss = (~real).astype(np.float64)
        put("cgm_missing_frac_12", blocked_window_sum(miss, 12, seg_id) / 12.0)
        put("cgm_missing_frac_288", blocked_window_sum(miss, w24, seg_id) / w24)
        del miss
        put("min_since_real_cgm", steps_since_event(real, seg_id))
        gap_len = df["_gap_len"].to_numpy(dtype=np.float64)
        put("max_cgm_gap_288", blocked_rolling(gap_len, w24, "max", seg_id, min_frac=1.0)
            * _MIN)

    # ---- calendar ----------------------------------------------------------
    minute_of_day = (df["date"].dt.hour.to_numpy() * 60
                     + df["date"].dt.minute.to_numpy()).astype(np.float64)
    ang = 2 * np.pi * minute_of_day / 1440.0
    put("tod_sin", np.sin(ang)); put("tod_cos", np.cos(ang))
    dow = df["date"].dt.dayofweek.to_numpy().astype(np.float64)
    dang = 2 * np.pi * dow / 7.0
    put("dow_sin", np.sin(dang)); put("dow_cos", np.cos(dang))
    put("is_weekend", (dow >= 5).astype(np.float64))
    starts, _ = segment_bounds(seg_id)
    seg_start = starts[seg_id - seg_id.min()]
    put("hours_into_segment",
        (np.arange(len(seg_id), dtype=np.float64) - seg_start) * _MIN / 60.0)
    del minute_of_day, ang, dow, dang

    # ---- static ------------------------------------------------------------
    for c in ("age", "age_of_diagnosis", "height_m", "weight_kg", "bmi",
              "diabetes_duration"):
        put(c, df[c].to_numpy(dtype=np.float64))

    # Categoricals are returned as integer codes plus their level tables, so each
    # model adapter can choose one-hot / native-categorical / target encoding.
    cat_levels: dict[str, list] = {}
    cat_cols = ["gender", "insulin_delivery_modality"]
    if fc.use_cohort_feature:
        cat_cols.append("source_file")
    for c in cat_cols:
        codes, levels = pd.factorize(df[c], use_na_sentinel=True)
        X[:, col[c]] = codes[anchor_idx].astype(np.float32)
        cat_levels[c] = list(levels)

    meta = {"categorical_levels": cat_levels,
            "categorical_columns": cat_cols,
            "groups": {n: g for n, g in plan}}
    return X, names, meta
