"""Predictive metrics and error grids (plan 8.1).

RMSE is primary, MAE and MARD accompany it, and every number is reported **per
horizon** rather than averaged across horizons: a model that is excellent at
30 minutes and useless at 120 has a respectable average and no clinical value.

Error grids
-----------
* **DTS** (Klonoff et al. 2024) -- the grid the public MetaboNet leaderboard uses.
  The borders below are straight lines transcribed from the organisers' scorer,
  and ``tests/test_metrics.py`` checks this implementation against a vendored copy
  of that scorer point for point.
* **Clarke** and **Parkes (Consensus, type 1)** -- the classical grids, reported
  as familiar cross-checks.

MARD asymmetry worth keeping in mind when reading results: the denominator is the
true value, so a 20 mg/dL error at a hypoglycaemic 40 costs 50% while the same
error at 400 costs 5%. The hypoglycaemic tail dominates that metric, which is why
the stratified numbers in ``clinical_metrics`` matter more than the aggregate.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

ZONES = ("A", "B", "C", "D", "E")


def _finite(y_true, y_pred):
    t = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_pred, dtype=np.float64)
    m = np.isfinite(t) & np.isfinite(p) & (t > 0)
    return t[m], p[m]


# ---------------------------------------------------------------------- scalars
def rmse(y_true, y_pred) -> float:
    t, p = _finite(y_true, y_pred)
    return float(np.sqrt(np.mean((p - t) ** 2))) if t.size else float("nan")


def mae(y_true, y_pred) -> float:
    t, p = _finite(y_true, y_pred)
    return float(np.mean(np.abs(p - t))) if t.size else float("nan")


def mard(y_true, y_pred) -> float:
    """Mean absolute relative difference, %."""
    t, p = _finite(y_true, y_pred)
    return float(np.mean(np.abs(p - t) / t) * 100.0) if t.size else float("nan")


def bias(y_true, y_pred) -> float:
    """Signed mean error: positive means the model over-predicts glucose."""
    t, p = _finite(y_true, y_pred)
    return float(np.mean(p - t)) if t.size else float("nan")


# ------------------------------------------------------------------- DTS grid
def dts_zones(y_true, y_pred) -> np.ndarray:
    """DTS error-grid zone per point as 'A'..'E' ('?' where either is non-finite)."""
    ref = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)

    ab_up = np.where(ref <= 50, 60.0, (540.0 / 450.0) * (ref - 50) + 60.0)
    bc_up = np.where(ref <= 50, 86.5, (513.5 / 297.0) * (ref - 50) + 86.5)
    cd_up = np.where(ref <= 50, 124.0, (476.0 / 191.0) * (ref - 50) + 124.0)
    de_up = np.where(ref <= 50, 179.0, (421.0 / 117.0) * (ref - 50) + 179.0)
    ab_lo = np.where(ref <= 62.5, 0.0, (430.0 / 537.5) * (ref - 62.5) + 50.0)
    bc_lo = np.where(ref <= 97.5, 0.0, (257.0 / 502.5) * (ref - 97.5) + 50.0)
    cd_lo = np.where(ref <= 153.0, 0.0, (147.0 / 447.0) * (ref - 153.0) + 50.0)
    de_lo = np.where(ref <= 238.0, 0.0, (76.0 / 362.0) * (ref - 238.0) + 50.0)

    zone = np.full(ref.shape, "E", dtype="<U1")
    zone[((pred <= de_up) & (pred > cd_up)) | ((pred < cd_lo) & (pred >= de_lo))] = "D"
    zone[((pred <= cd_up) & (pred > bc_up)) | ((pred < bc_lo) & (pred >= cd_lo))] = "C"
    zone[((pred <= bc_up) & (pred > ab_up)) | ((pred < ab_lo) & (pred >= bc_lo))] = "B"
    zone[(pred <= ab_up) & (pred >= ab_lo)] = "A"
    zone[~np.isfinite(ref) | ~np.isfinite(pred)] = "?"
    return zone


def clarke_zones(y_true, y_pred) -> np.ndarray:
    ref = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    zone = np.full(ref.shape, "B", dtype="<U1")
    a = (np.abs(pred - ref) <= 0.2 * ref) | ((ref < 70) & (pred < 70))
    e = ((ref <= 70) & (pred >= 180)) | ((ref >= 180) & (pred <= 70))
    c = (((ref >= 70) & (ref <= 290)) & (pred >= ref + 110)) | \
        (((ref >= 130) & (ref <= 180)) & (pred <= (7.0 / 5.0) * ref - 182))
    d = ((ref >= 240) & (pred >= 70) & (pred <= 180)) | \
        ((ref <= 175.0 / 3.0) & (pred <= 180) & (pred >= 70)) | \
        (((ref >= 175.0 / 3.0) & (ref <= 70)) & (pred >= (6.0 / 5.0) * ref))
    zone[d] = "D"; zone[c] = "C"; zone[e] = "E"; zone[a] = "A"
    zone[~np.isfinite(ref) | ~np.isfinite(pred)] = "?"
    return zone


_PARKES_UPPER = {
    "AB": [(0, 50), (30, 50), (140, 170), (280, 380), (430, 550)],
    "BC": [(0, 60), (30, 60), (50, 80), (70, 110), (260, 550)],
    "CD": [(0, 100), (25, 100), (50, 125), (80, 215), (125, 550)],
    "DE": [(0, 150), (35, 155), (50, 550)],
}
_PARKES_LOWER = {
    "AB": [(50, 0), (50, 30), (170, 145), (385, 300), (550, 450)],
    "BC": [(120, 0), (120, 30), (260, 130), (550, 250)],
    "CD": [(250, 0), (250, 40), (410, 110), (550, 160)],
}


def parkes_zones(y_true, y_pred) -> np.ndarray:
    ref = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    zone = np.full(ref.shape, "A", dtype="<U1")
    up = pred >= ref
    for name, letter in (("AB", "B"), ("BC", "C"), ("CD", "D"), ("DE", "E")):
        v = _PARKES_UPPER[name]
        bound = np.interp(ref, [x for x, _ in v], [y for _, y in v],
                          left=v[0][1], right=np.inf)
        zone[up & (pred > bound)] = letter
    for name, letter in (("AB", "B"), ("BC", "C"), ("CD", "D")):
        v = _PARKES_LOWER[name]
        bound = np.interp(ref, [x for x, _ in v], [y for _, y in v],
                          left=-np.inf, right=v[-1][1])
        zone[(~up) & (pred < bound)] = letter
    zone[~np.isfinite(ref) | ~np.isfinite(pred)] = "?"
    return zone


def zone_fractions(zones: np.ndarray, prefix: str) -> Dict[str, float]:
    valid = zones[zones != "?"]
    n = max(valid.size, 1)
    out = {f"{prefix}_{z}": float((valid == z).sum()) / n * 100.0 for z in ZONES}
    out[f"{prefix}_AB"] = out[f"{prefix}_A"] + out[f"{prefix}_B"]
    return out


# ------------------------------------------------------------------- reporting
def point_metrics(y_true, y_pred, grids: bool = True) -> Dict[str, float]:
    t, p = _finite(y_true, y_pred)
    if t.size == 0:
        return {"n": 0}
    out: Dict[str, float] = {
        "n": int(t.size),
        "rmse": rmse(t, p),
        "mae": mae(t, p),
        "mard": mard(t, p),
        "bias": bias(t, p),
        # ISO 15197-style agreement, a familiar secondary read
        "within_15_15": float(np.mean((np.abs(p - t) <= 15) | (np.abs(p - t) / t <= 0.15)) * 100),
        "within_20_20": float(np.mean((np.abs(p - t) <= 20) | (np.abs(p - t) / t <= 0.20)) * 100),
    }
    if grids:
        out.update(zone_fractions(dts_zones(t, p), "dts"))
        out.update(zone_fractions(clarke_zones(t, p), "clarke"))
        out.update(zone_fractions(parkes_zones(t, p), "parkes"))
    return out


def per_subject_metrics(y_true, y_pred, subjects) -> Dict[str, float]:
    """Subject-averaged metrics.

    Row-weighted scores are dominated by whoever has the longest recording -- one
    cohort is 61% of the corpus -- so the subject distribution is reported too.
    """
    t = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_pred, dtype=np.float64)
    s = np.asarray(subjects)
    m = np.isfinite(t) & np.isfinite(p) & (t > 0)
    t, p, s = t[m], p[m], s[m]
    if t.size == 0:
        return {}
    order = np.argsort(s, kind="stable")
    t, p, s = t[order], p[order], s[order]
    bounds = np.flatnonzero(np.diff(s)) + 1
    rmses, mards = [], []
    for chunk_t, chunk_p in zip(np.split(t, bounds), np.split(p, bounds)):
        rmses.append(np.sqrt(np.mean((chunk_p - chunk_t) ** 2)))
        mards.append(np.mean(np.abs(chunk_p - chunk_t) / chunk_t) * 100)
    rmses, mards = np.array(rmses), np.array(mards)
    return {
        "n_subjects": int(rmses.size),
        "rmse_subject_mean": float(rmses.mean()),
        "rmse_subject_median": float(np.median(rmses)),
        "rmse_subject_p90": float(np.quantile(rmses, 0.90)),
        "rmse_subject_worst": float(rmses.max()),
        "mard_subject_mean": float(mards.mean()),
        "mard_subject_median": float(np.median(mards)),
    }
