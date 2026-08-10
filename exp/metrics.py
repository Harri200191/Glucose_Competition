"""Scoring.

The competition scores **MARD** and **error-grid A-zone %** as two independent
tracks, with RMSE only breaking ties (README).

Caveat you must read before trusting the grid numbers
-----------------------------------------------------
The official track uses the **DTS / Surveillance Error Grid** (Klonoff et al.),
whose zones come from a published continuous risk surface, not a closed-form
boundary set. That surface ships with the organisers' toolkit, which is not
vendored in this repo. What is implemented here is:

  * **Clarke** error grid -- the classical formulation, unambiguous and widely
    reimplemented;
  * **Parkes / Consensus** error grid for Type 1 -- from the published boundary
    vertices in Pfützner et al. (2013).

Both are reasonable proxies for "how clinically dangerous is this error", and
Parkes is the closer relative of the DTS grid. Neither is a substitute for the
official scorer: **re-score with the organisers' toolkit before submitting.**
`dts_zones()` is a stub that makes that explicit rather than silently wrong.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["mard", "rmse", "mae", "clarke_zones", "parkes_zones", "dts_zones",
           "zone_fractions", "evaluate", "evaluate_by_subject"]


def _finite(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    m = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0)
    return y_true[m], y_pred[m], m


# --------------------------------------------------------------------- scalars
def mard(y_true, y_pred) -> float:
    """Mean Absolute Relative Difference, %. The competition's primary metric.

    Note the asymmetry this creates: the denominator is the true value, so a
    20 mg/dL error at a hypo of 40 costs 50% while the same error at 400 costs 5%.
    EDA 9.2 -- the hypoglycaemic tail dominates this metric.
    """
    t, p, _ = _finite(y_true, y_pred)
    if t.size == 0:
        return float("nan")
    return float(np.mean(np.abs(p - t) / t) * 100.0)


def rmse(y_true, y_pred) -> float:
    t, p, _ = _finite(y_true, y_pred)
    return float(np.sqrt(np.mean((p - t) ** 2))) if t.size else float("nan")


def mae(y_true, y_pred) -> float:
    t, p, _ = _finite(y_true, y_pred)
    return float(np.mean(np.abs(p - t))) if t.size else float("nan")


# ----------------------------------------------------------------- Clarke grid
def clarke_zones(y_true, y_pred) -> np.ndarray:
    """Clarke error-grid zone per point, as bytes b'A'..b'E'.

    `y_true` is the reference (sensor) value, `y_pred` the prediction. Both mg/dL.
    """
    ref = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    zone = np.full(ref.shape, "B", dtype="<U1")

    # A: within 20% of reference, or both below 70.
    a = (np.abs(pred - ref) <= 0.2 * ref) | ((ref < 70) & (pred < 70))

    # E: treatment would be opposite of correct.
    e = ((ref <= 70) & (pred >= 180)) | ((ref >= 180) & (pred <= 70))

    # C: over-correction.
    c = (((ref >= 70) & (ref <= 290)) & (pred >= ref + 110)) | \
        (((ref >= 130) & (ref <= 180)) & (pred <= (7.0 / 5.0) * ref - 182))

    # D: dangerous failure to detect.
    d = ((ref >= 240) & (pred >= 70) & (pred <= 180)) | \
        ((ref <= 175.0 / 3.0) & (pred <= 180) & (pred >= 70)) | \
        (((ref >= 175.0 / 3.0) & (ref <= 70)) & (pred >= (6.0 / 5.0) * ref))

    # Precedence: A wins, then E, then C, then D, else B.
    zone[d] = "D"
    zone[c] = "C"
    zone[e] = "E"
    zone[a] = "A"
    zone[~np.isfinite(ref) | ~np.isfinite(pred)] = "?"
    return zone


# ----------------------------------------------------------------- Parkes grid
# Boundary vertices as (reference, predicted), Type 1 diabetes.
# Pfützner, Klonoff, Pardo & Parkes (2013), J Diabetes Sci Technol 7(5):1275-81.
_PARKES_T1_UPPER = {
    "AB": [(0, 50), (30, 50), (140, 170), (280, 380), (430, 550)],
    "BC": [(0, 60), (30, 60), (50, 80), (70, 110), (260, 550)],
    "CD": [(0, 100), (25, 100), (50, 125), (80, 215), (125, 550)],
    "DE": [(0, 150), (35, 155), (50, 550)],
}
_PARKES_T1_LOWER = {
    "AB": [(50, 0), (50, 30), (170, 145), (385, 300), (550, 450)],
    "BC": [(120, 0), (120, 30), (260, 130), (550, 250)],
    "CD": [(250, 0), (250, 40), (410, 110), (550, 160)],
}


def _curve_upper(verts, x):
    """Boundary height at x. +inf past the last vertex -- the boundary has left
    the grid, so nothing can be above it there."""
    vx = np.array([v[0] for v in verts], dtype=np.float64)
    vy = np.array([v[1] for v in verts], dtype=np.float64)
    return np.interp(x, vx, vy, left=vy[0], right=np.inf)


def _curve_lower(verts, x):
    """-inf before the first vertex, for the same reason mirrored."""
    vx = np.array([v[0] for v in verts], dtype=np.float64)
    vy = np.array([v[1] for v in verts], dtype=np.float64)
    return np.interp(x, vx, vy, left=-np.inf, right=vy[-1])


def parkes_zones(y_true, y_pred) -> np.ndarray:
    """Parkes (Consensus) error-grid zone per point, Type 1 boundaries."""
    ref = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    zone = np.full(ref.shape, "A", dtype="<U1")

    up = pred >= ref
    lo = ~up

    # Upper half: A -> B -> C -> D -> E as the prediction rises above reference.
    for name, letter in (("AB", "B"), ("BC", "C"), ("CD", "D"), ("DE", "E")):
        above = up & (pred > _curve_upper(_PARKES_T1_UPPER[name], ref))
        zone[above] = letter

    # Lower half: Type 1 has no zone E below the diagonal.
    for name, letter in (("AB", "B"), ("BC", "C"), ("CD", "D")):
        below = lo & (pred < _curve_lower(_PARKES_T1_LOWER[name], ref))
        zone[below] = letter

    zone[~np.isfinite(ref) | ~np.isfinite(pred)] = "?"
    return zone


def dts_zones(y_true, y_pred):
    """The official DTS / Surveillance Error Grid. Not reimplemented here."""
    raise NotImplementedError(
        "The DTS Surveillance Error Grid is defined by a published risk surface that "
        "ships with the organisers' toolkit, not by closed-form boundaries. Score with "
        "the official toolkit; use parkes_zones() as the local proxy."
    )


def zone_fractions(zones: np.ndarray) -> dict[str, float]:
    valid = zones[zones != "?"]
    n = max(len(valid), 1)
    out = {f"zone_{z}": float((valid == z).sum()) / n * 100.0 for z in "ABCDE"}
    out["zone_AB"] = out["zone_A"] + out["zone_B"]
    return out


# ------------------------------------------------------------------- reporting
def evaluate(y_true, y_pred) -> dict[str, float]:
    """Every headline metric for one set of predictions."""
    t, p, _ = _finite(y_true, y_pred)
    if t.size == 0:
        return {"n": 0}
    res: dict[str, float] = {
        "n": int(t.size),
        "MARD": mard(t, p),
        "RMSE": rmse(t, p),
        "MAE": mae(t, p),
        # ISO 15197:2013 style agreement rate, a useful secondary read.
        "within_15_15": float(np.mean((np.abs(p - t) <= 15) | (np.abs(p - t) / t <= 0.15)) * 100),
        "within_20_20": float(np.mean((np.abs(p - t) <= 20) | (np.abs(p - t) / t <= 0.20)) * 100),
        # error concentrated where it is clinically dangerous
        "MARD_hypo_lt70": mard(t[t < 70], p[t < 70]) if (t < 70).any() else float("nan"),
        "MARD_hyper_gt250": mard(t[t > 250], p[t > 250]) if (t > 250).any() else float("nan"),
        "bias": float(np.mean(p - t)),
    }
    ck = zone_fractions(clarke_zones(t, p))
    res.update({f"clarke_{k[5:]}": v for k, v in ck.items()})
    pk = zone_fractions(parkes_zones(t, p))
    res.update({f"parkes_{k[5:]}": v for k, v in pk.items()})
    return res


def evaluate_by_subject(y_true, y_pred, subject_ids) -> dict[str, float]:
    """Subject-averaged metrics.

    EDA 8.3: the top 10% of subjects hold a disproportionate share of rows and
    `Loop` alone is 61% of the corpus, so a row-weighted score is really a score
    on the biggest recordings. Report both.
    """
    df = pd.DataFrame({"t": np.asarray(y_true, dtype=np.float64),
                       "p": np.asarray(y_pred, dtype=np.float64),
                       "s": np.asarray(subject_ids)})
    df = df[np.isfinite(df.t) & np.isfinite(df.p) & (df.t > 0)]
    if df.empty:
        return {}
    per = df.groupby("s", observed=True).apply(
        lambda g: pd.Series({
            "MARD": float(np.mean(np.abs(g.p - g.t) / g.t) * 100),
            "RMSE": float(np.sqrt(np.mean((g.p - g.t) ** 2))),
            "n": len(g),
        }), include_groups=False)
    return {
        "n_subjects": int(len(per)),
        "MARD_subject_mean": float(per.MARD.mean()),
        "MARD_subject_median": float(per.MARD.median()),
        "MARD_subject_p90": float(per.MARD.quantile(0.90)),
        "MARD_subject_worst": float(per.MARD.max()),
        "RMSE_subject_mean": float(per.RMSE.mean()),
    }
