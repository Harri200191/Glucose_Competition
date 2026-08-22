"""Metrics, checked against the organisers' own scorer.

The DTS error grid is the metric the public leaderboard reports, so an
independent reimplementation is only trustworthy if it agrees with the official
one point for point. ``official_toolkit_metrics.py`` is the vendored original.
"""
from __future__ import annotations

import numpy as np
import pytest

from metabonet_forecasting.evaluation import metrics as M
from metabonet_forecasting.evaluation import official_toolkit_metrics as OFFICIAL
from metabonet_forecasting.evaluation.bootstrap import bootstrap_metric, paired_bootstrap

ZONES = ("A", "B", "C", "D", "E")


@pytest.mark.parametrize("name,n", [("uniform", 50_000), ("realistic", 50_000),
                                    ("hypo", 20_000)])
def test_dts_zone_percentages_match_the_official_scorer(name, n):
    rng = np.random.default_rng(abs(hash(name)) % 2 ** 32)
    if name == "uniform":
        ref, pred = rng.uniform(20, 450, n), rng.uniform(20, 450, n)
    elif name == "realistic":
        ref = rng.uniform(40, 400, n)
        pred = np.clip(ref * rng.normal(1.0, 0.18, n), 5, 600)
    else:
        ref, pred = rng.uniform(30, 80, n), rng.uniform(20, 250, n)

    mine = M.zone_fractions(M.dts_zones(ref, pred), "dts")
    theirs = OFFICIAL.calculate_dts_error_grid(pred, ref)
    for z in ZONES:
        assert mine[f"dts_{z}"] == pytest.approx(theirs[f"DTS_{z}_ZONE_PERCENT"],
                                                 abs=0.01)


def test_dts_zone_assignment_matches_point_for_point_on_the_borders():
    """Border cases decide zone membership; check them one point at a time."""
    rng = np.random.default_rng(7)
    knots = np.array([50.0, 62.5, 97.5, 153.0, 238.0])
    ref = np.concatenate([rng.uniform(20, 450, 400), rng.choice(knots, 400)])
    pred = np.concatenate([rng.uniform(5, 600, 400),
                           ref[400:] * rng.uniform(0.2, 3.0, 400)])
    mine = M.dts_zones(ref, pred)
    for i in range(ref.size):
        d = OFFICIAL.calculate_dts_error_grid([pred[i]], [ref[i]])
        theirs = next(z for z in ZONES if d[f"DTS_{z}_ZONE_PERCENT"] == 100.0)
        assert mine[i] == theirs, f"ref={ref[i]:.3f} pred={pred[i]:.3f}"


def test_perfect_predictions_are_all_zone_a():
    g = np.linspace(20, 450, 2000)
    for zones in (M.dts_zones(g, g), M.clarke_zones(g, g), M.parkes_zones(g, g)):
        assert (zones == "A").all()


def test_mard_rmse_mae_and_bias_are_what_they_claim():
    y = np.array([100.0, 200.0, 50.0])
    p = np.array([110.0, 180.0, 55.0])
    assert M.rmse(y, p) == pytest.approx(np.sqrt((100 + 400 + 25) / 3))
    assert M.mae(y, p) == pytest.approx((10 + 20 + 5) / 3)
    assert M.mard(y, p) == pytest.approx((0.10 + 0.10 + 0.10) / 3 * 100)
    assert M.bias(y, p) == pytest.approx((10 - 20 + 5) / 3)


def test_metrics_ignore_missing_labels():
    y = np.array([100.0, np.nan, 150.0])
    p = np.array([110.0, 999.0, 150.0])
    assert M.point_metrics(y, p)["n"] == 2
    assert M.rmse(y, p) == pytest.approx(np.sqrt((100 + 0) / 2))


def test_mard_is_asymmetric_in_the_hypoglycaemic_direction():
    """A fixed absolute error costs far more at a low reference value.

    Worth pinning down: it is why the stratified numbers, not the aggregate, are
    the ones to read.
    """
    low = M.mard(np.array([40.0]), np.array([60.0]))
    high = M.mard(np.array([400.0]), np.array([420.0]))
    assert low == pytest.approx(50.0)
    assert high == pytest.approx(5.0)


# ------------------------------------------------------------------- bootstrap
def test_patient_bootstrap_interval_contains_the_point_estimate():
    rng = np.random.default_rng(0)
    subjects = np.repeat(np.arange(30), 50)
    y = rng.uniform(70, 250, subjects.size)
    p = y + rng.normal(0, 15, subjects.size)
    out = bootstrap_metric(y, p, subjects, "rmse", n_boot=200, seed=0)
    assert out["lo"] <= out["point"] <= out["hi"]
    assert out["n_subjects"] == 30


def test_paired_bootstrap_detects_a_consistently_better_model():
    import pandas as pd
    rng = np.random.default_rng(1)
    n, subjects = 3000, np.repeat(np.arange(20), 150)
    y = rng.uniform(70, 250, n)
    a = pd.DataFrame({"anchor": np.arange(n), "subject": subjects,
                      "y_true_60": y, "y_pred_60": y + rng.normal(0, 10, n)})
    b = pd.DataFrame({"anchor": np.arange(n), "y_pred_60": y + rng.normal(0, 25, n)})
    res = paired_bootstrap(a, b, 60, "rmse", n_boot=200, seed=0)
    assert res["difference"] < 0 and res["hi"] < 0        # A better, interval excludes 0
    assert res["favours"] == "A"
    assert res["subjects_A_better"] > 90
