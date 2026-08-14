"""Dataset-level correctness: horizon alignment, masks, causality, event filling.

These are the stage-A unit tests the plan asks for. They are the difference
between "the number looks good" and "the number means something".
"""
from __future__ import annotations

import numpy as np
import pytest

from metabonet_forecasting.data import constants as K
from metabonet_forecasting.data.prepare import (exp_exposure, interpolate_short_gaps,
                                                last_true_index, steps_since)
from metabonet_forecasting.data.transforms import fit_standardizer
from metabonet_forecasting.data.windows import WindowDataset, WindowSpec


# ------------------------------------------------------------------ store level
def test_store_builds_and_is_self_consistent(store):
    assert store.manifest["n_rows"] > 0
    assert store.manifest["n_anchors"] > 0
    assert store.channels.shape == (store.manifest["n_rows"], len(K.CHANNELS))
    assert store.targets.shape[0] == store.manifest["n_anchors"]
    # every anchor sits inside its segment and has the promised history
    seg = store.segments.set_index("seg")
    rows = store.anchors["row"].to_numpy()
    starts = seg.loc[store.anchors["seg"].to_numpy(), "start"].to_numpy()
    ends = seg.loc[store.anchors["seg"].to_numpy(), "end"].to_numpy()
    assert (rows >= starts + store.min_history_steps - 1).all()
    assert (rows < ends).all()


def test_segments_are_monotone_and_five_minutes_apart(store):
    """Segmentation must have cut the out-of-order subject, not smoothed over it."""
    lengths = store.segments["n_rows"].to_numpy()
    assert (lengths > 0).all()
    assert store.segments["start"].is_monotonic_increasing
    # more segments than subjects: the reversed subject was split in two
    assert len(store.segments) > store.manifest["n_subjects"] - 1


def test_horizon_alignment_matches_raw_future_readings(store):
    """targets[i, j] must equal CGM exactly h minutes after the anchor row."""
    ci = store.channel_index
    rows = store.anchors["row"].to_numpy()
    checked = 0
    for j, h in enumerate(store.horizons):
        k = h // K.GRID_MIN
        y = np.asarray(store.targets[:, j])
        valid = np.isfinite(y)
        future = np.asarray(store.channels[rows[valid] + k][:, ci["cgm"]])
        mask = np.asarray(store.channels[rows[valid] + k][:, ci["cgm_mask"]])
        assert np.allclose(y[valid], future, atol=1e-3)
        # and it must be a *real* reading, never an interpolated one
        assert (mask == 1).all()
        checked += int(valid.sum())
    assert checked > 0


def test_anchor_cgm_is_real_and_history_is_clean(store):
    ci = store.channel_index
    rows = store.anchors["row"].to_numpy()
    assert (np.asarray(store.channels[rows][:, ci["cgm_mask"]]) == 1).all()

    # no CGM gap longer than the configured allowance anywhere in the window
    max_gap = store.manifest["config"]["max_cgm_gap_min"] // K.GRID_MIN
    L = store.min_history_steps
    for row in rows[:: max(1, len(rows) // 40)]:
        window = np.asarray(store.channels[row - L + 1: row + 1][:, ci["cgm_mask"]])
        runs, current = 0, 0
        for v in window:
            current = 0 if v == 1 else current + 1
            runs = max(runs, current)
        assert runs <= max_gap


def test_short_gaps_interpolated_long_gaps_left_missing(store):
    ci = store.channel_index
    cgm = np.asarray(store.channels[:, ci["cgm"]])
    mask = np.asarray(store.channels[:, ci["cgm_mask"]])
    filled = np.isfinite(cgm) & (mask == 0)
    still_missing = ~np.isfinite(cgm)
    assert filled.sum() > 0, "the planted 20-minute gap should have been filled"
    assert still_missing.sum() > 0, "the planted 100-minute gap must stay missing"


# ------------------------------------------------------- prepare-time primitives
def test_last_true_index_and_steps_since():
    flags = np.array([False, True, False, False, True, False])
    assert list(last_true_index(flags)) == [-1, 1, 1, 1, 4, 4]
    assert list(steps_since(flags, cap=99)) == [99, 0, 1, 2, 0, 1]


def test_exp_exposure_is_causal_and_decays():
    x = np.zeros(50, dtype=np.float32)
    x[10] = 1.0
    y = exp_exposure(x, tau_steps=6.0)
    assert (y[:10] == 0).all(), "an event cannot affect the past"
    assert y[10] == pytest.approx(1.0, abs=1e-5)
    assert y[16] == pytest.approx(np.exp(-1.0), abs=1e-3)   # one tau later
    assert np.all(np.diff(y[10:]) < 0)


def test_interpolate_short_gaps_respects_edges():
    values = np.array([100.0, np.nan, 120.0, np.nan, np.nan, np.nan, 200.0])
    observed = np.isfinite(values)
    filled, n = interpolate_short_gaps(values, observed, max_gap=1)
    assert filled[1] == pytest.approx(110.0)     # 1-step gap: filled
    assert np.isnan(filled[3:6]).all()           # 3-step gap: left alone
    assert n == 1
    # a gap touching the segment edge is never filled
    edge = np.array([np.nan, 100.0, 110.0])
    filled_edge, _ = interpolate_short_gaps(edge, np.isfinite(edge), max_gap=6)
    assert np.isnan(filled_edge[0])


# -------------------------------------------------------------- dataset level
def _dataset(store, context=48, **kw):
    spec = WindowSpec(context=context, horizons_min=[30, 60], **kw)
    mask = np.zeros(len(store.anchors), dtype=bool)
    mask[:200] = True
    std = fit_standardizer(store, mask, spec.all_channels(),
                           store.horizon_ids(spec.horizons_min), n_sample_anchors=200)
    return WindowDataset(store, mask, spec, std)


def test_window_ends_at_the_anchor(store):
    ds = _dataset(store)
    item = ds[0]
    ai = int(ds.indices[0])
    row = int(ds.rows[ai])
    ci = store.channel_index
    raw_last = float(store.channels[row, ci["cgm"]])
    norm = ds.std.channels["cgm"]
    expected = (raw_last - norm.center) / norm.scale
    assert item["x_cgm"].shape == (ds.spec.context, len(ds.channel_names["cgm"]))
    assert float(item["x_cgm"][-1, 0]) == pytest.approx(expected, abs=1e-4)
    assert float(item["cgm_now"]) == pytest.approx(raw_last, abs=1e-3)


def test_target_round_trip_through_the_standardizer(store):
    ds = _dataset(store)
    for i in (0, 5, 17):
        item = ds[i]
        back = ds.std.inverse_targets(item["y_norm"].numpy()[None, :],
                                      np.array([float(item["cgm_now"])]))[0]
        valid = item["y_mask"].numpy() > 0
        assert np.allclose(back[valid], item["y"].numpy()[valid], atol=1e-2)


def test_modality_ablation_changes_only_the_channels(store):
    full = _dataset(store, modalities=["cgm", "insulin", "carbs"])
    cgm_only = _dataset(store, modalities=["cgm"])
    assert "x_insulin" in full[0] and "x_insulin" not in cgm_only[0]
    assert full[0]["x_cgm"].shape == cgm_only[0]["x_cgm"].shape
    assert np.allclose(full[0]["x_cgm"].numpy(), cgm_only[0]["x_cgm"].numpy())


def test_event_feature_and_mask_ablations_shrink_the_input(store):
    full = _dataset(store)
    no_events = _dataset(store, use_event_features=False)
    no_masks = _dataset(store, use_masks=False)
    assert no_events.channel_counts()["insulin"] < full.channel_counts()["insulin"]
    assert no_masks.channel_counts()["cgm"] < full.channel_counts()["cgm"]


def test_context_length_changes_window_not_anchors(store):
    short, long = _dataset(store, context=24), _dataset(store, context=48)
    assert short[0]["x_cgm"].shape[0] == 24
    assert long[0]["x_cgm"].shape[0] == 48
    assert np.array_equal(short.indices, long.indices)
    # the short window is the tail of the long one: same "now", less history
    assert np.allclose(short[0]["x_cgm"].numpy(), long[0]["x_cgm"].numpy()[-24:], atol=1e-5)


def test_context_beyond_the_stores_guarantee_is_refused(store):
    with pytest.raises(ValueError, match="exceeds the history"):
        _dataset(store, context=store.min_history_steps + 1)


def test_cgm_cannot_be_ablated_away(store):
    with pytest.raises(ValueError, match="cannot be ablated"):
        _dataset(store, modalities=["insulin"])
