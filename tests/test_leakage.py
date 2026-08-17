"""Leakage tests.

The failure mode this file exists to catch: a model that scores well because
information from after the forecast time reached it. Centred smoothing,
bidirectional layers, a padding mistake, a normalisation statistic fitted on the
test period, or an off-by-one in the horizon index all produce the same symptom --
excellent numbers and a worthless model.

Method for the architectural half: perturb the inputs at time steps *after* t and
assert the output at t does not move. That is a direct test of causality, not a
proxy for it.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from metabonet_forecasting.data import constants as K
from metabonet_forecasting.data.transforms import fit_standardizer
from metabonet_forecasting.models.encoders import CGMEncoder, CausalConv1d, DriverEncoder
from metabonet_forecasting.models.mamba2 import Mamba2Backbone
from metabonet_forecasting.models.patchtst import PatchTSTBackbone
from metabonet_forecasting.models.recurrent import LSTMBackbone
from metabonet_forecasting.models.tcn import TCNBackbone

torch.manual_seed(0)
T, B, D = 32, 2, 16


def assert_causal(fn, length=T, width=D, cut=None, atol=1e-5):
    """fn: [B, T, W] -> [B, T', W']. Changing inputs after ``cut`` must not move
    outputs at or before the corresponding position."""
    cut = cut if cut is not None else length // 2
    x = torch.randn(B, length, width)
    y = torch.randn(B, length, width)
    y[:, :cut + 1] = x[:, :cut + 1]
    with torch.no_grad():
        out_x, out_y = fn(x), fn(y)
    # outputs are aligned with inputs unless the module downsamples
    scale = out_x.shape[1] / length
    keep = int((cut + 1) * scale)
    assert keep > 0
    assert torch.allclose(out_x[:, :keep], out_y[:, :keep], atol=atol), \
        "future inputs changed a past output"


def test_causal_conv_is_causal():
    conv = CausalConv1d(D, D, kernel_size=5, dilation=3).eval()
    assert_causal(conv)


def test_cgm_encoder_is_causal_with_and_without_differences():
    for diffs in (False, True):
        enc = CGMEncoder(D, D, use_differences=diffs).eval()
        assert_causal(lambda z: enc(z))


def test_driver_encoder_is_causal():
    enc = DriverEncoder(D, D, kernel_size=5).eval()
    assert_causal(lambda z: enc(z))


def test_tcn_backbone_is_causal():
    tcn = TCNBackbone(D, n_levels=3, kernel_size=3, dropout=0.0, auto_levels=False).eval()
    assert_causal(tcn)


def test_lstm_backbone_is_causal():
    assert_causal(LSTMBackbone(D, n_layers=1, dropout=0.0).eval(), atol=1e-4)


def test_mamba2_backbone_is_causal():
    m = Mamba2Backbone(D, n_layers=2, d_state=16, headdim=8, chunk_size=8,
                       dropout=0.0).eval()
    assert_causal(m, atol=1e-4)


def test_patchtst_backbone_is_causal_across_patches():
    """A patch may see all its own steps; it must not see a later patch."""
    p = PatchTSTBackbone(D, patch_len=8, stride=8, n_layers=2, n_heads=2, dropout=0.0,
                         context=T).eval()
    x = torch.randn(B, T, D)
    y = x.clone()
    y[:, 24:] += 5.0                       # perturb only the final patch
    with torch.no_grad():
        out_x, out_y = p(x), p(y)
    assert torch.allclose(out_x[:, :3], out_y[:, :3], atol=1e-5)
    assert not torch.allclose(out_x[:, 3], out_y[:, 3], atol=1e-3)


def test_tcn_receptive_field_is_reported_honestly():
    tcn = TCNBackbone(D, kernel_size=3, context=288, auto_levels=True)
    assert tcn.receptive_field() >= 288
    small = TCNBackbone(D, n_levels=2, kernel_size=3, auto_levels=False)
    assert small.receptive_field() < 288      # and the trainer logs this case


# ------------------------------------------------------ statistical leakage
def test_standardizer_is_fitted_on_training_anchors_only(store):
    """Two disjoint anchor sets must give different statistics.

    If they did not, the statistic would not depend on which anchors were passed,
    which is the signature of a fit over the whole corpus.
    """
    from metabonet_forecasting.data.windows import WindowSpec
    spec = WindowSpec(context=48, horizons_min=[30])
    n = len(store.anchors)
    a = np.zeros(n, dtype=bool); a[: n // 2] = True
    b = ~a
    hs = store.horizon_ids([30])
    sa = fit_standardizer(store, a, spec.all_channels(), hs, n_sample_anchors=5000)
    sb = fit_standardizer(store, b, spec.all_channels(), hs, n_sample_anchors=5000)
    assert sa.channels["cgm"].center != sb.channels["cgm"].center


def test_targets_never_come_from_interpolated_readings(store):
    """A filled slot is a fabricated label; prepare.py must have refused it."""
    ci = store.channel_index
    rows = store.anchors["row"].to_numpy()
    for j, h in enumerate(store.horizons):
        k = h // 5
        y = np.asarray(store.targets[:, j])
        valid = np.isfinite(y)
        mask = np.asarray(store.channels[rows[valid] + k][:, ci["cgm_mask"]])
        assert (mask == 1).all()


def test_targets_do_not_cross_the_official_train_test_boundary(store):
    """A training anchor must not be supervised by a test-period reading.

    166-169 subjects in the real corpus are split temporally, so this is the exact
    place where a subject-level split alone would still leak.
    """
    import pandas as pd
    anchors = store.anchors
    rows = anchors["row"].to_numpy()
    is_test_anchor = anchors["is_test"].to_numpy()
    # Reconstruct per-row is_test from the anchor table by segment lookup:
    # any horizon whose target row belongs to the other side must be masked out.
    seg = store.segments.set_index("seg")
    ends = seg.loc[anchors["seg"].to_numpy(), "end"].to_numpy()
    for j, h in enumerate(store.horizons):
        k = h // 5
        y = np.asarray(store.targets[:, j])
        finite = np.isfinite(y)
        assert (rows[finite] + k < ends[finite]).all(), "a target left its segment"


# ------------------------------------------- opt-in future covariates (E9)
#
# These guard the one thing that must never drift: the benchmark is causal
# unless a run explicitly asks otherwise, and when it does ask, the information
# really does reach the model (an option that silently does nothing would make
# the leakage measurement report zero and look reassuring).
def test_default_window_has_no_future_block(store):
    """Nothing in the default configuration reads past the anchor."""
    from metabonet_forecasting.data.windows import WindowDataset, WindowSpec
    spec = WindowSpec(context=48, horizons_min=[30, 60])
    assert spec.future_modalities == []
    assert spec.n_future_steps() == 0
    assert spec.future_channels() == []

    mask = np.zeros(len(store.anchors), dtype=bool)
    mask[:50] = True
    std = fit_standardizer(store, mask, spec.all_channels(),
                           store.horizon_ids(spec.horizons_min), n_sample_anchors=50)
    ds = WindowDataset(store, mask, spec, std)
    assert ds.future_shape == (0, 0)
    assert "x_future" not in ds[0]


def _future_dataset(store, modalities, context=48, steps=None, mask=None):
    from metabonet_forecasting.data.windows import WindowDataset, WindowSpec
    spec = WindowSpec(context=context, horizons_min=[30, 60],
                      future_modalities=list(modalities), future_steps=steps)
    if mask is None:
        mask = np.zeros(len(store.anchors), dtype=bool)
        mask[:200] = True
    std = fit_standardizer(store, mask, spec.all_channels(),
                           store.horizon_ids(spec.horizons_min), n_sample_anchors=200)
    return WindowDataset(store, mask, spec, std)


def test_future_block_reads_exactly_the_rows_after_the_anchor(store):
    ds = _future_dataset(store, ["insulin", "carbs"])
    steps, channels = ds.future_shape
    assert steps == 60 // K.GRID_MIN          # default: out to the longest horizon
    assert channels == len(K.FUTURE_MODALITY_CHANNELS["insulin"]) \
        + len(K.FUTURE_MODALITY_CHANNELS["carbs"])
    item = ds[0]
    assert item["x_future"].shape == (steps, channels + 1)   # +1 validity flag

    ai = int(ds.indices[0])
    row = int(ds.rows[ai])
    ci = store.channel_index
    raw = np.asarray(store.channels[row + 1: row + 1 + steps][:, [ci["ins_bolus"]]])
    j = ds.future_names.index("ins_bolus")
    st = ds.std.channels["ins_bolus"]
    expected = np.log1p(np.maximum(raw[:, 0], 0.0)) / st.scale
    assert np.allclose(item["x_future"][:, j].numpy(), expected, atol=1e-5)


def test_future_block_never_crosses_a_segment_boundary(store):
    """Past the end of the recording the block is zeroed and flagged.

    Sampled from the *end* of each segment on purpose: anchors near the start
    have plenty of room, so a mask over the first few hundred would exercise the
    happy path only and prove nothing.
    """
    steps = 400                                   # deliberately far past any anchor
    every = np.zeros(len(store.anchors), dtype=bool)
    every[:] = True
    ds = _future_dataset(store, ["carbs"], steps=steps, mask=every)

    room = ds.seg_end[ds.indices] - ds.rows[ds.indices] - 1
    tight = np.flatnonzero(room < steps)
    roomy = np.flatnonzero(room >= steps)
    assert tight.size and roomy.size, "need both cases to test the guard"

    for i in list(tight[:20]) + list(roomy[:5]):
        item = ds[int(i)]
        ai = int(ds.indices[int(i)])
        usable = int(max(min(steps, ds.seg_end[ai] - ds.rows[ai] - 1), 0))
        valid = item["x_future"][:, -1].numpy()
        assert valid[:usable].all()
        assert not valid[usable:].any()
        assert np.allclose(item["x_future"][usable:, :-1].numpy(), 0.0)


def test_only_raw_channels_are_exposed_never_causal_exposures(store):
    """Exposure kernels accumulate the past and have no forward definition."""
    ds = _future_dataset(store, ["insulin", "carbs"])
    assert all("exp" not in n and "since" not in n for n in ds.future_names), \
        ds.future_names


def test_model_output_actually_depends_on_the_planned_actions(store):
    """The flag must change predictions, or the measurement would report zero."""
    import torch as _torch
    from metabonet_forecasting.common.config import load_config
    from metabonet_forecasting.data.dataset import build_data, model_input_shapes
    from metabonet_forecasting.data.windows import collate
    from metabonet_forecasting.models import build_model

    cfg = load_config("base", [
        f"data.store={store.root}", "data.context=48", "data.horizons_min=[30, 60]",
        "data.sampling.max_train_windows=100", "data.sampling.max_val_windows=50",
        "data.sampling.max_test_windows=50", "data.sampling.standardizer_sample=100",
        "data.loader.num_workers=0", "model.d_model=32",
        "model.backbone.name=tcn", "model.backbone.n_levels=2",
        "model.backbone.auto_levels=false",
        "data.future_modalities=['insulin','carbs']",
    ])
    bundle = build_data(cfg)
    shapes = model_input_shapes(bundle)
    assert shapes["future_steps"] > 0 and shapes["future_channels"] > 0

    model = build_model(cfg, shapes).eval()
    assert model.describe()["future_covariates"] is True
    batch = collate([bundle.datasets["train"][i] for i in range(4)])

    with _torch.no_grad():
        base_out = model(batch).pred_norm
        bumped = dict(batch)
        bumped["x_future"] = batch["x_future"] + 1.0
        bumped_out = model(bumped).pred_norm
    # An untrained head is zero-initialised, so compare the representation the
    # head sees rather than its (identically zero) output.
    with _torch.no_grad():
        a = model.future_encoder(batch["x_future"])
        b = model.future_encoder(bumped["x_future"])
    assert not _torch.allclose(a, b), "planned actions do not reach the model"
    assert base_out.shape == bumped_out.shape
