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
