"""Model contracts.

Two things are checked: that every architecture honours the common API on the
common batch, and that the one piece of non-trivial mathematics in the repository
-- the chunked SSD scan behind Mamba-2 -- computes what it claims to compute.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from metabonet_forecasting.common.config import load_config
from metabonet_forecasting.data.dataset import build_data, model_input_shapes
from metabonet_forecasting.models import available_backbones, build_model
from metabonet_forecasting.models.mamba2 import ssd_chunked, ssd_recurrent_reference
from metabonet_forecasting.data.windows import collate

BACKBONES = ["tcn", "patchtst", "mamba2", "lstm", "tide", "tft",
             "moderntcn", "xlstm", "transformerpp", "mambaformer"]


# ------------------------------------------------------------------ SSD kernel
def test_chunked_ssd_matches_the_explicit_recurrence():
    torch.manual_seed(0)
    b, l, h, p, n = 2, 48, 3, 8, 16
    X = torch.randn(b, l, h, p, dtype=torch.float64)
    A = -torch.rand(b, l, h, dtype=torch.float64) * 0.5      # negative log-decay
    B = torch.randn(b, l, h, n, dtype=torch.float64)
    C = torch.randn(b, l, h, n, dtype=torch.float64)
    for chunk in (8, 16, 24):
        y_chunked, _ = ssd_chunked(X, A, B, C, chunk)
        y_ref = ssd_recurrent_reference(X, A, B, C)
        assert torch.allclose(y_chunked, y_ref, atol=1e-8), f"chunk={chunk}"


def test_chunked_ssd_handles_lengths_that_do_not_divide_the_chunk():
    torch.manual_seed(1)
    b, l, h, p, n = 1, 37, 2, 4, 8
    X = torch.randn(b, l, h, p, dtype=torch.float64)
    A = -torch.rand(b, l, h, dtype=torch.float64)
    B = torch.randn(b, l, h, n, dtype=torch.float64)
    C = torch.randn(b, l, h, n, dtype=torch.float64)
    y, _ = ssd_chunked(X, A, B, C, chunk_size=16)
    assert y.shape == (b, l, h, p)
    assert torch.allclose(y, ssd_recurrent_reference(X, A, B, C), atol=1e-8)


def test_ssd_state_decays_rather_than_exploding():
    """A long, quiet stretch after a single impulse must fade, not grow."""
    b, l, h, p, n = 1, 200, 1, 1, 4
    X = torch.zeros(b, l, h, p, dtype=torch.float64)
    X[0, 0] = 1.0
    A = torch.full((b, l, h), -0.05, dtype=torch.float64)
    B = torch.ones(b, l, h, n, dtype=torch.float64)
    C = torch.ones(b, l, h, n, dtype=torch.float64)
    y, _ = ssd_chunked(X, A, B, C, chunk_size=25)
    trace = y[0, :, 0, 0].abs()
    assert trace[0] > trace[50] > trace[150]
    assert torch.isfinite(trace).all()


# ----------------------------------------------------------------- model API
@pytest.fixture(scope="module")
def bundle_and_shapes(request):
    store_path = request.getfixturevalue("store_path")
    cfg = load_config("base", [
        f"data.store={store_path}", "data.context=48", "data.horizons_min=[30, 60]",
        "data.sampling.max_train_windows=200", "data.sampling.max_val_windows=100",
        "data.sampling.max_test_windows=100",
        "data.sampling.standardizer_sample=200",
        "data.loader.num_workers=0", "data.loader.batch_size=8",
    ])
    bundle = build_data(cfg)
    return cfg, bundle, model_input_shapes(bundle)


@pytest.mark.parametrize("backbone", BACKBONES)
def test_every_backbone_produces_the_expected_shape(bundle_and_shapes, backbone):
    cfg, bundle, shapes = bundle_and_shapes
    cfg = cfg.copy()
    cfg.set_path("model.backbone.name", backbone)
    if backbone in ("tft", "tide"):
        cfg.set_path("model.fusion.name", "native")
    if backbone == "patchtst":
        cfg.set_path("model.backbone.patch_len", 8)
        cfg.set_path("model.backbone.stride", 8)
        cfg.set_path("model.backbone.n_heads", 4)
    if backbone == "mamba2":
        cfg.set_path("model.backbone.chunk_size", 12)
        cfg.set_path("model.backbone.headdim", 32)

    model = build_model(cfg, shapes).eval()
    batch = collate([bundle.datasets["train"][i] for i in range(4)])
    with torch.no_grad():
        out = model(batch)
    assert out.pred_norm.shape == (4, shapes["n_horizons"])
    assert torch.isfinite(out.pred_norm).all()
    assert model.n_parameters() > 0


@pytest.mark.parametrize("fusion", ["raw_concat", "concat", "gated", "film"])
def test_every_fusion_runs_on_the_same_batch(bundle_and_shapes, fusion):
    cfg, bundle, shapes = bundle_and_shapes
    cfg = cfg.copy()
    cfg.set_path("model.fusion.name", fusion)
    cfg.set_path("model.backbone.name", "tcn")
    model = build_model(cfg, shapes).eval()
    batch = collate([bundle.datasets["train"][i] for i in range(4)])
    with torch.no_grad():
        out = model(batch)
    assert out.pred_norm.shape == (4, shapes["n_horizons"])


def test_film_starts_as_the_identity(bundle_and_shapes):
    """gamma = 1 + delta with delta initialised to zero: FiLM must begin as a no-op.

    That makes any later modulation a departure the optimiser had to pay for,
    rather than noise injected at step zero.
    """
    cfg, bundle, shapes = bundle_and_shapes
    cfg = cfg.copy()
    cfg.set_path("model.fusion.name", "film")
    cfg.set_path("model.backbone.name", "tcn")
    model = build_model(cfg, shapes).eval()
    batch = collate([bundle.datasets["train"][i] for i in range(4)])
    with torch.no_grad():
        model(batch)
    assert torch.allclose(model.fusion.last_gamma, torch.ones_like(model.fusion.last_gamma))


def test_native_models_refuse_the_common_fusion_module(bundle_and_shapes):
    cfg, _, shapes = bundle_and_shapes
    cfg = cfg.copy()
    cfg.set_path("model.backbone.name", "tft")
    cfg.set_path("model.fusion.name", "film")
    with pytest.raises(ValueError, match="native form"):
        build_model(cfg, shapes)


def test_head_starts_at_the_persistence_forecast(bundle_and_shapes):
    """With delta targets, a zero-initialised head predicts "no change"."""
    cfg, bundle, shapes = bundle_and_shapes
    cfg = cfg.copy()
    cfg.set_path("model.backbone.name", "tcn")
    model = build_model(cfg, shapes).eval()
    batch = collate([bundle.datasets["train"][i] for i in range(4)])
    with torch.no_grad():
        out = model(batch)
    assert torch.allclose(out.pred_norm, torch.zeros_like(out.pred_norm))


def test_registry_lists_every_architecture_in_the_plan():
    assert set(available_backbones()) >= {"tcn", "patchtst", "mamba2", "tft", "tide"}


# ------------------------------------------------- ablation switches actually run
VARIANTS = [
    ("no_event_features", ["data.use_event_features=false"]),
    ("no_masks", ["data.use_masks=false"]),
    ("ffill", ["data.missing_policy=ffill"]),
    ("learned_missing", ["data.missing_policy=learned",
                         "model.encoder.missing_embedding=true"]),
    ("no_static", ["data.use_static=false"]),
    ("source_identity", ["data.use_source_identity=true"]),
    ("no_time", ["data.use_time_features=false"]),
    ("cgm_differences", ["model.encoder.cgm_differences=true"]),
    ("absolute_target", ["data.target_mode=absolute"]),
    ("quantiles", ["model.head.quantiles=[0.1, 0.5, 0.9]"]),
    ("cgm_only", ["data.modalities=['cgm']"]),
    ("with_activity", ["data.modalities=['cgm','insulin','carbs','activity']"]),
    ("gated", ["model.fusion.name=gated"]),
    ("subject_kfold", ["data.split.scheme=subject_kfold", "data.split.n_folds=3"]),
]


@pytest.mark.parametrize("name,overrides", VARIANTS, ids=[v[0] for v in VARIANTS])
def test_every_ablation_switch_builds_and_runs(store_path, name, overrides):
    """Each config switch the experiment scripts use must actually work.

    These are the knobs stages D and E turn. A switch that silently does nothing,
    or that crashes an hour into a sweep, invalidates the ablation it belongs to.
    """
    cfg = load_config("base", [
        f"data.store={store_path}", "data.context=48", "data.horizons_min=[30, 60]",
        "data.sampling.max_train_windows=200", "data.sampling.max_val_windows=100",
        "data.sampling.max_test_windows=100", "data.sampling.standardizer_sample=200",
        "data.loader.num_workers=0", "model.d_model=32",
        "model.backbone.n_levels=2", "model.backbone.auto_levels=false",
    ] + overrides)
    bundle = build_data(cfg)
    shapes = model_input_shapes(bundle)
    model = build_model(cfg, shapes).eval()
    batch = collate([bundle.datasets["train"][i] for i in range(4)])
    with torch.no_grad():
        out = model(batch)
    assert out.pred_norm.shape == (4, 2)
    assert torch.isfinite(out.pred_norm).all()
    if "quantiles" in name:
        assert out.quantiles is not None and out.quantiles.shape == (4, 2, 3)
        # monotone by construction
        assert (out.quantiles.diff(dim=-1) >= 0).all()


def test_missing_policies_differ_where_cgm_is_absent(store_path):
    """zero and ffill must not produce identical windows, or the ablation is empty."""
    import numpy as np
    from metabonet_forecasting.data.transforms import fill_missing
    x = np.array([[1.0], [np.nan], [np.nan], [4.0]], dtype=np.float32)
    assert fill_missing(x, None, "zero")[1, 0] == 0.0
    assert fill_missing(x, None, "ffill")[1, 0] == 1.0
    assert fill_missing(x, None, "ffill")[3, 0] == 4.0
