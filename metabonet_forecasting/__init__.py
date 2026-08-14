"""MetaboNet multimodal glucose forecasting benchmark.

A controlled experimental system for the question: *how should heterogeneous
diabetes time-series signals be encoded and fused* so that sequence models learn
transferable glucose dynamics rather than short-term autocorrelation?

Layout
------
``common``      config/registry/seed/logging plumbing shared by everything
``data``        corpus -> sequence store -> windows -> dataloaders
``models``      modality encoders, fusion, temporal backbones, forecast heads
``training``    the single training loop every architecture shares
``evaluation``  predictive + clinical metrics, patient bootstrap, plots
``baselines``   persistence / linear extrapolation / ridge / GBDT on the same windows

Nothing in ``models`` knows how data was built; nothing in ``training`` knows which
architecture it is training. That separation is what makes the backbone comparison
controlled.
"""

__version__ = "0.1.0"
