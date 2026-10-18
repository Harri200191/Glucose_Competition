"""Engineered per-anchor features.

WHY THIS EXISTS, AND WHAT IT IS NOT
-----------------------------------
The models in this repo are sequence models over the raw 288x23 lattice: they
compute whatever rolling statistics they need inside their own hidden state. A
tabular/GBM competitor cannot, so it is handed hundreds of pre-computed
features instead. The open question this package exists to answer is whether
those features carry anything a causal sequence model does not already extract.

The repo's own prior on that is not encouraging. Experiment B (docs/experiments/
B_event_features.md) deleted all ELEVEN engineered pharmacology channels and
measured the cost at -0.37 / -0.26 / -0.22 DTS-A@60 for Mamba-2 / TCN /
PatchTST, against a seed noise of 0.34 / 0.18 / 0.06 -- i.e. 0 -> 11 engineered
features bought about 0.3 points, at the noise floor. Experiment F found the
patient statics worth 0.24. The whole A-F program spans 1.19 points.

So this is a falsification exercise, and it is built to be read honestly:
the arms differ in the feature block and in nothing else.

SHAPE OF THE THING
------------------
Features are PER ANCHOR, not per timestep. 700 extra channels over the store's
154.8M rows would be 0.4 TB; 700 features over the 1.39M anchors a mini-corpus
experiment touches is 3.9 GB. They are therefore computed once from each
anchor's own causal window and stored beside the run, never in the store --
which also means changing them does not bump SCHEMA_VERSION or invalidate any
existing build.

CAUSALITY IS STRUCTURAL HERE TOO. A feature sees exactly
``store.channels[row-L+1 : row+1]`` and the anchor's own scalars, the same
window the model sees. There is no path to a future row: the builder gathers by
that slice and nothing else.
"""
from .spec import FEATURE_FAMILIES, build_columns, compute_block   # noqa: F401
