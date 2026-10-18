"""Build each backbone for real and report its parameter count.

    python scripts/analysis/_check_capacity.py <d_model> <srcid true|false> <backbone>...

Called from the run_c96s preflight. Parameter counts are independent of context
length for these architectures -- transformerpp uses RoPE and the state-space
backbones have no learned positional table -- so this builds the shapes off the
small 288-step store, which takes seconds instead of the ~20 minutes a ctx1152
data build costs. The numbers are still exact for the run being launched.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from metabonet_forecasting.common.config import load_config
from metabonet_forecasting.data.dataset import build_data, model_input_shapes
from metabonet_forecasting.models import build_model

BASE = [
    "data.store=artifacts/store/main",
    "data.context=288",
    "data.target_mode=log_ratio",
    "data.future_modalities=['insulin']",
    "data.sampling.max_train_windows=20000",
    "data.sampling.max_val_windows=5000",
    "model.fusion.name=film",
    "model.head.name=horizon_cross",
    "model.head.chain=dist",
    "model.head.quantiles=[0.05,0.1,0.25,0.5,0.75,0.9,0.95]",
]


def main(d_model, srcid, backbones):
    base = BASE + [f"data.use_source_identity={srcid}"]
    shapes = model_input_shapes(build_data(load_config("model/transformerpp", base)))
    for b in backbones:
        cfg = load_config(f"model/{b}", base + [f"model.d_model={d_model}"])
        m = build_model(cfg, shapes)
        n = sum(p.numel() for p in m.parameters())
        print(f"    {b:<16} {n/1e6:6.2f} M parameters")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2], sys.argv[3:]))
