#!/usr/bin/env bash
# Check the environment can run the benchmark, and say precisely what is missing.
#
#   scripts/00_setup_env.sh              # check only
#   scripts/00_setup_env.sh --install    # pip install -r requirements.txt first
#
# Nothing here is required for correctness -- it exists so that a failure three
# hours into a sweep is not the first time anyone learns torch was missing.

#
# Planned future actions (off by default):
#   FUTURE=insulin|carbs|both  expose post-anchor insulin/carbs to the model.
#   Runs get a _futins/_futcarb/_futboth name suffix and are a DIFFERENT task
#   from the causal benchmark -- see scripts/lib/common.sh.

source "$(dirname "$0")/lib/common.sh"

if [[ "${1:-}" == "--install" ]]; then
  log "installing requirements"
  $PY -m pip install -r "$REPO_ROOT/requirements.txt"
fi

log "python: $($PY --version 2>&1) at $(command -v $PY)"

$PY - <<'PY'
import importlib, sys

required = ["numpy", "pandas", "pyarrow", "scipy", "sklearn", "torch", "yaml",
            "matplotlib", "pytest"]
optional = {
    "mamba_ssm": "official Mamba-2 CUDA kernels (Ampere+ only); the pure-PyTorch "
                 "SSD scan is used without it and is what the tests cover",
}

missing = []
for name in required:
    try:
        m = importlib.import_module(name)
        print(f"  ok       {name:12s} {getattr(m, '__version__', '')}")
    except Exception as exc:
        missing.append(name)
        print(f"  MISSING  {name:12s} ({exc})")

for name, why in optional.items():
    try:
        importlib.import_module(name)
        print(f"  ok       {name:12s} (optional)")
    except Exception:
        print(f"  absent   {name:12s} (optional) -- {why}")

try:
    import torch
    print(f"\n  torch {torch.__version__}, cuda={torch.version.cuda}, "
          f"available={torch.cuda.is_available()}, devices={torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"    gpu {i}: {p.name}, {p.total_memory / 2**30:.0f} GiB, sm_{p.major}{p.minor}")
except Exception as exc:
    print(f"  torch unusable: {exc}")

if missing:
    print(f"\nmissing required packages: {', '.join(missing)}")
    print("run: scripts/00_setup_env.sh --install")
    sys.exit(1)
print("\nenvironment ok")
PY

CORPUS="${CORPUS:-$REPO_ROOT/data/metabonet_public.parquet}"
if [[ -f "$CORPUS" ]]; then
  log "corpus present: $CORPUS ($(du -h "$CORPUS" | cut -f1))"
else
  warn "corpus not found at $CORPUS -- download it from https://metabo-net.org/"
fi

if [[ -f "$STORE/manifest.json" ]]; then
  log "sequence store present: $STORE"
  $PY -c "
import json;m=json.load(open('$STORE/manifest.json'))
print(f\"  {m['n_rows']:,} rows, {m['n_anchors']:,} anchors, {m['n_subjects']:,} subjects\")"
else
  warn "no sequence store yet -- run scripts/01_prepare_data.sh"
fi
