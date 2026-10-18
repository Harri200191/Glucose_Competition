#!/usr/bin/env bash
# Mirror the published CGM-JEPA weights locally. Idempotent; ~8 MB total.
#
#   scripts/download_cgm_fm_weights.sh            # -> artifacts/fm_weights
#   DEST=/somewhere scripts/download_cgm_fm_weights.sh
#
# Source: https://huggingface.co/CRUISEResearchGroup/CGM-JEPA  (MIT licence)
# Layout matches `huggingface-cli download ... --local-dir`, so either tool works.
#
# There is deliberately no GlucoFM download here: Google/UNSW have not released a
# checkpoint. See docs/CGM_FOUNDATION_MODELS.md.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${DEST:-$REPO_ROOT/artifacts/fm_weights}"
HF="${HF:-https://huggingface.co/CRUISEResearchGroup/CGM-JEPA/resolve/main}"

for sub in cgm_jepa x_cgm_jepa; do
  mkdir -p "$DEST/$sub"
  for f in config.json model.safetensors; do
    if [[ -s "$DEST/$sub/$f" ]]; then
      echo "have    $sub/$f"; continue
    fi
    echo "fetch   $sub/$f"
    curl -fsSL --retry 3 -o "$DEST/$sub/$f.part" "$HF/$sub/$f"
    mv "$DEST/$sub/$f.part" "$DEST/$sub/$f"
  done
done

# Load both checkpoints strictly. A partial download that still parses would
# otherwise surface as a mysteriously weak "pretrained" run.
"${PY:-python}" - "$DEST" <<'PYEOF'
import sys
from metabonet_forecasting.models.cgm_jepa_net import load_pretrained
for v in ("cgm_jepa", "x_cgm_jepa"):
    m = load_pretrained(sys.argv[1], v)
    n = sum(p.numel() for p in m.parameters())
    assert n == 522_160, f"{v}: {n:,} parameters, expected 522,160"
    print(f"verified {v}: {n:,} trainable params, strict load OK")
PYEOF
echo "CGM-JEPA weights ready in $DEST"
