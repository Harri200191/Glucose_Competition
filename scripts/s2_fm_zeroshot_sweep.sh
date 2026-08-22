#!/usr/bin/env bash
# Zero-shot foundation-model ranking on the official test split, stride 8
# (~47k anchors). CPU, sequential, niced — the GPUs belong to the trained runs.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/tayyibul/anaconda3/envs/glucose/bin/python
STORE=${STORE:-/home/hrehman5/glucose/artifacts/store/main}
COMMON="--runs-dir runs --stride 8 --batch 256
        --set data.store=$STORE data.loader.num_workers=0"

run() {
  local name="$1"; shift
  if [[ -f "runs/$name/predictions_test.parquet" ]]; then
    echo "skip $name (exists)"; return
  fi
  echo "=== $name ==="
  OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES="" TIREX_NO_CUDA=1 \
    nice -n 10 $PY -m metabonet_forecasting.fm.zeroshot \
    --name "$name" $COMMON "$@" \
    || echo "FAILED $name"
}

run fm_chronos2_zs        --model chronos2
run fm_chronos2_zs_futins --model chronos2 --covariates insulin
run fm_tirex_zs           --model tirex
run fm_chronosbolt_zs     --model chronos-bolt
run fm_timesfm_zs         --model timesfm
echo "SWEEP DONE"
