#!/usr/bin/env bash
# Two-minute end-to-end check on the smoke store: every architecture trains for a
# few steps, predicts, and is scored.
#
#   scripts/01_prepare_data.sh --smoke   # once
#   scripts/02_smoke_test.sh
#
# Run this after any change to the data pipeline or the model API. It will not
# tell you whether a model is good; it tells you whether the plumbing is intact.
# Never report a number from these runs.
#
# Note on argument style: `train`/`baseline` append their arguments to a single
# --set list, so pass bare key=value overrides, not another --set.

#
# Planned future actions (off by default):
#   FUTURE=insulin|carbs|both  expose post-anchor insulin/carbs to the model.
#   Runs get a _futins/_futcarb/_futboth name suffix and are a DIFFERENT task
#   from the causal benchmark -- see scripts/lib/common.sh.

source "$(dirname "$0")/lib/common.sh"

STORE="${SMOKE_STORE:-$REPO_ROOT/artifacts/store/smoke}"
RUNS_DIR="$REPO_ROOT/runs/smoke"
RESULTS_DIR="$REPO_ROOT/results/smoke"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
require_store
mkdir -p "$RUNS_DIR" "$RESULTS_DIR"

SMOKE_SET=(
  data.context=96
  data.sampling.max_train_windows=4000
  data.sampling.max_val_windows=1000
  data.sampling.max_test_windows=2000
  data.sampling.standardizer_sample=4000
  data.loader.batch_size=64
  data.loader.num_workers=2
  training.epochs=1
  training.max_steps_per_epoch=30
  training.warmup_steps=5
  training.log_every=10
)

queue_init
for backbone in tcn patchtst mamba2 lstm tide tft; do
  case "$backbone" in
    tft|tide) fusion="native" ;;
    *)        fusion="film" ;;
  esac
  train "smoke_${backbone}" "model/${backbone}" \
    "${SMOKE_SET[@]}" "model.fusion.name=${fusion}"
done
queue_wait

log "baselines"
for model in persistence linear_trend ridge; do
  baseline "smoke_baseline_${model}" experiment/e1_baselines "$model" \
    data.context=96 \
    data.sampling.max_train_windows=4000 \
    data.sampling.max_test_windows=2000 \
    data.loader.num_workers=0
done
queue_wait

summarise
log "smoke test complete. Logs: $LOG_DIR  Runs: $RUNS_DIR  Results: $RESULTS_DIR"
