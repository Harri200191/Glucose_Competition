#!/usr/bin/env bash
# Stage C / experiment E2 -- controlled backbone comparison.
#
#   scripts/20_stageC_backbones.sh              # 3 backbones x 3 seeds
#   SEEDS="0" scripts/20_stageC_backbones.sh    # one seed, faster
#
# Every run uses the same store, the same anchors, the same context, the same
# horizons, the same head and the same optimiser budget. The input representation
# is held at the simplest setting (raw concatenation) so that what differs
# between these runs is the temporal backbone and nothing else.
#
# TFT and TiDE are included here in native form: they are the two architectures
# the plan says must not be given the common front end, so this is the only stage
# where they appear against the others on equal footing.
#
# Cost: ~2-6 h per run on one V100 at 2M training windows; the queue spreads runs
# across GPUs.

source "$(dirname "$0")/lib/common.sh"
require_store

SEEDS="${SEEDS:-0 1 2}"
CONTEXT="${CONTEXT:-288}"

queue_init
for seed in $SEEDS; do
  for backbone in tcn patchtst mamba2; do
    train "e2_${backbone}_rawconcat_ctx${CONTEXT}_s${seed}" \
      "model/${backbone}" \
      "seed=${seed}" \
      "data.context=${CONTEXT}" \
      model.fusion.name=raw_concat
  done

  # Native-form architectures. TFT's variable selection is expensive at 24 h of
  # context, so it keeps its config's own (shorter) context; that difference is
  # recorded in its manifest and must be stated wherever the table is printed.
  train "e2_tide_native_ctx${CONTEXT}_s${seed}" model/tide \
    "seed=${seed}" "data.context=${CONTEXT}"
  train "e2_tft_native_s${seed}" model/tft "seed=${seed}"
done
queue_wait

summarise "e2_*"
log "stage C done. Compare two runs with:"
log "  $PY -m metabonet_forecasting.evaluation.aggregate $RUNS_DIR \\"
log "     --compare e2_mamba2_rawconcat_ctx${CONTEXT}_s0 e2_tcn_rawconcat_ctx${CONTEXT}_s0 --horizon 60"
