#!/usr/bin/env bash
# Stage B / experiment E1 -- the floor.
#
#   scripts/10_stageB_baselines.sh
#
# persistence, linear extrapolation, ridge on lagged features, gradient-boosted
# trees, and the simple LSTM. All on the same windows as every neural run, so the
# comparison is exact rather than approximate.
#
# Read the result this way: persistence at 120 minutes is the number every
# architecture in this repository has to beat before any of its other properties
# are interesting.
#
# Cost: minutes for persistence and linear trend, ~1-3 h for GBDT on 400k windows.

#
# Planned future actions (off by default):
#   FUTURE=insulin|carbs|both  expose post-anchor insulin/carbs to the model.
#   Runs get a _futins/_futcarb/_futboth name suffix and are a DIFFERENT task
#   from the causal benchmark -- see scripts/lib/common.sh.

source "$(dirname "$0")/lib/common.sh"
require_store
queue_init

log "tabular and trivial baselines"
for model in persistence linear_trend ridge gbdt; do
  baseline "e1_${model}" experiment/e1_baselines "$model"
done
queue_wait

log "simple LSTM through the common front end"
train "e1_lstm_raw_concat" model/lstm
queue_wait

summarise "e1_*"
log "stage B done -> $RESULTS_DIR/primary_table.csv"
