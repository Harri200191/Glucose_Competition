#!/usr/bin/env bash
# Stage F / experiment E3 -- the context-length study.
#
#   scripts/50_stageF_context_length.sh
#   CONTEXTS="24 96 288" SEEDS="0" scripts/50_stageF_context_length.sh
#
# 2 / 4 / 8 / 12 / 24 hours (24, 48, 96, 144, 288 steps) for TCN, PatchTST and
# Mamba-2. This is the experiment that decides whether the selective-state
# argument for Mamba-2 has any purchase here: if long context is worthless on
# this task, a linear-time long-context model is solving a problem nobody has.
#
# The comparison is clean because the anchor set does not change with context --
# the store guarantees 24 h of clean history for every anchor, so a 2 h run and a
# 24 h run are scored on identical samples with identical labels.
#
# The TCN's depth is chosen per context so its receptive field covers the window;
# each run records the field it actually had.
#
# Cost: 15 runs per seed. Short contexts are much cheaper, so the sweep starts
# with them.

source "$(dirname "$0")/lib/common.sh"
require_store

CONTEXTS="${CONTEXTS:-24 48 96 144 288}"
BACKBONES="${BACKBONES:-tcn patchtst mamba2}"
SEEDS="${SEEDS:-0 1}"

queue_init
for context in $CONTEXTS; do
  for backbone in $BACKBONES; do
    for seed in $SEEDS; do
      extra=()
      # PatchTST needs a patch shorter than the window; at 2 h use 4-step patches.
      if [[ "$backbone" == "patchtst" ]]; then
        if   (( context <= 24 ));  then extra=(model.backbone.patch_len=4  model.backbone.stride=4)
        elif (( context <= 48 ));  then extra=(model.backbone.patch_len=8  model.backbone.stride=8)
        else                            extra=(model.backbone.patch_len=16 model.backbone.stride=16)
        fi
      fi
      train "e3_${backbone}_ctx${context}_s${seed}" "model/${backbone}" \
        "seed=${seed}" "data.context=${context}" \
        model.fusion.name=raw_concat "${extra[@]}"
    done
  done
done
queue_wait

summarise "e3_*"

log "context curve figure"
$PY -m metabonet_forecasting.evaluation.plots "$RESULTS_DIR/all_runs.csv" \
  --out "$REPO_ROOT/figures" --metric rmse || warn "plotting failed"
log "stage F done -> figures/context_curve_rmse.png"
