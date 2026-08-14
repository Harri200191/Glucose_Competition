#!/usr/bin/env bash
# Stage D / experiments E4 and E5 -- the core methodological contribution.
#
#   scripts/30_stageD_fusion_ablations.sh                 # Mamba-2 ladder + transfer
#   BACKBONES="mamba2" scripts/30_stageD_fusion_ablations.sh
#
# The ladder, in order, each rung adding exactly one thing:
#
#   1 raw_concat            normalised channels -> linear -> backbone
#   2 concat                + modality-specific shallow encoders
#   3 gated                 + learned per-modality gates
#   4 film                  + CGM-centred conditioning (the proposed model)
#   5 film -event_features  4 without the decay kernels and time-since-event
#   6 film -masks           4 without observation masks
#   7 film +learned missing 4 with a learned missing-modality embedding
#
# Rungs 5-7 are subtractive on purpose: the question is not "does the full model
# work" but "which component is carrying the result". Rung 4 minus rung 1 is the
# headline claim; rung 4 minus rungs 2 and 3 is what separates the claim about
# *conditioning* from the claim about *encoding*.
#
# E5 (is the gain fusion-specific or Mamba-specific?) is the same ladder repeated
# on TCN and PatchTST -- which is why BACKBONES defaults to all three.

source "$(dirname "$0")/lib/common.sh"
require_store

BACKBONES="${BACKBONES:-mamba2 tcn patchtst}"
SEEDS="${SEEDS:-0 1 2}"
CONTEXT="${CONTEXT:-288}"

queue_init
for backbone in $BACKBONES; do
  for seed in $SEEDS; do
    base=("seed=${seed}" "data.context=${CONTEXT}")

    train "e4_${backbone}_rawconcat_s${seed}" "model/${backbone}" \
      "${base[@]}" model.fusion.name=raw_concat
    train "e4_${backbone}_concat_s${seed}" "model/${backbone}" \
      "${base[@]}" model.fusion.name=concat
    train "e4_${backbone}_gated_s${seed}" "model/${backbone}" \
      "${base[@]}" model.fusion.name=gated
    train "e4_${backbone}_film_s${seed}" "model/${backbone}" \
      "${base[@]}" model.fusion.name=film

    # component knock-outs, on the FiLM model only
    train "e4_${backbone}_film_noevents_s${seed}" "model/${backbone}" \
      "${base[@]}" model.fusion.name=film data.use_event_features=false
    train "e4_${backbone}_film_nomasks_s${seed}" "model/${backbone}" \
      "${base[@]}" model.fusion.name=film data.use_masks=false
    train "e4_${backbone}_film_learnedmissing_s${seed}" "model/${backbone}" \
      "${base[@]}" model.fusion.name=film data.missing_policy=learned \
      model.encoder.missing_embedding=true
  done
done
queue_wait

summarise "e4_*"
log "stage D done. The headline paired comparison:"
for backbone in $BACKBONES; do
  log "  $PY -m metabonet_forecasting.evaluation.aggregate $RUNS_DIR \\"
  log "     --compare e4_${backbone}_film_s0 e4_${backbone}_rawconcat_s0 --horizon 120"
done
