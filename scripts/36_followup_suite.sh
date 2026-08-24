#!/usr/bin/env bash
# The follow-up suite: every experiment agreed after the first review round.
#
#   scripts/36_followup_suite.sh
#   GPUS=0,1,2,3 scripts/36_followup_suite.sh          # restrict to some cards
#   RUN_GROUPS="A D" scripts/36_followup_suite.sh          # run only some groups
#   DRY_RUN=1 scripts/36_followup_suite.sh             # print the plan, run nothing
#
# ── GPUs ─────────────────────────────────────────────────────────────────────
# Set GPUS to the physical nvidia-smi indices you are allowed to use. The queue
# in lib/common.sh does admission control, not round-robin: a job starts only on
# a card that is *actually* free (>= GPU_FREE_MB reported free), and otherwise
# the queue blocks and re-checks every GPU_POLL_S seconds. So if the GPUs are
# busy — with your jobs or anyone else's — this waits rather than failing.
#
#   GPUS="0,1,2,3,4,5,6,7"   which cards to use     (default: all visible)
#   JOBS_PER_GPU=1           our jobs per card      (raise only for small models)
#
# Concurrency follows directly from those two: GPUS with 3 cards and
# JOBS_PER_GPU=1 means exactly 3 runs at a time, one per card, and the 4th run
# does not start until one of the three finishes. Give it 8 cards and it runs 8.
# If some of the cards you named are busy with someone else's job, it uses only
# the free ones and picks the rest up as they drain.
#   GPU_FREE_MB=12000        free memory required before dispatch
#   GPU_POLL_S=30            how often to re-check when nothing is free
#
# ── What differs from every earlier sweep ────────────────────────────────────
# ALL TRAINING WINDOWS. Previous runs capped training at 2,000,000 of the
# 12,118,476 available windows and then made ~14 passes over that fixed subset,
# which the training curves show was overfitting it. Here the cap is removed and
# the epoch count is cut to keep the compute budget comparable: 3 passes over
# 12.1 M windows is ~142,000 optimiser steps, against ~109,000 before.
#
# Consequences worth knowing before reading the results:
#   * `patience` is counted in epochs, so at 3 epochs early stopping can never
#     fire. Every run trains to completion. That is intended, not an oversight.
#   * the learning-rate schedule is defined over the run, so its shape differs
#     from the earlier sweep. Compare runs *within* this suite; do not compare a
#     number here against a number from the old table.
#   * random reads are spread over the whole 14 GB store rather than a cached
#     subset, so expect a slower step time even at equal step count.

source "$(dirname "$0")/lib/common.sh"

EPOCHS="${EPOCHS:-3}"
CONTEXT="${CONTEXT:-288}"
RUN_GROUPS="${RUN_GROUPS:-A B C D E F}"   # NB: not GROUPS -- bash owns that name
BACKBONES="${BACKBONES:-tcn patchtst mamba2}"

# Applied to every run in this suite: the whole point of it.
EXTRA_SET="${EXTRA_SET:-} data.sampling.max_train_windows=null training.epochs=${EPOCHS}"

require_store
mkdir -p "$LOG_DIR"

# Populates the device list from $GPUS. Without this the queue silently
# falls back to its CPU path: CUDA_VISIBLE_DEVICES is never set, every job
# lands on GPU 0 whatever its state, and they run one at a time.
queue_init

has_group() { [[ " $RUN_GROUPS " == *" $1 "* ]]; }

log "follow-up suite"
log "  context ${CONTEXT} · epochs ${EPOCHS} · ALL training windows (cap removed)"
log "  groups: ${RUN_GROUPS}"
log "  GPUs:   ${GPUS:-<none detected>}   (jobs/GPU ${JOBS_PER_GPU}, need ${GPU_FREE_MB} MiB free)"

# ─────────────────────────────────────────────────────────────────── group A
# Shared front end vs minimal input, inside each architecture. This is the
# comparison TFT/TiDE could not answer: they are different architectures, and
# TFT additionally ran at a different context length, so two things varied.
if has_group A; then
  log "── A · fusion grid (backbone x fusion), 6 runs"
  for bb in $BACKBONES; do
    train "fA_${bb}_rawconcat_s0" "model/${bb}" \
      "seed=0" "data.context=${CONTEXT}" "model.fusion.name=raw_concat"
    train "fA_${bb}_film_s0" "model/${bb}" \
      "seed=0" "data.context=${CONTEXT}" "model.fusion.name=film"
  done
fi

# ─────────────────────────────────────────────────────────────────── group B
# Engineered insulin/carbohydrate exposure features on vs off, for every
# backbone. Pairs with the raw_concat arm of group A, so the only difference is
# the 11 engineered channels.
if has_group B; then
  log "── B · engineered event features OFF, 3 runs"
  for bb in $BACKBONES; do
    train "fB_${bb}_noevents_s0" "model/${bb}" \
      "seed=0" "data.context=${CONTEXT}" "model.fusion.name=raw_concat" \
      "data.use_event_features=false"
  done
fi

# ─────────────────────────────────────────────────────────────────── group C
# Architecture-appropriate readout. For TCN and Mamba-2, `last` IS the native
# readout: both are causal, so the final position has absorbed the whole window.
# PatchTST's native readout is the flatten-all-patches head, which only exists in
# the native model — that half is group D.
if has_group C; then
  log "── C · last-step-only readout, 3 runs"
  for bb in $BACKBONES; do
    train "fC_${bb}_lastonly_s0" "model/${bb}" \
      "seed=0" "data.context=${CONTEXT}" "model.fusion.name=raw_concat" \
      "model.head.pooling=last"
  done
fi

# ─────────────────────────────────────────────────────────────────── group D
# Near-canonical PatchTST: channel-independent patching, overlapping patches and
# the native flatten-all-patches head. The stride-16 arm isolates the overlap
# from the other two changes. REVIN adds per-instance normalisation; it is not
# part of the agreed scope, so it is off unless asked for.
if has_group D; then
  log "── D · native PatchTST, 2 runs (+1 if REVIN=1)"
  train "fD_patchtstnative_s0" model/patchtst_native \
    "seed=0" "data.context=${CONTEXT}"
  train "fD_patchtstnative_stride16_s0" model/patchtst_native \
    "seed=0" "data.context=${CONTEXT}" "model.backbone.stride=16"
  if [[ "${REVIN:-0}" == "1" ]]; then
    train "fD_patchtstnative_revin_s0" model/patchtst_native \
      "seed=0" "data.context=${CONTEXT}" "model.backbone.instance_norm=true"
  fi
fi

# ─────────────────────────────────────────────────────────────────── group E
# Extra seeds for the paired comparison. Evaluation windows are already identical
# across runs: the test sample is drawn with data.sampling.seed, which is separate
# from the training seed, so changing `seed` cannot change which windows are
# scored. Verified by checksumming two runs' anchor lists.
if has_group E; then
  log "── E · seeds 1 and 2 for the paired comparison, 8 runs"
  for s in 1 2; do
    for bb in $BACKBONES; do
      train "fA_${bb}_rawconcat_s${s}" "model/${bb}" \
        "seed=${s}" "data.context=${CONTEXT}" "model.fusion.name=raw_concat"
    done
    train "fD_patchtstnative_s${s}" model/patchtst_native \
      "seed=${s}" "data.context=${CONTEXT}"
  done
fi

# ─────────────────────────────────────────────────────────────────── group F
# Do the static patient features earn their place? They are 32 of the 67 input
# dimensions in raw_concat -- nearly half the input width -- and they are the
# same 32 numbers repeated at every one of the 288 steps, because a patient's age
# does not change during a window. Nothing in the study has ever tested whether
# removing them costs anything. Pairs with the raw_concat arm of group A.
if has_group F; then
  log "── F · static patient features OFF, 3 runs"
  for bb in $BACKBONES; do
    train "fF_${bb}_nostatic_s0" "model/${bb}" \
      "seed=0" "data.context=${CONTEXT}" "model.fusion.name=raw_concat" \
      "data.use_static=false"
  done
fi

queue_wait

# ─────────────────────────────────────────────────────── score, plot, write up
log "aggregating"
summarise "f[ABCDEF]_*"

log "building figures and per-experiment write-ups"
# The repo interpreter's matplotlib is broken by the numpy-2 upgrade. Point
# PLOT_PY at one that works to get figures; the markdown is written either way.
"${PLOT_PY:-$PY}" scripts/analysis/followup_report.py || \
  warn "report script failed; results/all_runs.csv is still valid"

log "done."
log "  write-ups   docs/experiments/*.md   (start with 00_summary.md)"
log "  figures     artifacts/figures/followup/*.png"
log "  raw table   $RESULTS_DIR/all_runs.csv"
