#!/usr/bin/env bash
# Point 5: does future insulin help because of physiology, or because it leaks?
#
#   scripts/run_futins_experiment.sh                # main experiment
#   EPOCHS=8 scripts/run_futins_experiment.sh       # shorter first pass
#   STAGE=eval scripts/run_futins_experiment.sh     # analysis only
#   SUBGROUP=1 scripts/run_futins_experiment.sh     # + robustness arm (4 more runs)
#   GPUS=0,3 scripts/run_futins_experiment.sh       # override GPU auto-detect
#
# ===========================================================================
# THE ARGUMENT
#
# Future insulin obviously helps. The question is *why*, and the two answers
# make opposite predictions:
#
#   physiology  the model learns what a dose does to glucose  -> helps everyone
#   leakage     a closed-loop pump computes the dose FROM the glucose being
#               forecast, so the dose is the answer in disguise
#                                                  -> helps AID far more
#
# The corpus already hints at leakage before any model is trained:
# corr(insulin over the next 2 h, glucose CHANGE over the next 2 h) is +0.183
# for AID and +0.118 for non-AID. Positive is pharmacologically backwards --
# insulin lowers glucose -- and is only possible if the dose is a reaction.
#
# MAIN DESIGN: two models, identical in every way except
# data.future_modalities, both trained on ALL patients, then evaluated
# separately on AID and non-AID patients. Splitting at evaluation rather than
# at training keeps the comparison free of the 1,621-vs-264 training-size
# confound. The statistic is the INTERACTION:
#
#       gain_AID  -  gain_nonAID
#
# ROBUSTNESS (SUBGROUP=1): also train one model per therapy group, so the
# leakage claim does not rest on a model that saw 85% AID data. Expect wide
# intervals on the non-AID side -- 264 patients is thin, and the script says so
# rather than hiding it.
#
# Results land in results/futins_experiment/.
# ===========================================================================

set -euo pipefail
source "$(dirname "$0")/lib/common.sh"

NEED_MB="${NEED_MB:-14000}"
EPOCHS="${EPOCHS:-5}"
STAGE="${STAGE:-all}"
SUBGROUP="${SUBGROUP:-0}"
RUNS_DIR="${RUNS_DIR:-$REPO_ROOT/runs}"
OUT="${OUT:-$REPO_ROOT/results/futins_experiment}"
TAG="${TAG:-fx}"
SPLIT="${SPLIT:-test}"
SEED="${SEED:-0}"
BACKBONE="${BACKBONE:-model/mamba2}"
BATCH="${BATCH:-256}"
LR="${LR:-0.0008}"

# Everything the two arms share. The ONLY difference between them is
# data.future_modalities, which is the whole point of the experiment.
COMMON=(
  "seed=$SEED"
  "data.context=288"
  "data.horizons_min=[30,60,90,120]"
  "data.use_source_identity=false"
  "data.sampling.max_train_windows=null"
  "data.sampling.max_val_windows=150000"
  "data.sampling.max_test_windows=null"     # full test: the subgroups need it
  "data.split.scheme=official"
  "data.split.val_frac=0.12"
  "data.split.seed=$SEED"
  "data.internal_holdout.enabled=true"
  "data.loader.num_workers=8"
  "data.loader.batch_size=$BATCH"
  "model.fusion.name=film"
  "training.epochs=$EPOCHS"
  "training.patience=3"
  "training.min_delta=0.02"
  "training.lr=$LR"
  "training.predict_val=true"
)

mkdir -p "$OUT"

# ------------------------------------------------------------------ 1. train
if [[ "$STAGE" == "all" || "$STAGE" == "train" ]]; then
  gpu_autoconfig "$NEED_MB"
  queue_init
  log "arm A: CAUSAL  (future_modalities=[])"
  train "${TAG}_causal" "$BACKBONE" "${COMMON[@]}" "data.future_modalities=[]"
  log "arm B: FUTINS  (future_modalities=['insulin'])"
  train "${TAG}_futins" "$BACKBONE" "${COMMON[@]}" "data.future_modalities=['insulin']"

  if [[ "$SUBGROUP" == "1" ]]; then
    for grp in AID non-AID; do
      safe="${grp//-/}"
      log "robustness: $grp-only training, both arms"
      train "${TAG}_${safe}_causal" "$BACKBONE" "${COMMON[@]}" \
        "data.future_modalities=[]" "data.split.modality=$grp"
      train "${TAG}_${safe}_futins" "$BACKBONE" "${COMMON[@]}" \
        "data.future_modalities=['insulin']" "data.split.modality=$grp"
    done
  fi
  queue_wait
fi

[[ "$STAGE" == "train" ]] && { log "trained; re-run with STAGE=eval"; exit 0; }

# ------------------------------------------------------------- 2. main result
C="$RUNS_DIR/${TAG}_causal"; F="$RUNS_DIR/${TAG}_futins"
for d in "$C" "$F"; do
  [[ -f "$d/predictions_${SPLIT}.parquet" ]] \
    || die "$d has no predictions_${SPLIT}.parquet -- run STAGE=train first"
done

log "checking the two arms differ ONLY in future_modalities"
$PY "$REPO_ROOT/scripts/analysis/_verify_protocol.py" "$C" "$F" \
  > "$OUT/protocol_check.txt" 2>&1 || true
if grep -q "MISMATCH" "$OUT/protocol_check.txt"; then
  if [[ "$(grep -c 'MISMATCH' "$OUT/protocol_check.txt")" != "1" ]] \
     || ! grep -A1 MISMATCH "$OUT/protocol_check.txt" | grep -q future_modalities; then
    cat "$OUT/protocol_check.txt"
    die "the two arms differ in more than future_modalities -- the comparison
would not isolate the feature. See $OUT/protocol_check.txt"
  fi
fi
log "  OK: the arms are identical apart from the feature under test"

log "main experiment: stratified paired gain + interaction"
$PY -m metabonet_forecasting.evaluation.futins_experiment \
  --causal "$C" --futins "$F" --split "$SPLIT" \
  --out "$OUT/main" 2>&1 | tee "$OUT/main.txt"

# ---------------------------------------------------- 3. robustness (opt-in)
if [[ "$SUBGROUP" == "1" ]]; then
  for grp in AID nonAID; do
    CC="$RUNS_DIR/${TAG}_${grp}_causal"; FF="$RUNS_DIR/${TAG}_${grp}_futins"
    if [[ -f "$CC/predictions_${SPLIT}.parquet" && -f "$FF/predictions_${SPLIT}.parquet" ]]; then
      log "robustness: models trained on $grp only"
      $PY -m metabonet_forecasting.evaluation.futins_experiment \
        --causal "$CC" --futins "$FF" --split "$SPLIT" \
        --out "$OUT/subgroup_$grp" 2>&1 | tee "$OUT/subgroup_$grp.txt"
    else
      warn "$grp subgroup arm incomplete -- skipping that robustness check"
    fi
  done
fi

# ------------------------------------------------------------------ 4. wrap
{
  echo "# Future insulin: physiology or leakage?"
  echo
  echo "Backbone \`$BACKBONE\`, $EPOCHS epochs, split \`$SPLIT\`, seed $SEED."
  echo "Both arms identical except \`data.future_modalities\`."
  echo
  [[ -f "$OUT/main/REPORT.md" ]] && sed '1d' "$OUT/main/REPORT.md"
  if [[ "$SUBGROUP" == "1" ]]; then
    for grp in AID nonAID; do
      [[ -f "$OUT/subgroup_$grp/REPORT.md" ]] || continue
      echo; echo "## Robustness: trained on $grp only"; echo
      sed '1d' "$OUT/subgroup_$grp/REPORT.md"
    done
  fi
  echo
  echo "## How to read this"
  echo
  echo "- \`gain\` is zone-A points added by future insulin, paired per patient."
  echo "- The result is the **interaction**: a large positive"
  echo "  \`gain_AID - gain_nonAID\` means the feature works by leakage, because"
  echo "  a closed-loop pump computes the dose from the glucose being forecast."
  echo "- non-AID is only 264 patients against 1,621, so its interval is wide."
  echo "  A wide interval is not evidence of no effect."
} > "$OUT/SUMMARY.md"

log "done"
log "  main       : $OUT/main/REPORT.md"
log "  summary    : $OUT/SUMMARY.md"
[[ "$SUBGROUP" == "1" ]] && log "  robustness : $OUT/subgroup_*/REPORT.md"
