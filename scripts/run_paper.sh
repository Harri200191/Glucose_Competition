#!/usr/bin/env bash
# The frozen CAUSAL protocol for the paper. Point 3 of the review, end to end.
#
#   scripts/run_paper.sh                       # everything
#   EPOCHS=8 scripts/run_paper.sh              # shorter first pass
#   STAGE=train scripts/run_paper.sh           # models only
#   STAGE=eval  scripts/run_paper.sh           # scoring + comparison only
#   STAGE=fm    scripts/run_paper.sh           # just the Chronos-2 arm
#   GPUS=0,3 scripts/run_paper.sh              # override GPU auto-detection
#   SPLITS="test internal_holdout" scripts/run_paper.sh
#
# ===========================================================================
# THE PROTOCOL. Do not edit these mid-study; that is what "frozen" means.
#
#   1. CAUSAL. data.future_modalities=[] for every model, no exceptions. The
#      pump reacts to glucose, so future insulin is the answer arriving by the
#      back door: in this corpus corr(insulin next 2h, glucose CHANGE next 2h)
#      is +0.183 for AID patients -- positive, i.e. pharmacologically backwards,
#      which is only possible if the dose is a reaction to the rise. See
#      docs/PAPER_PROTOCOL_EXPLAINED.md.
#   2. NO source identity. It is a study label, not a patient characteristic.
#   3. FULL training data. max_train_windows=null.
#   4. IDENTICAL splits. One scheme, one val_frac, one seed -- and verified
#      after training by reading every run's own resolved config, so the script
#      fails loudly instead of quietly comparing models trained on different
#      data.
#   5. IDENTICAL budget. Same epochs, patience and schedule for every model, so
#      a difference is the architecture and not the compute.
#   6. Chronos-2 zero-shot as the foundation-model comparator, on the same
#      anchors, also causal (no covariates).
#
# Everything lands in results/paper/. Nothing here writes a submission.
# ===========================================================================

set -euo pipefail
source "$(dirname "$0")/lib/common.sh"

# GPUs are auto-detected: gpu_autoconfig (scripts/lib/common.sh) reads
# nvidia-smi, keeps the cards with at least NEED_MB free right now, and reports
# which ones it skipped and why. Set GPUS=0,3 to override; the queue still does
# admission control and waits for room either way.
# NEED_MB is sized for the heaviest member: Mamba-2 at 288-step context and
# batch 256 holds ~10.5 GiB, so 14 GiB admits it with headroom.
NEED_MB="${NEED_MB:-14000}"

EPOCHS="${EPOCHS:-20}"
STAGE="${STAGE:-all}"
RUNS_DIR="${RUNS_DIR:-$REPO_ROOT/runs}"
OUT="${OUT:-$REPO_ROOT/results/paper}"
TAG="${TAG:-p}"
MARGIN="${MARGIN:-2.0}"          # equivalence margin, DTS zone-A points
REFERENCE="${REFERENCE:-${TAG}_tcn}"
SPLITS="${SPLITS:-test internal_holdout}"
SEED="${SEED:-0}"

# ---- the frozen protocol, applied identically to every trained model -------
PROTOCOL=(
  "seed=$SEED"
  "data.context=288"
  "data.horizons_min=[30,60,90,120]"
  "data.future_modalities=[]"            # (1) CAUSAL -- the whole point
  "data.use_source_identity=false"       # (2)
  "data.sampling.max_train_windows=null" # (3) full data
  "data.sampling.max_val_windows=150000"
  "data.sampling.max_test_windows=400000"
  "data.split.scheme=official"           # (4) identical splits
  "data.split.val_frac=0.12"
  "data.split.seed=$SEED"
  "data.internal_holdout.enabled=true"
  "data.loader.num_workers=8"
  "model.fusion.name=film"
  "training.epochs=$EPOCHS"              # (5) identical budget
  "training.patience=3"
  "training.min_delta=0.02"
  "training.predict_val=true"
)

# name:config:batch:lr -- batch and lr differ only because the architectures
# differ in memory footprint and stable step size. Everything else is identical.
MODELS=(
  "${TAG}_tcn:model/tcn:512:0.0012"
  "${TAG}_mamba2:model/mamba2:256:0.0008"
  "${TAG}_moderntcn:model/moderntcn:512:0.0014"
  "${TAG}_transformerpp:model/transformerpp:512:0.0007"
)
FM_NAME="${TAG}_chronos2_zs"

mkdir -p "$OUT"

# ------------------------------------------------------------------ 1. train
if [[ "$STAGE" == "all" || "$STAGE" == "train" ]]; then
  gpu_autoconfig "$NEED_MB"
  queue_init
  for spec in "${MODELS[@]}"; do
    IFS=: read -r name cfg bs lr <<< "$spec"
    log "queueing $name  ($cfg, batch $bs, lr $lr, CAUSAL)"
    train "$name" "$cfg" "${PROTOCOL[@]}" \
      "data.loader.batch_size=$bs" "training.lr=$lr"
  done
  queue_wait
fi

# ------------------------------------------- 2. Chronos-2, the FM comparator
if [[ "$STAGE" == "all" || "$STAGE" == "train" || "$STAGE" == "fm" ]]; then
  if $PY -c "import chronos" 2>/dev/null; then
    gpu_autoconfig "$NEED_MB"
    FM_GPU="${GPUS%%,*}"
    for sp in $SPLITS; do
      log "Chronos-2 zero-shot on $sp (causal: no covariates) on GPU $FM_GPU"
      $PY -m metabonet_forecasting.fm.zeroshot \
        --model chronos2 --name "$FM_NAME" --runs-dir "$RUNS_DIR" \
        --split "$sp" --gpu "$FM_GPU" \
        --set "data.store=$STORE" "data.internal_holdout.enabled=true" \
              "data.split.seed=$SEED" \
        || warn "Chronos-2 failed on $sp"
    done
  else
    warn "the 'chronos' package is not installed, so the foundation-model"
    warn "comparator is SKIPPED. Everything else still runs. To include it:"
    warn "    pip install 'chronos-forecasting>=2.0'"
    warn "then re-run with STAGE=fm. The paper wants this arm -- it answers"
    warn "'would an off-the-shelf model have done just as well?'"
  fi
fi

if [[ "$STAGE" == "train" ]]; then
  log "training done; re-run with STAGE=eval"; exit 0
fi

# --------------------------------------- 3. collect + VERIFY identical splits
present=()
for spec in "${MODELS[@]}"; do
  IFS=: read -r name _ _ _ <<< "$spec"
  if [[ -d "$RUNS_DIR/$name" ]]; then present+=("$RUNS_DIR/$name")
  else warn "$name missing -- excluded"; fi
done
[[ -d "$RUNS_DIR/$FM_NAME" ]] && present+=("$RUNS_DIR/$FM_NAME")
[[ ${#present[@]} -ge 2 ]] || die "need at least two runs; check the logs"

log "verifying every run used the identical split and protocol"
if ! $PY "$REPO_ROOT/scripts/analysis/_verify_protocol.py" "${present[@]}" \
     | tee "$OUT/protocol_check.txt"; then
  die "protocol check FAILED -- see $OUT/protocol_check.txt. These runs are not
comparable. Retrain the odd ones out, or drop them from the comparison."
fi

# ------------------------------------------------------- 4. score each split
for sp in $SPLITS; do
  log "=== scoring split: $sp ==="
  for r in "${present[@]}"; do
    [[ -f "$r/predictions_${sp}.parquet" ]] || { warn "$(basename "$r"): no $sp"; continue; }
    $PY "$REPO_ROOT/scripts/analysis/_score_split.py" "$r" "$sp" >/dev/null \
      || warn "scoring $(basename "$r") [$sp] failed"
  done

  log "paired comparison on $sp (reference $REFERENCE, margin +/-$MARGIN)"
  $PY -m metabonet_forecasting.evaluation.compare "${present[@]}" \
    --split "$sp" --reference "$REFERENCE" --margin "$MARGIN" \
    --out "$OUT/compare_$sp" 2>&1 | tee "$OUT/compare_$sp.txt"

  if [[ "$sp" == "internal_holdout" ]]; then
    log "annual-shaped estimate per run (strict competition draws)"
    : > "$OUT/annual_estimates.txt"
    for r in "${present[@]}"; do
      [[ -f "$r/predictions_${sp}.parquet" ]] || continue
      { echo "### $(basename "$r")"
        $PY -m metabonet_forecasting.evaluation.annual_estimate "$r" \
          --split "$sp" --strict 2>/dev/null || echo "  (failed)"
        echo; } >> "$OUT/annual_estimates.txt"
    done
    cat "$OUT/annual_estimates.txt"
  fi
done

# ------------------------------------------------------------ 5. diagnostics
BEST="$REFERENCE"
if [[ -f "$OUT/compare_internal_holdout/leaderboard.csv" ]]; then
  BEST=$($PY -c "
import pandas as pd, sys
print(pd.read_csv(sys.argv[1]).iloc[0]['run'])" \
    "$OUT/compare_internal_holdout/leaderboard.csv" 2>/dev/null) || BEST="$REFERENCE"
fi
log "failure diagnostics: reference ($REFERENCE) and best ($BEST)"
for r in "$REFERENCE" "$BEST"; do
  [[ -d "$RUNS_DIR/$r" ]] || continue
  bash "$REPO_ROOT/scripts/38_diagnose.sh" "$RUNS_DIR/$r" internal_holdout || true
done

# ---------------------------------------------------------------- 6. summary
$PY "$REPO_ROOT/scripts/analysis/_paper_summary.py" \
  "$OUT" "$RUNS_DIR" "$REFERENCE" "$MARGIN" "$SPLITS" > "$OUT/SUMMARY.md" \
  || warn "summary generation failed"

log "done -- everything under $OUT"
log "  protocol check : $OUT/protocol_check.txt"
for sp in $SPLITS; do log "  $sp report     : $OUT/compare_$sp/REPORT.md"; done
log "  annual-shaped  : $OUT/annual_estimates.txt"
log "  SUMMARY        : $OUT/SUMMARY.md"
log "  figures        : $RUNS_DIR/$BEST/figures/internal_holdout/"
