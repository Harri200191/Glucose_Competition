#!/usr/bin/env bash
# The frozen CAUSAL protocol for the paper. Point 3 of the review, end to end.
#
#   ./run_paper.sh                       # everything
#   GPUS=0,7 ./run_paper.sh              # restrict to free cards
#   EPOCHS=8 ./run_paper.sh              # shorter first pass
#   STAGE=train ./run_paper.sh           # models only
#   STAGE=eval  ./run_paper.sh           # scoring + comparison only
#   SPLITS="test internal_holdout" ./run_paper.sh
#
# ===========================================================================
# THE PROTOCOL. Do not edit these mid-study; that is what "frozen" means.
#
#   1. CAUSAL. data.future_modalities=[] for every model, no exceptions. The
#      pump reacts to glucose, so future insulin is the answer arriving by the
#      back door: in this corpus corr(insulin next 2h, glucose CHANGE next 2h)
#      is +0.183 for AID patients -- positive, i.e. pharmacologically backwards,
#      which is only possible if the dose is a reaction. See
#      docs/PAPER_PROTOCOL_EXPLAINED.md.
#   2. NO source identity. It is a study label, not a patient characteristic.
#   3. FULL training data. max_train_windows=null.
#   4. IDENTICAL splits. One scheme, one val_frac, one seed, verified after
#      training by comparing every run's resolved config -- the script fails
#      loudly rather than silently comparing models trained on different data.
#   5. IDENTICAL budget. Same epochs, patience and schedule for every model, so
#      a difference is the architecture and not the compute.
#   6. Chronos-2 zero-shot as the foundation-model comparator, scored on the
#      same anchors. Also causal: no covariates.
#
# Everything lands in results/paper/. Nothing here writes a submission.
# ===========================================================================

set -euo pipefail
source "$(dirname "$0")/scripts/lib/common.sh"

GPUS="${GPUS:-0,7}"
GPU_FREE_MB="${GPU_FREE_MB:-12000}"
EPOCHS="${EPOCHS:-20}"
STAGE="${STAGE:-all}"
RUNS_DIR="${RUNS_DIR:-$REPO_ROOT/runs}"
OUT="${OUT:-$REPO_ROOT/results/paper}"
TAG="${TAG:-p}"
MARGIN="${MARGIN:-2.0}"          # equivalence margin, DTS zone-A points
REFERENCE="${REFERENCE:-${TAG}_tcn}"
SPLITS="${SPLITS:-test internal_holdout}"
SEED="${SEED:-0}"

# ---- the frozen protocol, applied to every trained model -------------------
PROTOCOL=(
  "seed=$SEED"
  "data.context=288"
  "data.horizons_min=[30,60,90,120]"
  "data.future_modalities=[]"            # (1) CAUSAL
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

# name:config:batch:lr   -- batch and lr differ only because the models differ
# in memory footprint and stable step size; everything else is identical.
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
    for sp in $SPLITS; do
      log "Chronos-2 zero-shot on $sp (causal: no covariates)"
      $PY -m metabonet_forecasting.fm.zeroshot \
        --model chronos2 --name "$FM_NAME" --runs-dir "$RUNS_DIR" \
        --split "$sp" --gpu "${GPUS%%,*}" \
        --set "data.store=$STORE" "data.internal_holdout.enabled=true" \
              "data.split.seed=$SEED" \
        || warn "Chronos-2 failed on $sp"
    done
  else
    warn "the 'chronos' package is not installed, so the foundation-model"
    warn "comparator is SKIPPED. Everything else still runs. To include it:"
    warn "    pip install 'chronos-forecasting>=2.0'"
    warn "and re-run with STAGE=fm. The paper needs this arm -- it is the"
    warn "answer to 'would an off-the-shelf model have done just as well?'"
  fi
fi

if [[ "$STAGE" == "train" ]]; then
  log "training done; re-run with STAGE=eval"; exit 0
fi

# --------------------------------------- 3. collect + VERIFY identical splits
present=(); names=()
for spec in "${MODELS[@]}"; do
  IFS=: read -r name _ _ _ <<< "$spec"
  [[ -d "$RUNS_DIR/$name" ]] && { present+=("$RUNS_DIR/$name"); names+=("$name"); } \
    || warn "$name missing"
done
[[ -d "$RUNS_DIR/$FM_NAME" ]] && { present+=("$RUNS_DIR/$FM_NAME"); names+=("$FM_NAME"); }
[[ ${#present[@]} -ge 2 ]] || die "need at least two runs; check the logs"

log "verifying every run used the identical split and protocol"
$PY "$REPO_ROOT/scripts/analysis/_verify_protocol.py" "${present[@]}" \
  | tee "$OUT/protocol_check.txt"

# ------------------------------------------------------- 4. score every split
for sp in $SPLITS; do
  log "=== scoring split: $sp ==="
  for r in "${present[@]}"; do
    [[ -f "$r/predictions_${sp}.parquet" ]] || { warn "$(basename "$r") has no $sp"; continue; }
    $PY -m metabonet_forecasting.evaluation.evaluate "$r" --split "$sp" >/dev/null 2>&1 \
      || $PY - "$r" "$sp" <<'PYEOF'
import sys
from metabonet_forecasting.evaluation.evaluate import score_run
score_run(sys.argv[1], split=sys.argv[2])
PYEOF
  done

  log "paired comparison on $sp (reference: $REFERENCE, margin +/-$MARGIN)"
  $PY -m metabonet_forecasting.evaluation.compare "${present[@]}" \
    --split "$sp" --reference "$REFERENCE" --margin "$MARGIN" \
    --out "$OUT/compare_$sp" | tee "$OUT/compare_$sp.txt"

  if [[ "$sp" == "internal_holdout" ]]; then
    log "annual-shaped estimate for each run"
    : > "$OUT/annual_estimates.txt"
    for r in "${present[@]}"; do
      [[ -f "$r/predictions_${sp}.parquet" ]] || continue
      { echo "### $(basename "$r")";
        $PY -m metabonet_forecasting.evaluation.annual_estimate "$r" \
          --split "$sp" --strict || true; echo; } >> "$OUT/annual_estimates.txt"
    done
    cat "$OUT/annual_estimates.txt"
  fi
done

# ------------------------------------------------------------ 5. diagnostics
log "failure diagnostics for the reference and the best run"
BEST=$($PY - "$OUT/compare_internal_holdout/leaderboard.csv" <<'PYEOF'
import sys
import pandas as pd
print(pd.read_csv(sys.argv[1]).iloc[0]["run"])
PYEOF
) || BEST="$REFERENCE"
for r in "$REFERENCE" "$BEST"; do
  [[ -d "$RUNS_DIR/$r" ]] || continue
  bash "$REPO_ROOT/scripts/38_diagnose.sh" "$RUNS_DIR/$r" internal_holdout || true
done

# ---------------------------------------------------------------- 6. wrap up
$PY "$REPO_ROOT/scripts/analysis/_paper_summary.py" "$OUT" "$RUNS_DIR" \
  "$REFERENCE" "$MARGIN" "$SPLITS" > "$OUT/SUMMARY.md" || warn "summary failed"

log "done"
log "  protocol check : $OUT/protocol_check.txt"
for sp in $SPLITS; do
  log "  $sp: $OUT/compare_$sp/REPORT.md"
done
log "  annual-shaped  : $OUT/annual_estimates.txt"
log "  summary        : $OUT/SUMMARY.md"
log "  figures        : $RUNS_DIR/$BEST/figures/internal_holdout/"
