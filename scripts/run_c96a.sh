#!/usr/bin/env bash
# ANNUAL-BOARD SPECIALISTS: warm-start the zone_t c96t models on an IOBP2-heavy mix.
#
# THE MEASUREMENT (105-annual, zone_t k=3, 3 seeds, 89 UNSEEN IOBP2 patients):
#     IOBP2 excluded   79.755
#     5.5% natural     92.229   +12.47   <- what c96t already has
#     20%              93.648   +13.89
#     55%              94.424   +14.67   <- +2.2 over natural, curve still rising
# Seed spreads 0.14-0.30; every step clears the 0.6 band. The annual template's
# fingerprint matches IOBP2 (Dexcom G6 + iLet, median age 17, 85/94-day spans).
# COST: livemix falls +0.63 -> +0.02, so these are ANNUAL-ONLY. Never ship to live.
#
# 1,245,347 IOBP2 anchors at ctx1152 / 0.55 = 2.26M windows = one epoch with no
# duplication, ~2-3 h at ctx1152. 3 epochs. mix.target needs an explicit cap
# (a null cap cannot express a mix -- run_loss.sh dies on it for that reason).
# internal_holdout.enabled=false: the parents already trained on the 89, so no
# clean holdout number exists for these; the evidence is the screen above.
set -euo pipefail
source "$(dirname "$0")/lib/common.sh"
STORE="$REPO_ROOT/artifacts/store/ctx1152"
IOBP2_55="{'IOBP2':55.0,'Loop':27.0,'ReplaceBG':10.0,'PEDAP':4.0,'BrisT1D':1.5,'CTR3':1.0,'HUPA-UCM':0.8,'AZT1D':0.7}"
Q="[0.05,0.1,0.25,0.5,0.75,0.9,0.95]"
# common.sh has ALREADY set GPU_FREE_MB=12000 by the time this line runs, so a
# ${GPU_FREE_MB:-30000} default silently keeps 12000 -- which is how c96a_mambaformer
# landed on a card still running inference and OOMed. Assign unconditionally.
GPU_FREE_MB="${FT_FREE_MB:-30000}"
JOBS_PER_GPU=1
queue_init
ft() {  # ft <backbone> <init snapshot> <bs> <lr> <k> [extra...]
  local b=$1 init=$2 bs=$3 lr=$4 k=$5; shift 5
  # train() appends everything after --set, which would swallow --init-from as
  # a config override. queue_run directly, with --init-from BEFORE --set.
  [[ -f "$RUNS_DIR/c96a_$b/metrics.json" ]] && { log "skip c96a_$b (scored)"; return; }
  queue_run "c96a_$b" $PY -m metabonet_forecasting.training.train \
    --config "model/$b" --name "c96a_$b" --runs-dir "$RUNS_DIR" \
    --init-from "$RUNS_DIR/$init/best.pt" --set \
    "seed=0" "data.store=$STORE" "data.context=1152" "data.target_mode=log_ratio" \
    "data.future_modalities=['insulin']" "data.use_source_identity=false" \
    "data.split.scheme=official" "data.split.val_frac=0.12" "data.split.seed=0" \
    "data.internal_holdout.enabled=false" "data.sampling.max_train_windows=2260000" \
    "data.sampling.max_val_windows=150000" "data.sampling.mix.by=cohort" \
    "data.sampling.mix.target=$IOBP2_55" "data.loader.batch_size=$bs" "data.loader.num_workers=8" \
    "model.d_model=128" "model.fusion.name=film" "model.head.name=horizon_cross" \
    "model.head.chain=dist" "model.head.quantiles=$Q" "model.head.law.name=student_t" \
    "model.head.law.k=$k" "training.loss.name=zone_t" "training.loss.horizon_weights=[1,1,1.5,2]" \
    "training.epochs=3" "training.patience=3" "training.min_delta=0.0" \
    "training.select_metric=val_zone_a" "training.warmup_steps=1000" \
    "training.lr=$lr" "training.predict_val=true" "$@"
}
ARMS="${ARMS:-transformerpp mamba2 mambaformer}"
[[ " $ARMS " == *" transformerpp "* ]] && ft transformerpp c96t_transformerpp_e9 144 0.00023 3
[[ " $ARMS " == *" mamba2 "* ]]        && ft mamba2        c96t_mamba2_e9        72  0.00013 1
[[ " $ARMS " == *" mambaformer "* ]]   && ft mambaformer   c96t_mambaformer_e7   72  0.00017 1
queue_wait
