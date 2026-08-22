#!/usr/bin/env bash
# The candidate runs of the new approach, at FULL scale: all 12.1M training
# windows, every GPU the admission-control queue can get, future insulin, FiLM
# front end, quantile head + hybrid loss (for the DTS decision layer), and
# validation predictions saved (for calibration and ensemble weighting).
#
# Baselines to beat (Haris): causal mamba2+film 33.31 RMSE60;
# futins mamba2+film 24.49 RMSE60 / 83.4 DTS-A (internal test).
#
#   scripts/s3_full_new_backbones.sh              # all 8 GPUs by default
#   GPUS="0,2" scripts/s3_full_new_backbones.sh
FUTURE="${FUTURE:-insulin}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
source "$(dirname "$0")/lib/common.sh"
queue_init

EPOCHS="${EPOCHS:-8}"
QUANTS="[0.05,0.1,0.25,0.5,0.75,0.9,0.95]"
COMMON_SET="training.epochs=$EPOCHS training.patience=3 training.min_delta=0.02
            training.predict_val=true
            data.context=288 model.fusion.name=film seed=0
            data.sampling.max_train_windows=null
            model.head.quantiles=$QUANTS
            training.loss.name=hybrid training.loss.quantiles=$QUANTS
            training.loss.alpha=0.5"

train f_moderntcn     model/moderntcn     $COMMON_SET
train f_transformerpp model/transformerpp $COMMON_SET
train f_xlstm         model/xlstm         $COMMON_SET

queue_wait
summarise
