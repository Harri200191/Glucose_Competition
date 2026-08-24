#!/usr/bin/env bash
# The two targeted bets, full scale, queued behind whatever is running:
#   1. mambaformer + pooled head    — hybrid upgrade of the incumbent backbone
#   2. mambaformer + horizon_cross  — per-horizon queries cross-attending over
#      history and the future insulin schedule (the problem-shaped head)
# The pair isolates the head's contribution.
FUTURE="${FUTURE:-insulin}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
GPU_FREE_MB="${GPU_FREE_MB:-22000}"
source "$(dirname "$0")/lib/common.sh"
queue_init

EPOCHS="${EPOCHS:-20}"
QUANTS="[0.05,0.1,0.25,0.5,0.75,0.9,0.95]"
COMMON_SET="training.epochs=$EPOCHS training.patience=3 training.min_delta=0.02
            training.predict_val=true
            data.context=288 model.fusion.name=film seed=0
            data.sampling.max_train_windows=null
            data.loader.num_workers=10
            data.loader.batch_size=512 training.lr=0.001
            model.head.quantiles=$QUANTS
            training.loss.name=hybrid training.loss.quantiles=$QUANTS
            training.loss.alpha=0.5"

train f_mambaformer       model/mambaformer $COMMON_SET
train f_mambaformer_cross model/mambaformer $COMMON_SET model.head.name=horizon_cross

queue_wait
summarise
