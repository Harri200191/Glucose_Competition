#!/usr/bin/env bash
# Ablations around the winning recipe (FiLM + future insulin) on the NEW
# backbones, in priority order:
#   1. mambaformer + horizon_cross head  — isolates the new head vs the pooled
#      mambaformer already running
#   2. fusion ablation: raw_concat twins of moderntcn / transformerpp
#      (does FiLM's advantage transfer to the new architectures?)
#   3. futboth twins (insulin + carbs): Haris's absolute best internally was
#      futboth; carbs are absent from the annual holdout but present in the
#      live-leaderboard corpus, so this variant matters for the live board.
# Batch 256-512 so runs fit the partial slots the shared box actually offers.
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
GPU_FREE_MB="${GPU_FREE_MB:-11000}"
source "$(dirname "$0")/lib/common.sh"
queue_init

EPOCHS="${EPOCHS:-20}"
QUANTS="[0.05,0.1,0.25,0.5,0.75,0.9,0.95]"
COMMON_SET="training.epochs=$EPOCHS training.patience=3 training.min_delta=0.02
            training.predict_val=true
            data.context=288 seed=0
            data.sampling.max_train_windows=null
            data.loader.num_workers=8
            model.head.quantiles=$QUANTS
            training.loss.name=hybrid training.loss.quantiles=$QUANTS
            training.loss.alpha=0.5"
FUTINS="data.future_modalities=['insulin']"
FUTBOTH="data.future_modalities=['insulin','carbs']"

train f_mambaformer_cross_futins model/mambaformer $COMMON_SET $FUTINS \
      model.fusion.name=film model.head.name=horizon_cross \
      data.loader.batch_size=256 training.lr=0.001
train f_moderntcn_rawconcat_futins model/moderntcn $COMMON_SET $FUTINS \
      model.fusion.name=raw_concat \
      data.loader.batch_size=512 training.lr=0.0014
train f_transformerpp_rawconcat_futins model/transformerpp $COMMON_SET $FUTINS \
      model.fusion.name=raw_concat \
      data.loader.batch_size=512 training.lr=0.0007
train f_moderntcn_futboth model/moderntcn $COMMON_SET $FUTBOTH \
      model.fusion.name=film \
      data.loader.batch_size=512 training.lr=0.0014
train f_transformerpp_futboth model/transformerpp $COMMON_SET $FUTBOTH \
      model.fusion.name=film \
      data.loader.batch_size=512 training.lr=0.0007

queue_wait
summarise
