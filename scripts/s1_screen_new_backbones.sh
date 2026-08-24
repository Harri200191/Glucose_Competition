#!/usr/bin/env bash
# Screening sweep for the NEW state-of-the-art backbones (moderntcn, xlstm,
# transformerpp): rank them cheaply — 2M-window cap, 8 epochs, early stopping —
# before spending full-data GPU-days on the winners.
#
# All runs use the competition submission recipe (FUTURE=insulin, FiLM front
# end, context 288). Reference at the same budget: Haris's
# e9_mamba2_film_s0_futins scored RMSE60 24.758 / DTS-A 82.88 on the 2M cap.
#
#   GPUS="0,2" scripts/s1_screen_new_backbones.sh
FUTURE="${FUTURE:-insulin}"
source "$(dirname "$0")/lib/common.sh"
queue_init

EPOCHS="${EPOCHS:-8}"
COMMON_SET="training.epochs=$EPOCHS training.patience=2 training.min_delta=0.02
            data.context=288 model.fusion.name=film seed=0"

train s1_moderntcn     model/moderntcn     $COMMON_SET
train s1_xlstm         model/xlstm         $COMMON_SET
train s1_transformerpp model/transformerpp $COMMON_SET

queue_wait
summarise
