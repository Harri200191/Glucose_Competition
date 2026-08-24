#!/usr/bin/env bash
# The candidate runs of the new approach, at FULL scale and FULL VRAM:
# all 12.1M training windows, batch sized to use the card (~4x the default,
# learning rate sqrt-scaled to match), up to 20 epochs with early stopping
# (patience 3, min_delta 0.02 mg/dL) so training runs exactly as long as the
# loss keeps buying something. Future insulin + FiLM + quantile head + hybrid
# loss + validation predictions, as before.
#
# Scored at every horizon (30/60/90/120) automatically.
#
#   scripts/s4_full_scale.sh
FUTURE="${FUTURE:-insulin}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
GPU_FREE_MB="${GPU_FREE_MB:-24000}"     # big-batch runs need a mostly-free card
source "$(dirname "$0")/lib/common.sh"
queue_init

EPOCHS="${EPOCHS:-20}"
QUANTS="[0.05,0.1,0.25,0.5,0.75,0.9,0.95]"
COMMON_SET="training.epochs=$EPOCHS training.patience=3 training.min_delta=0.02
            training.predict_val=true
            data.context=288 model.fusion.name=film seed=0
            data.sampling.max_train_windows=null
            data.loader.num_workers=12
            model.head.quantiles=$QUANTS
            training.loss.name=hybrid training.loss.quantiles=$QUANTS
            training.loss.alpha=0.5"

# batch 256 -> 1024 is a 4x step: lr scales by sqrt(4)=2. xLSTM materialises
# fp32 [B, h, L, L] decay matrices, so it takes the 2x step instead.
train f_moderntcn     model/moderntcn     $COMMON_SET \
      data.loader.batch_size=1024 training.lr=0.002
train f_transformerpp model/transformerpp $COMMON_SET \
      data.loader.batch_size=1024 training.lr=0.001
train f_xlstm         model/xlstm         $COMMON_SET \
      data.loader.batch_size=512 training.lr=0.0007

queue_wait
summarise
