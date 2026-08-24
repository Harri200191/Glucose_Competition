#!/usr/bin/env bash
# Chronos-2 LoRA fine-tune on the full corpus (insulin as future-known
# covariate), then evaluation on the official test split with the same stride-8
# thinning as the zero-shot sweep so the numbers are directly comparable.
#
# Waits for a GPU with enough free memory before starting, so it can be queued
# behind the training runs and fire whenever one of them early-stops.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/tayyibul/anaconda3/envs/glucose/bin/python
STORE=${STORE:-/home/hrehman5/glucose/artifacts/store/main}
FREE_MB=${FREE_MB:-14000}
NAME=${NAME:-c2ft_lora}

while true; do
  gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | awk -F', ' -v m="$FREE_MB" '$2 >= m {print $1; exit}')
  [[ -n "${gpu:-}" ]] && break
  sleep 120
done
echo "$(date +%H:%M) using GPU $gpu"

$PY -m metabonet_forecasting.fm.finetune \
  --name "$NAME" --gpu "$gpu" --mode lora \
  --steps "${STEPS:-8000}" --batch "${BATCH:-128}" --context-length 1024 \
  --runs-dir runs --set "data.store=$STORE" data.loader.num_workers=0 \
  || { echo "FINETUNE FAILED"; exit 1; }

$PY -m metabonet_forecasting.fm.zeroshot \
  --model chronos2 --model-id "runs/$NAME/model" \
  --covariates insulin --name "${NAME}_eval" --gpu "$gpu" \
  --runs-dir runs --stride 8 --batch 512 \
  --set "data.store=$STORE" data.loader.num_workers=0 \
  || echo "EVAL FAILED"
echo "C2FT DONE"
