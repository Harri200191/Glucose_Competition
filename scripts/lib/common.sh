#!/usr/bin/env bash
# Shared helpers for every script in this directory.
#
# The important one is the GPU queue: the ablation grids are dozens of
# independent runs and this box has several GPUs, so `queue_run` dispatches jobs
# round-robin and blocks when all slots are busy. One log file per run, always.
#
# Source it, do not execute it:  source "$(dirname "$0")/lib/common.sh"

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PY="${PY:-python}"
RUNS_DIR="${RUNS_DIR:-$REPO_ROOT/runs}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/$(date +%Y%m%d)}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/results}"
STORE="${STORE:-$REPO_ROOT/artifacts/store/main}"

# GPUs to use, comma separated. Default: every visible device.
GPUS="${GPUS:-$(${PY} -c 'import torch;print(",".join(str(i) for i in range(torch.cuda.device_count())) or "")' 2>/dev/null || echo "")}"
# How many runs share one GPU. Raise only if the models are small.
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
# Extra --set overrides applied to every run, e.g. EXTRA_SET="training.epochs=3"
EXTRA_SET="${EXTRA_SET:-}"
# DRY_RUN=1 prints the commands instead of running them.
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$RUNS_DIR" "$LOG_DIR" "$RESULTS_DIR"

log()  { printf '\033[1;34m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*"; }
warn() { printf '\033[1;33m[%s] WARN\033[0m %s\n' "$(date +%H:%M:%S)" "$*"; }
die()  { printf '\033[1;31m[%s] ERROR\033[0m %s\n' "$(date +%H:%M:%S)" "$*" >&2; exit 1; }

require_store() {
  [[ -f "$STORE/manifest.json" ]] || die \
    "no sequence store at $STORE. Build one first: scripts/01_prepare_data.sh"
}

# ------------------------------------------------------------------ GPU queue
_gpu_list=()
_slot_pids=()
_slot_gpu=()

queue_init() {
  _gpu_list=()
  if [[ -n "$GPUS" ]]; then
    IFS=',' read -r -a devices <<< "$GPUS"
    for g in "${devices[@]}"; do
      for ((j = 0; j < JOBS_PER_GPU; j++)); do _gpu_list+=("$g"); done
    done
  else
    _gpu_list=("cpu")
  fi
  _slot_pids=()
  _slot_gpu=()
  log "queue: ${#_gpu_list[@]} slots on GPUs [${GPUS:-cpu}]"
}

_wait_for_slot() {
  while (( ${#_slot_pids[@]} >= ${#_gpu_list[@]} )); do
    local still_pids=() still_gpu=()
    for i in "${!_slot_pids[@]}"; do
      if kill -0 "${_slot_pids[$i]}" 2>/dev/null; then
        still_pids+=("${_slot_pids[$i]}"); still_gpu+=("${_slot_gpu[$i]}")
      else
        wait "${_slot_pids[$i]}" || warn "job ${_slot_pids[$i]} exited non-zero (see logs)"
      fi
    done
    _slot_pids=("${still_pids[@]:-}"); _slot_gpu=("${still_gpu[@]:-}")
    [[ -z "${_slot_pids[0]:-}" ]] && _slot_pids=() && _slot_gpu=()
    (( ${#_slot_pids[@]} >= ${#_gpu_list[@]} )) && sleep 5
  done
}

_free_gpu() {
  for g in "${_gpu_list[@]}"; do
    local used=0
    for busy in "${_slot_gpu[@]:-}"; do [[ "$busy" == "$g" ]] && used=$((used + 1)); done
    local capacity=0
    for slot in "${_gpu_list[@]}"; do [[ "$slot" == "$g" ]] && capacity=$((capacity + 1)); done
    (( used < capacity )) && { echo "$g"; return; }
  done
  echo "${_gpu_list[0]}"
}

# queue_run <run_name> <command...>
queue_run() {
  local name="$1"; shift
  local logfile="$LOG_DIR/${name}.log"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "GPU=? $* > $logfile"
    return
  fi
  _wait_for_slot
  local gpu; gpu="$(_free_gpu)"
  log "start $name on GPU $gpu -> $logfile"
  (
    if [[ "$gpu" != "cpu" ]]; then export CUDA_VISIBLE_DEVICES="$gpu"; fi
    "$@" > "$logfile" 2>&1
  ) &
  _slot_pids+=("$!")
  _slot_gpu+=("$gpu")
}

queue_wait() {
  local failed=0
  for pid in "${_slot_pids[@]:-}"; do
    [[ -z "$pid" ]] && continue
    wait "$pid" || { warn "job $pid failed"; failed=$((failed + 1)); }
  done
  _slot_pids=(); _slot_gpu=()
  (( failed > 0 )) && warn "$failed job(s) failed; check $LOG_DIR"
  log "all queued jobs finished"
}

# train <run_name> <config> [extra --set args...]
train() {
  local name="$1" config="$2"; shift 2
  if [[ -f "$RUNS_DIR/$name/metrics.json" && "${SKIP_EXISTING:-1}" == "1" ]]; then
    log "skip $name (already scored; SKIP_EXISTING=0 to force)"
    return
  fi
  queue_run "$name" $PY -m metabonet_forecasting.training.train \
    --config "$config" --name "$name" --runs-dir "$RUNS_DIR" \
    --set "data.store=$STORE" $EXTRA_SET "$@"
}

# baseline <run_name> <config> <model> [extra --set args...]
baseline() {
  local name="$1" config="$2" model="$3"; shift 3
  if [[ -f "$RUNS_DIR/$name/metrics.json" && "${SKIP_EXISTING:-1}" == "1" ]]; then
    log "skip $name (already scored)"
    return
  fi
  queue_run "$name" $PY -m metabonet_forecasting.baselines.run_baselines \
    --config "$config" --model "$model" --name "$name" --runs-dir "$RUNS_DIR" \
    --set "data.store=$STORE" $EXTRA_SET "$@"
}

summarise() {
  log "aggregating $RUNS_DIR -> $RESULTS_DIR"
  $PY -m metabonet_forecasting.evaluation.aggregate "$RUNS_DIR" \
    --out "$RESULTS_DIR" --pattern "${1:-*}" || warn "nothing to aggregate yet"
}
