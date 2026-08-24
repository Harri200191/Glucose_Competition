#!/usr/bin/env bash
# Shared helpers for every script in this directory.
#
# The important one is the GPU queue. The ablation grids are dozens of
# independent runs and this box has 8 GPUs shared with ~48 other users, so
# `queue_run` does admission control rather than plain round-robin: a job starts
# only on a device that is *actually* free right now, and otherwise the queue
# blocks. Two conditions have to hold together:
#
#   1. we are running fewer than JOBS_PER_GPU of our own jobs on that device;
#   2. nvidia-smi reports at least GPU_FREE_MB MiB free on it.
#
# (2) is the one that matters here — another user's job is invisible to (1), and
# dispatching on top of it is how a sweep dies at 3 a.m. with CUDA OOM. When no
# device qualifies the queue polls every GPU_POLL_S seconds until one does,
# whether that is because one of our jobs exited or because somebody else's did.
#
# A device that has just been handed a job is held back for GPU_WARMUP_S
# seconds: torch takes tens of seconds to allocate, so its memory does not show
# up in nvidia-smi yet and the same device would otherwise look free twice.
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

# GPUs to use, comma separated, in *physical* nvidia-smi index space -- which is
# also what CUDA_VISIBLE_DEVICES takes, so the memory we read and the device we
# hand the child can never drift apart. (Asking torch for the count instead
# returns 0..n-1 of whatever the parent shell already masked, which silently
# means a different card.) Default: whatever CUDA_VISIBLE_DEVICES already
# restricts us to, else every device on the box.
_nvidia_indices() {
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  # `|| true`: pipefail is on, and a driver hiccup here must degrade to "no GPU",
  # not abort every script that sources this file.
  nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null \
    | tr -d ' ' | paste -sd, - || true
}
GPUS_EXPLICIT="${GPUS:-}"
GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES:-$(_nvidia_indices)}}"
# How many of OUR runs share one GPU. Raise only if the models are small.
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
# Free memory a device must report before we put a job on it. Measured on the
# stage-E sweep: Mamba-2 at 288-step context, batch 256, holds 10.2-10.8 GiB, so
# 12 GiB admits a run with headroom and refuses a card that cannot take one.
# Raise it for a heavier model; check with
#   nvidia-smi --query-compute-apps=pid,used_memory --format=csv
GPU_FREE_MB="${GPU_FREE_MB:-12000}"
# How often to re-check when nothing qualifies, and how long to hold a device
# back after handing it a job so its allocation shows up in nvidia-smi first.
GPU_POLL_S="${GPU_POLL_S:-30}"
GPU_WARMUP_S="${GPU_WARMUP_S:-90}"

# gpu_autoconfig [need_mb] -- pick the cards that can actually take a job now.
#
# The queue below already refuses a device under GPU_FREE_MB and waits, so this
# is not about correctness; it is about not queueing eight jobs behind a box
# where someone else is using six of the cards, and about saying so out loud
# instead of appearing to hang. It narrows GPUS to the devices with room right
# now and leaves the queue to do admission control on those.
#
# An explicit GPUS= or CUDA_VISIBLE_DEVICES in the environment always wins: this
# only fills in the default.
gpu_autoconfig() {
  local need="${1:-$GPU_FREE_MB}"
  if [[ -n "${GPUS_EXPLICIT:-}" || -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    log "GPUs: using the ones you asked for [$GPUS] (need >= ${need} MiB)"
    GPU_FREE_MB="$need"; return 0
  fi
  command -v nvidia-smi >/dev/null 2>&1 || { warn "no nvidia-smi; CPU only"; return 0; }

  local fit=() tight=() idx used total free best=0
  while IFS=, read -r idx used total; do
    idx="${idx// /}"; used="${used// /}"; total="${total// /}"
    [[ -n "$idx" && -n "$total" ]] || continue
    free=$(( total - used ))
    (( free > best )) && best=$free
    if (( free >= need )); then fit+=("$idx"); else tight+=("${idx}:${free}MiB"); fi
  done < <(nvidia-smi --query-gpu=index,memory.used,memory.total              --format=csv,noheader,nounits 2>/dev/null || true)

  if (( ${#fit[@]} > 0 )); then
    GPUS="$(IFS=,; echo "${fit[*]}")"
    log "GPUs: ${#fit[@]} card(s) with >= ${need} MiB free -> [$GPUS]"
    (( ${#tight[@]} > 0 )) && log "      skipping busy: ${tight[*]}"
  else
    # Nothing fits. Keep every card and let the queue block: cards free up, and
    # exiting here would throw away an overnight run for a transient state.
    warn "no GPU currently has ${need} MiB free (best is ${best} MiB)."
    warn "keeping every card and waiting -- the queue admits jobs as they drain."
    warn "busy now: ${tight[*]:-none}"
  fi
  GPU_FREE_MB="$need"
}
# Give up after this many seconds of waiting. 0 = wait forever (the default:
# a sweep that queues behind other users is doing the right thing).
GPU_WAIT_MAX_S="${GPU_WAIT_MAX_S:-0}"
# Extra --set overrides applied to every run, e.g. EXTRA_SET="training.epochs=3"
EXTRA_SET="${EXTRA_SET:-}"

# ---------------------------------------------------- planned future actions
# FUTURE lets any script in this directory read insulin and/or carbohydrate
# values from *after* the forecast anchor:
#
#   FUTURE=none      (default) strictly causal -- the benchmark
#   FUTURE=insulin   future insulin schedule visible
#   FUTURE=carbs     future carbohydrate entries visible
#   FUTURE=both      both
#
# Off unless asked for, and every enabled run gets a name suffix so it can never
# overwrite or be confused with its causal twin. This is a different task, not a
# better model: 85.8% of this corpus is closed-loop, where insulin at t+k is a
# function of the CGM at t+k, so most of any gain is the label leaking backwards.
# Use it to measure that; never to report a forecasting result.
FUTURE="${FUTURE:-none}"
case "$FUTURE" in
  none|"")        _FUTURE_SET=""; _FUTURE_TAG="" ;;
  insulin)        _FUTURE_SET="data.future_modalities=['insulin']"; _FUTURE_TAG="_futins" ;;
  carbs)          _FUTURE_SET="data.future_modalities=['carbs']";   _FUTURE_TAG="_futcarb" ;;
  both|all)       _FUTURE_SET="data.future_modalities=['insulin','carbs']"; _FUTURE_TAG="_futboth" ;;
  *) echo "FUTURE must be one of: none insulin carbs both (got '$FUTURE')" >&2; exit 2 ;;
esac
# Optional: how many 5-minute steps past the anchor to expose. Default (unset)
# is out to the longest horizon being forecast.
FUTURE_STEPS="${FUTURE_STEPS:-}"
if [[ -n "$_FUTURE_SET" && -n "$FUTURE_STEPS" ]]; then
  _FUTURE_SET="$_FUTURE_SET data.future_steps=$FUTURE_STEPS"
  _FUTURE_TAG="${_FUTURE_TAG}${FUTURE_STEPS}"
fi
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
_gpu_ids=()          # devices we may schedule on, physical indices
_slot_pid=()         # one entry per in-flight job
_slot_gpu=()
_slot_name=()
declare -A _gpu_ready_at=()   # gpu -> epoch second before which we leave it alone
declare -A _gpu_free_mb=()    # gpu -> MiB free at the last poll
_queue_failed=0

queue_init() {
  _gpu_ids=()
  [[ -n "$GPUS" ]] && IFS=',' read -r -a _gpu_ids <<< "$GPUS"
  _slot_pid=(); _slot_gpu=(); _slot_name=()
  _gpu_ready_at=(); _gpu_free_mb=()
  _queue_failed=0
  if (( ${#_gpu_ids[@]} == 0 )); then
    log "queue: no GPU visible -- running one job at a time on CPU"
  else
    log "queue: GPUs [$GPUS], <=$JOBS_PER_GPU job(s) each, admitted only at >= ${GPU_FREE_MB} MiB free"
  fi
}

# Harvest exited jobs so their slots (and their share of the device) come back.
_reap() {
  local pid=() gpu=() name=() i rc
  for i in "${!_slot_pid[@]}"; do
    if kill -0 "${_slot_pid[$i]}" 2>/dev/null; then
      pid+=("${_slot_pid[$i]}"); gpu+=("${_slot_gpu[$i]}"); name+=("${_slot_name[$i]}")
      continue
    fi
    rc=0; wait "${_slot_pid[$i]}" || rc=$?
    if (( rc == 0 )); then
      log "done  ${_slot_name[$i]} (GPU ${_slot_gpu[$i]})"
    else
      warn "FAILED ${_slot_name[$i]} rc=$rc -- $LOG_DIR/${_slot_name[$i]}.log"
      _queue_failed=$(( _queue_failed + 1 ))
    fi
  done
  _slot_pid=("${pid[@]}"); _slot_gpu=("${gpu[@]}"); _slot_name=("${name[@]}")
  return 0
}

# One nvidia-smi call for the whole box, not one per device.
_poll_gpu_mem() {
  _gpu_free_mb=()
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  local idx used total
  while IFS=, read -r idx used total; do
    idx="${idx// /}"; used="${used// /}"; total="${total// /}"
    [[ -n "$idx" && -n "$total" ]] && _gpu_free_mb["$idx"]=$(( total - used ))
  done < <(nvidia-smi --query-gpu=index,memory.used,memory.total \
             --format=csv,noheader,nounits 2>/dev/null)
  return 0
}

_gpu_state() {
  local g out=""
  for g in "${_gpu_ids[@]}"; do out+="${g}=${_gpu_free_mb[$g]:-?}MiB "; done
  echo "${out% }"
}

# Echo the eligible device with the most headroom, or fail if there is none.
_pick_gpu() {
  local now g i mine ready free best="" best_free=-1
  now=$(date +%s)
  for g in "${_gpu_ids[@]}"; do
    mine=0
    for i in "${!_slot_gpu[@]}"; do
      [[ "${_slot_gpu[$i]}" == "$g" ]] && mine=$(( mine + 1 ))
    done
    (( mine >= JOBS_PER_GPU )) && continue
    ready="${_gpu_ready_at[$g]:-0}"
    (( now < ready )) && continue
    # No nvidia-smi reading for this device: fall back to capacity alone rather
    # than deadlocking on a number we cannot obtain.
    free="${_gpu_free_mb[$g]:-$GPU_FREE_MB}"
    (( free < GPU_FREE_MB )) && continue
    (( free > best_free )) && { best_free=$free; best="$g"; }
  done
  [[ -n "$best" ]] || return 1
  echo "$best"
}

# Block until a device is genuinely free, then leave it in _ACQUIRED_GPU.
#
# This sets a global instead of echoing, because the caller would have to write
# `gpu="$(_acquire_gpu)"` -- a subshell, in which _reap would harvest exited jobs
# from a *copy* of the slot arrays and the parent would go on believing every
# slot is still busy. That deadlocks after the first full pass.
_ACQUIRED_GPU=""
_acquire_gpu() {
  local waited=0
  _ACQUIRED_GPU=""
  # A caller that forgot queue_init has an empty device list while $GPUS names
  # real cards. Falling through to the CPU path there is silent and wrong: the
  # child inherits every device, torch takes cuda:0 regardless of who is on it,
  # and the sweep serialises. Fail loudly instead.
  if (( ${#_gpu_ids[@]} == 0 )) && [[ -n "$GPUS" ]]; then
    die "queue not initialised: GPUS=[$GPUS] but no devices registered. Call queue_init before the first queue_run/train."
  fi
  if (( ${#_gpu_ids[@]} == 0 )); then          # CPU: strictly one at a time
    while _reap; (( ${#_slot_pid[@]} >= 1 )); do sleep 5; done
    _ACQUIRED_GPU="cpu"; return 0
  fi
  while true; do
    _reap
    _poll_gpu_mem
    if _ACQUIRED_GPU="$(_pick_gpu)"; then return 0; fi
    if (( GPU_WAIT_MAX_S > 0 && waited >= GPU_WAIT_MAX_S )); then
      die "no GPU reached ${GPU_FREE_MB} MiB free in ${waited}s [$(_gpu_state)]"
    fi
    # Say something once we have actually blocked for a poll interval, then
    # every five minutes -- not on every dispatch while the queue is saturated,
    # which is the normal steady state and not news.
    if (( waited == GPU_POLL_S || (waited > 0 && waited % 300 == 0) )); then
      warn "waiting for >= ${GPU_FREE_MB} MiB free [$(_gpu_state)]; ${#_slot_pid[@]} of ours running, ${waited}s so far"
    fi
    sleep "$GPU_POLL_S"
    waited=$(( waited + GPU_POLL_S ))
  done
}

# queue_run <run_name> <command...>
queue_run() {
  local name="$1"; shift
  local logfile="$LOG_DIR/${name}.log"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "GPU=? $* > $logfile"
    return 0
  fi
  _acquire_gpu
  local gpu="$_ACQUIRED_GPU"
  log "start $name on GPU $gpu (${_gpu_free_mb[$gpu]:-?} MiB free) -> $logfile"
  (
    if [[ "$gpu" != "cpu" ]]; then export CUDA_VISIBLE_DEVICES="$gpu"; fi
    "$@" > "$logfile" 2>&1
  ) &
  _slot_pid+=("$!"); _slot_gpu+=("$gpu"); _slot_name+=("$name")
  if [[ "$gpu" != "cpu" ]]; then
    _gpu_ready_at["$gpu"]=$(( $(date +%s) + GPU_WARMUP_S ))
  fi
  return 0
}

queue_wait() {
  while (( ${#_slot_pid[@]} > 0 )); do
    _reap
    (( ${#_slot_pid[@]} > 0 )) && sleep 5
  done
  (( _queue_failed > 0 )) && warn "$_queue_failed job(s) failed; check $LOG_DIR"
  log "all queued jobs finished"
  _queue_failed=0
  return 0
}

# train <run_name> <config> [extra --set args...]
train() {
  local name="$1$_FUTURE_TAG" config="$2"; shift 2
  if [[ -f "$RUNS_DIR/$name/metrics.json" && "${SKIP_EXISTING:-1}" == "1" ]]; then
    log "skip $name (already scored; SKIP_EXISTING=0 to force)"
    return
  fi
  queue_run "$name" $PY -m metabonet_forecasting.training.train \
    --config "$config" --name "$name" --runs-dir "$RUNS_DIR" \
    --set "data.store=$STORE" $_FUTURE_SET $EXTRA_SET "$@"
}

# baseline <run_name> <config> <model> [extra --set args...]
baseline() {
  local name="$1$_FUTURE_TAG" config="$2" model="$3"; shift 3
  if [[ -f "$RUNS_DIR/$name/metrics.json" && "${SKIP_EXISTING:-1}" == "1" ]]; then
    log "skip $name (already scored)"
    return
  fi
  queue_run "$name" $PY -m metabonet_forecasting.baselines.run_baselines \
    --config "$config" --model "$model" --name "$name" --runs-dir "$RUNS_DIR" \
    --set "data.store=$STORE" $_FUTURE_SET $EXTRA_SET "$@"
}

summarise() {
  log "aggregating $RUNS_DIR -> $RESULTS_DIR"
  $PY -m metabonet_forecasting.evaluation.aggregate "$RUNS_DIR" \
    --out "$RESULTS_DIR" --pattern "${1:-*}" || warn "nothing to aggregate yet"
}
