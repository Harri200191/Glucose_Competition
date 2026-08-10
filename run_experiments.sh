#!/usr/bin/env bash
#
# run_experiments.sh -- build the dataset, then run the full model matrix with
# automatic GPU allocation, per-run logs and resume.
#
# Scheduling model
#   Two independent pools. GPU-backed learners (xgboost, tabpfn, the neural nets)
#   claim one whole GPU each; CPU-only learners (lightgbm, catboost-MAE, the
#   forests) never occupy a GPU slot. A job launches as soon as a slot in ITS pool
#   frees up, so a queue of CPU jobs cannot block the GPUs and vice versa.
#
#   Each GPU job runs with CUDA_VISIBLE_DEVICES pinned to exactly one device and is
#   told --gpu 0. That way the child sees a single device as ordinal 0, which keeps
#   allocation correct even if CUDA_VISIBLE_DEVICES is already set in the parent
#   shell, and stops one job's allocator from reserving memory on its neighbours.
#
#   A GPU counts as available when its used memory is below --gpu-free-mb, so the
#   script co-exists with other people's jobs on a shared box.
#
# Usage
#   ./run_experiments.sh                          # everything, official split
#   ./run_experiments.sh --dry-run                # print the plan, run nothing
#   ./run_experiments.sh --models "xgboost mlp" --horizons "30 60"
#   ./run_experiments.sh --split cohort --cohorts "Loop ReplaceBG IOBP2"
#   ./run_experiments.sh --gpus 2,3 --resume
#   ./run_experiments.sh --collect-only
#
set -euo pipefail

# --------------------------------------------------------------------- defaults
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-$HOME/anaconda3/envs/ecg_cpc/bin/python}"

DATA_DIR="$REPO/build/main"
RUNS_DIR="$REPO/runs"
LOG_ROOT="$REPO/logs"

STRIDE=6
SPLIT="official"
HORIZONS="30 60 90 120"
FOLDS="0"
COHORTS=""
SEED=0

MODELS="persistence linear_trend xgboost lightgbm histgb catboost randomforest extratrees mlp resmlp ft_transformer tcn gru tabpfn"

# Learners that want a GPU. Everything else runs CPU-only and skips the GPU pool.
GPU_MODELS=" xgboost xgboost_l2 catboost_gpu tabpfn mlp resmlp ft_transformer tcn gru "

# Per-model training-row caps. These learners do not finish in a sane wall clock on
# the full ~10 M-row matrix; the cap is part of the result, so it is explicit here
# rather than hidden in the model code.
declare -A TRAIN_CAP=( [randomforest]=3000000 [extratrees]=3000000 )
declare -A EVAL_CAP=( [tabpfn]=200000 )

GPUS_OVERRIDE=""
GPU_FREE_MB=2000        # a GPU with less than this in use counts as free
MAX_GPU_JOBS=0          # 0 = one job per available GPU
MAX_CPU_JOBS=0          # 0 = auto from core count
POLL_SEC=10

DO_BUILD=1
DO_RUN=1
DO_COLLECT=1
RESUME=0
DRY_RUN=0

# --------------------------------------------------------------------- arguments
usage() { sed -n '2,32p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data)          DATA_DIR="$2"; shift 2 ;;
    --runs-dir)      RUNS_DIR="$2"; shift 2 ;;
    --models)        MODELS="$2"; shift 2 ;;
    --horizons)      HORIZONS="$2"; shift 2 ;;
    --split)         SPLIT="$2"; shift 2 ;;
    --folds)         FOLDS="$2"; shift 2 ;;
    --cohorts)       COHORTS="$2"; shift 2 ;;
    --stride)        STRIDE="$2"; shift 2 ;;
    --seed)          SEED="$2"; shift 2 ;;
    --gpus)          GPUS_OVERRIDE="$2"; shift 2 ;;
    --gpu-free-mb)   GPU_FREE_MB="$2"; shift 2 ;;
    --gpu-jobs)      MAX_GPU_JOBS="$2"; shift 2 ;;
    --cpu-jobs)      MAX_CPU_JOBS="$2"; shift 2 ;;
    --python)        PY="$2"; shift 2 ;;
    --skip-build)    DO_BUILD=0; shift ;;
    --build-only)    DO_RUN=0; DO_COLLECT=0; shift ;;
    --collect-only)  DO_BUILD=0; DO_RUN=0; shift ;;
    --resume)        RESUME=1; shift ;;
    --dry-run)       DRY_RUN=1; shift ;;
    -h|--help)       usage ;;
    *) echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
  esac
done

# ----------------------------------------------------------------------- logging
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$LOG_ROOT/$STAMP"
RUN_LOG_DIR="$LOG_DIR/runs"
STATUS_DIR="$LOG_DIR/status"
SLOT_DIR="$LOG_DIR/.slots"
MASTER_LOG="$LOG_DIR/master.log"
mkdir -p "$RUN_LOG_DIR" "$STATUS_DIR" "$SLOT_DIR"
ln -sfn "$LOG_DIR" "$LOG_ROOT/latest"

log() { printf '%s  %s\n' "$(date '+%H:%M:%S')" "$*" | tee -a "$MASTER_LOG"; }
rule() { printf '%s\n' "$(printf '=%.0s' {1..78})" | tee -a "$MASTER_LOG"; }

# ---------------------------------------------------------------------- teardown
cleanup() {
  local rc=$?
  if [[ -n "$(jobs -pr 2>/dev/null || true)" ]]; then
    log "interrupted -- terminating $(jobs -pr | wc -l) running job(s)"
    # shellcheck disable=SC2046
    kill $(jobs -pr) 2>/dev/null || true
    sleep 2
    # shellcheck disable=SC2046
    kill -9 $(jobs -pr) 2>/dev/null || true
  fi
  rm -rf "$SLOT_DIR"
  exit $rc
}
trap cleanup INT TERM

# --------------------------------------------------------------------- preflight
preflight() {
  log "preflight"
  [[ -x "$PY" ]] || { log "  FATAL: python not found at $PY (set --python)"; exit 1; }
  log "  python      $("$PY" -V 2>&1)"

  local missing="" m
  for m in numpy pandas pyarrow sklearn xgboost torch; do
    "$PY" -c "import $m" >/dev/null 2>&1 || missing="$missing $m"
  done
  # optional learners: warn, do not fail
  for m in lightgbm catboost tabpfn; do
    if ! "$PY" -c "import $m" >/dev/null 2>&1; then
      log "  WARNING: $m not importable -- those runs will fail and be logged"
    fi
  done
  [[ -z "$missing" ]] || { log "  FATAL: missing required packages:$missing"; exit 1; }

  local xgbv; xgbv="$("$PY" -c 'import xgboost;print(xgboost.__version__)' 2>/dev/null || echo '?')"
  log "  xgboost     $xgbv"
  case "$xgbv" in
    2.*) : ;;
    *) log "  WARNING: xgboost $xgbv -- V100s are SM 70 and xgboost >=3.1 dropped that"
       log "           target. If GPU runs fail with 'not compiled for SM 70', pin 2.1.4." ;;
  esac

  if [[ ! -f "$REPO/data/metabonet_public.parquet" ]]; then
    log "  FATAL: data/metabonet_public.parquet not found"; exit 1
  fi
  log "  corpus      $(du -h "$REPO/data/metabonet_public.parquet" | cut -f1)"
}

# ------------------------------------------------------------------ GPU discovery
discover_gpus() {
  local found=()
  if [[ -n "$GPUS_OVERRIDE" ]]; then
    IFS=',' read -r -a found <<< "$GPUS_OVERRIDE"
    log "  GPUs        ${found[*]} (forced via --gpus)"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    local line idx used total
    while IFS=',' read -r idx used total; do
      idx="${idx// /}"; used="${used// /}"; total="${total// /}"
      [[ -n "$idx" ]] || continue
      if (( used < GPU_FREE_MB )); then
        found+=("$idx")
      else
        log "  GPU $idx busy (${used} MiB / ${total} MiB in use) -- skipping"
      fi
    done < <(nvidia-smi --query-gpu=index,memory.used,memory.total \
                        --format=csv,noheader,nounits 2>/dev/null)
    log "  GPUs free   ${found[*]:-none}"
  else
    log "  GPUs        nvidia-smi not found -- CPU only"
  fi
  GPU_LIST=("${found[@]}")
}

# ------------------------------------------------------------------- slot pools
# A slot is a file in $SLOT_DIR. The parent creates it on acquire; the child
# deletes it on exit (success or failure), so a crashed job always frees its slot.
try_acquire() {           # $1 = pool ("gpu"|"cpu") -> prints slot id, or fails
  local pool="$1" s
  local -n pool_slots="${pool^^}_SLOTS"
  for s in "${pool_slots[@]}"; do
    if [[ ! -e "$SLOT_DIR/$pool-$s.busy" ]]; then
      : > "$SLOT_DIR/$pool-$s.busy"
      printf '%s' "$s"
      return 0
    fi
  done
  return 1
}

needs_gpu() { [[ "$GPU_MODELS" == *" $1 "* ]]; }

run_tag() {               # model horizon fold cohort
  local t="$1_h$2_$SPLIT"
  [[ -n "${4:-}" ]] && t="${t}_$4"
  [[ "$SPLIT" == "subject" ]] && t="${t}_f$3"
  printf '%s' "$t"
}

# ------------------------------------------------------------------- job runner
launch() {                # pool slot model horizon fold cohort tag
  local pool="$1" slot="$2" model="$3" horizon="$4" fold="$5" cohort="$6" tag="$7"
  local logf="$RUN_LOG_DIR/$tag.log"
  local statf="$STATUS_DIR/$tag.tsv"
  local out="$RUNS_DIR/$tag"

  local -a cmd=( "$PY" -m exp.train
                 --data "$DATA_DIR" --model "$model" --horizon "$horizon"
                 --split "$SPLIT" --fold "$fold" --seed "$SEED" --out "$out" )
  [[ -n "$cohort" ]] && cmd+=( --holdout-cohort "$cohort" )
  [[ -n "${TRAIN_CAP[$model]:-}" ]] && cmd+=( --max-train "${TRAIN_CAP[$model]}" )
  [[ -n "${EVAL_CAP[$model]:-}"  ]] && cmd+=( --max-eval  "${EVAL_CAP[$model]}" )

  local device_env device_label njobs
  if [[ "$pool" == "gpu" ]]; then
    device_env="$slot"; device_label="gpu$slot"; cmd+=( --gpu 0 )
    njobs=$(( NCORES / (GPU_POOL_N + CPU_POOL_N) )); (( njobs < 4 )) && njobs=4
  else
    device_env=""; device_label="cpu"; cmd+=( --gpu -1 )
    njobs=$(( NCORES / CPU_POOL_N )); (( njobs < 4 )) && njobs=4
  fi
  cmd+=( --n-jobs "$njobs" )

  (
    trap 'rm -f "$SLOT_DIR/$pool-$slot.busy"' EXIT
    local t0 t1 rc
    t0=$(date +%s)
    {
      echo "### $tag"
      echo "### device=$device_label n_jobs=$njobs started=$(date -Is)"
      echo "### ${cmd[*]}"
      echo
    } > "$logf"
    set +e
    CUDA_VISIBLE_DEVICES="$device_env" OMP_NUM_THREADS="$njobs" \
      "${cmd[@]}" >> "$logf" 2>&1
    rc=$?
    set -e
    t1=$(date +%s)

    local mard status
    if (( rc == 0 )) && [[ -f "$out/result.json" ]]; then
      status="ok"
      mard=$("$PY" -c "import json,sys;print(f\"{json.load(open(sys.argv[1]))['test']['MARD']:.4f}\")" \
             "$out/result.json" 2>/dev/null || echo "")
    else
      status="FAILED"; mard=""
      echo "### exit=$rc" >> "$logf"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$tag" "$status" "$rc" "$((t1-t0))" "$device_label" "$mard" "$logf" > "$statf"

    if [[ "$status" == "ok" ]]; then
      log "  [done]  $tag  MARD ${mard:-?}%  ${device_label}  $((t1-t0))s"
    else
      log "  [FAIL]  $tag  rc=$rc  ${device_label}  $((t1-t0))s  -> $logf"
      tail -n 3 "$logf" | sed 's/^/          | /' | tee -a "$MASTER_LOG" >/dev/null
    fi
  ) &
}

# ------------------------------------------------------------------------- main
NCORES="$(nproc)"
rule
log "MetaboNet experiment sweep"
log "  repo        $REPO"
log "  log dir     $LOG_DIR"
log "  split       $SPLIT   horizons: $HORIZONS   seed: $SEED"
log "  cores       $NCORES"
rule

preflight
discover_gpus

GPU_SLOTS=("${GPU_LIST[@]:-}")
[[ ${#GPU_SLOTS[@]} -eq 1 && -z "${GPU_SLOTS[0]}" ]] && GPU_SLOTS=()
if (( MAX_GPU_JOBS > 0 )) && (( ${#GPU_SLOTS[@]} > MAX_GPU_JOBS )); then
  GPU_SLOTS=("${GPU_SLOTS[@]:0:MAX_GPU_JOBS}")
fi
if (( MAX_CPU_JOBS <= 0 )); then
  MAX_CPU_JOBS=$(( NCORES / 24 )); (( MAX_CPU_JOBS < 1 )) && MAX_CPU_JOBS=1
  (( MAX_CPU_JOBS > 4 )) && MAX_CPU_JOBS=4
fi
CPU_SLOTS=(); for ((i=0;i<MAX_CPU_JOBS;i++)); do CPU_SLOTS+=("$i"); done
GPU_POOL_N=${#GPU_SLOTS[@]}
CPU_POOL_N=${#CPU_SLOTS[@]}
log "  pools       gpu=${GPU_POOL_N} (${GPU_SLOTS[*]:-none})   cpu=${CPU_POOL_N}"

if (( GPU_POOL_N == 0 )); then
  log "  NOTE: no GPU available -- GPU-backed models will run on CPU and be slow."
fi

# ------------------------------------------------------------------ build stage
if (( DO_BUILD )); then
  if [[ -f "$DATA_DIR/manifest.json" ]] && (( RESUME )); then
    log "build: reusing $DATA_DIR (--resume)"
  else
    log "build: $DATA_DIR (stride=$STRIDE)"
    if (( DRY_RUN )); then
      log "  DRY RUN: $PY -m exp.build_dataset --out $DATA_DIR --stride $STRIDE"
    else
      bt0=$(date +%s)
      if ! "$PY" -m exp.build_dataset --out "$DATA_DIR" --stride "$STRIDE" \
            > "$LOG_DIR/build.log" 2>&1; then
        log "  FATAL: build failed -- see $LOG_DIR/build.log"
        tail -n 20 "$LOG_DIR/build.log" | sed 's/^/    | /' | tee -a "$MASTER_LOG"
        exit 1
      fi
      log "  built in $(( $(date +%s) - bt0 ))s -> $LOG_DIR/build.log"
      "$PY" -c "
import json;m=json.load(open('$DATA_DIR/manifest.json'))
print(f\"  {m['n_samples']:,} samples x {m['n_features']} features, \"
      f\"{m['n_subjects']} subjects, {m['n_cohorts']} cohorts\")" | tee -a "$MASTER_LOG"
    fi
  fi
fi

if (( DO_RUN )) && [[ ! -f "$DATA_DIR/manifest.json" ]] && (( ! DRY_RUN )); then
  log "FATAL: $DATA_DIR/manifest.json missing -- build first (drop --skip-build)"
  exit 1
fi

# -------------------------------------------------------------------- job queue
PENDING=()
for model in $MODELS; do
  for h in $HORIZONS; do
    case "$SPLIT" in
      subject) for f in $FOLDS; do PENDING+=("$model|$h|$f|"); done ;;
      cohort)
        [[ -n "$COHORTS" ]] || { log "FATAL: --split cohort needs --cohorts"; exit 2; }
        for c in $COHORTS; do PENDING+=("$model|$h|0|$c"); done ;;
      *) PENDING+=("$model|$h|0|") ;;
    esac
  done
done

# drop already-completed runs when resuming
if (( RESUME )); then
  KEPT=()
  for job in "${PENDING[@]}"; do
    IFS='|' read -r model h f c <<< "$job"
    tag="$(run_tag "$model" "$h" "$f" "$c")"
    if [[ -f "$RUNS_DIR/$tag/result.json" ]]; then
      log "  skip $tag (already complete)"
    else
      KEPT+=("$job")
    fi
  done
  PENDING=("${KEPT[@]:-}")
  [[ ${#PENDING[@]} -eq 1 && -z "${PENDING[0]}" ]] && PENDING=()
fi

TOTAL=${#PENDING[@]}
log "queued $TOTAL run(s)"
if (( DRY_RUN )); then
  for job in "${PENDING[@]}"; do
    IFS='|' read -r model h f c <<< "$job"
    pool="cpu"; needs_gpu "$model" && pool="gpu"
    printf '  %-6s %s\n' "[$pool]" "$(run_tag "$model" "$h" "$f" "$c")" | tee -a "$MASTER_LOG"
  done
  log "dry run -- nothing executed"
  rm -rf "$SLOT_DIR"
  exit 0
fi

# --------------------------------------------------------------------- scheduler
# Walk the pending list each pass and launch the first job whose pool has a free
# slot. A blocked GPU queue therefore never stalls ready CPU jobs, or vice versa.
if (( DO_RUN )) && (( TOTAL > 0 )); then
  mkdir -p "$RUNS_DIR"
  SWEEP_T0=$(date +%s)
  launched=0
  while (( ${#PENDING[@]} > 0 )); do
    progressed=0
    REMAIN=()
    for job in "${PENDING[@]}"; do
      IFS='|' read -r model h f c <<< "$job"
      pool="cpu"; needs_gpu "$model" && pool="gpu"
      # no GPUs on this box: fall back to the CPU pool rather than never running
      [[ "$pool" == "gpu" && $GPU_POOL_N -eq 0 ]] && pool="cpu"

      if (( progressed == 0 )) && slot="$(try_acquire "$pool")"; then
        tag="$(run_tag "$model" "$h" "$f" "$c")"
        launched=$((launched+1))
        log "  [$launched/$TOTAL] start $tag  ($pool slot $slot)"
        launch "$pool" "$slot" "$model" "$h" "$f" "$c" "$tag"
        progressed=1
      else
        REMAIN+=("$job")
      fi
    done
    PENDING=("${REMAIN[@]:-}")
    [[ ${#PENDING[@]} -eq 1 && -z "${PENDING[0]}" ]] && PENDING=()
    (( progressed == 0 )) && (( ${#PENDING[@]} > 0 )) && sleep "$POLL_SEC"
  done

  log "all jobs dispatched -- waiting for $(jobs -pr | wc -l) to finish"
  wait
  log "sweep finished in $(( ($(date +%s) - SWEEP_T0) / 60 )) min"
fi

# ----------------------------------------------------------------------- results
STATUS_TSV="$LOG_DIR/status.tsv"
{
  printf 'tag\tstatus\trc\tseconds\tdevice\tMARD\tlog\n'
  cat "$STATUS_DIR"/*.tsv 2>/dev/null | sort
} > "$STATUS_TSV"

nok=$(awk -F'\t' 'NR>1 && $2=="ok"' "$STATUS_TSV" | wc -l)
nbad=$(awk -F'\t' 'NR>1 && $2!="ok"' "$STATUS_TSV" | wc -l)
rule
log "runs: $nok ok, $nbad failed   (per-run status: $STATUS_TSV)"
if (( nbad > 0 )); then
  log "failures:"
  awk -F'\t' 'NR>1 && $2!="ok" {printf "    %-44s rc=%-4s %s\n", $1, $3, $7}' \
    "$STATUS_TSV" | tee -a "$MASTER_LOG"
fi

if (( DO_COLLECT )); then
  log "collecting results table"
  "$PY" -c "
from exp.run_all import collect
from pathlib import Path
collect(Path('$RUNS_DIR'))" 2>&1 | tee -a "$MASTER_LOG" || \
    log "  collect failed (no results yet?)"
fi

rm -rf "$SLOT_DIR"
rule
log "logs      $LOG_DIR       (symlinked as $LOG_ROOT/latest)"
log "per-run   $RUN_LOG_DIR"
log "table     $RUNS_DIR/results.csv"
(( nbad == 0 ))
