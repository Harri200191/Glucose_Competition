#!/usr/bin/env bash
# THE FINAL MODEL: three backbones, zone_t objective, 96 h context, trained on
# ALL the data but a small validation slice, 15 epochs, resumable mid-epoch.
#
#   scripts/run_final.sh                       # = STAGE=train, all three backbones
#   STAGE=train  GPUS=0,1,2 scripts/run_final.sh
#   STAGE=submit scripts/run_final.sh          # live + annual parquets from best.pt
#   STAGE=submit SNAPSHOT=ep07_end scripts/run_final.sh   # ... from any checkpoint
#   STAGE=ensemble scripts/run_final.sh        # equal-weight ensemble of the three
#   BACKBONES="mamba2" scripts/run_final.sh    # a subset
#
# Launch detached (t.bash idiom) so it survives a closed laptop:
#   setsid nohup scripts/run_final.sh > logs/run_final.log 2>&1 </dev/null & disown
#
# RESUME: just run the SAME command again. Each run resumes from its last.pt,
# which is rewritten at 1/4, 1/2 and 3/4 of every epoch and at every epoch end.
# The train order is a seeded permutation of (seed, epoch), so a mid-epoch
# resume continues the exact same order: no window repeated, none skipped.
# The script refuses to start a backbone whose run is already alive.
#
# NOTHING IS EVER UPLOADED. Submission files are written to
# artifacts/submission/; you upload them by hand.
#
# ===========================================================================
# WHAT GOES IN, AND THE NUMBER BEHIND EACH CHOICE
#
#   objective   training.loss.name=zone_t, model.head.law=student_t
#               The exact expected zone A under a Student-t, closed form.
#               Mini-corpus +0.455 (ModernTCN), +0.608 (mamba2), +0.689
#               (transformerpp) on livemix, band 0.21. Full scale, same backbone:
#               +0.63 / +0.35 / +0.66 on the live board, patient-bootstrap CIs
#               excluding zero. Stacking anything on it (band_offset,
#               sharpness, logprob, MARD, soft-DTS) measured -0.08 to -0.59.
#   law.k       transformerpp 3, mambaformer 1, mamba2 1 -- the variants proven
#               at full scale in c96t. k is flat from 3 to 7 (0.145 span).
#   zone_obj    prob (default). logprob is dropped: -0.152 when stacked, ~45%
#               slower per epoch, and NOT the cause of the mamba NaN (see below).
#   context     1152 (96 h). 24->48->72 h: +0.90 then +0.50 holdout; 72->96 h
#               +0.60 live (saturating: 576->864 was only +0.15).
#   target      log_ratio. Required by zone_t (the band is relative).
#   inputs      future insulin (legal on BOTH boards: the organisers ship it;
#               +12.45 measured). Never report as a forecasting result.
#               use_source_identity=false: the annual template has one
#               anonymised source_file level the embedding never saw.
#   head        horizon_cross, chain=dist (+0.39 holdout, grows with horizon),
#               FiLM fusion (raw_concat -0.306), 7 quantiles, d_model 128
#               (64: -0.326; 256: -0.062).
#   weights     horizon_weights [1,1,1.5,2]. Flat was +0.018, steeper -0.18/-0.28.
#   data        ALL of it, via data.split.scheme=all:
#                 20,503,100 train windows from 1,825 patients = 95.2% of the
#                 store and 26.6% more than c96t (16.2M). Includes both sides of
#                 the official is_test flag (so the live-board patients:
#                 2,978,880 of 3,232,757 official-test anchors), every cohort,
#                 and ALL 1,245,347 IOBP2 anchors (the internal-holdout 89 too).
#               More unique data is the one knob that beat the band on the
#               16-knob screen: 2x windows +0.542 livemix, +0.84 holdout.
#   val         79 NON-IOBP2 patients (val_frac 0.05, val_exclude IOBP2),
#               1,041,161 anchors. Picks the best of the 15 epochs. test == val,
#               so metrics.json is an HONEST held-out score on unseen patients.
#               val_frac MUST be set: base.yaml's 0.12 would hold out 228
#               patients, 38 of them IOBP2.
#   epochs      15, no early stopping (patience 100), select on val_zone_a,
#               min_delta 0 (0.02 was the bug that stopped c96s at epoch 3).
#               Full scale kept improving: epochs 6->9 were +0.13..+0.19 live.
#   ensemble    one model per backbone, equal weights, no recalibration.
#               Gain is backbone diversity: 3 transformerpp fine-tunes +0.022
#               over one; +mambaformer +0.211; +mamba2 +0.058; 9 models = 3.
#               Weights/affine may NOT be fitted on the live key any more --
#               these models train on the live-board patients. Equal weights
#               were within 0.1 of fitted ones on clean models (85.706 vs 85.793).
#
# THE MAMBA / MAMBAFORMER "EXPLOSIONS" -- FIXED IN CODE, NOT HERE
#   Root cause: models/mamba2.py cast the fp32 scan output back to fp16 under
#   AMP; on hypoglycaemic stretches deep in a 96 h window it exceeded 65,504,
#   became inf, and the gated norm turned inf into NaN. 16/73 mamba2 and 8/14
#   mambaformer runs, 0/296 other backbones. The scan AND the gated norm now
#   run in fp32. Verified on GPU under real AMP on the exact anchors that
#   failed: 64/64 finite (was 0), |AMP - fp32| 0.04 mg/dL mean.
#   Guards: training.max_nonfinite_batches=50 STOPS the run instead of
#   silently skipping (the skipped batches were the hypo/IOBP2-heavy ones), and
#   an epoch with any NaN val prediction can never become best.pt.
#   warmup 4000 steps: the one clean full-scale mamba2 relaunch used it.
#
# COST (V100-32GB, measured throughput 283 / 283 / 224 windows/s):
#   transformerpp ~20.7 h/epoch -> ~13 days for 15
#   mamba2        ~20.9 h/epoch -> ~13 days
#   mambaformer   ~26.5 h/epoch -> ~16.5 days
#   Every epoch end and every quarter leaves a usable checkpoint in
#   runs/final_<bb>/ckpt/, so you can submit before the run finishes
#   (STAGE=submit SNAPSHOT=epEE_end). NOTE: the LR follows a cosine over all
#   15 epochs, so an early snapshot has not had its LR annealed.
# ===========================================================================

set -euo pipefail
source "$(dirname "$0")/lib/common.sh"

STAGE="${STAGE:-train}"
TAG="${TAG:-final}"
read -r -a BB <<< "${BACKBONES:-transformerpp mambaformer mamba2}"
STORE="$REPO_ROOT/artifacts/store/ctx1152"
SUB="$REPO_ROOT/artifacts/submission"
EPOCHS="${EPOCHS:-15}"
VAL_FRAC="${VAL_FRAC:-0.05}"
Q="[0.05,0.1,0.25,0.5,0.75,0.9,0.95]"

lr_of() { case "$1" in transformerpp) echo 0.0007;; mambaformer) echo 0.0005;; mamba2) echo 0.0004;; esac; }
bs_of() { case "$1" in transformerpp) echo 144;;    mambaformer) echo 72;;     mamba2) echo 72;;     esac; }
k_of()  { case "$1" in transformerpp) echo 3;;      mambaformer) echo 1;;      mamba2) echo 1;;      esac; }

preflight() {
  require_store
  grep -q 'y = self.norm(y.reshape(b, l, self.d_inner), z)' "$REPO_ROOT/metabonet_forecasting/models/mamba2.py" \
    || die "the Mamba fp32 fix is NOT in models/mamba2.py -- mamba2/mambaformer would diverge again"
  grep -q 'def all_data_split' "$REPO_ROOT/metabonet_forecasting/data/splits.py" \
    || die "data.split.scheme=all is missing from data/splits.py"
  grep -q 'checkpoint_fractions' "$REPO_ROOT/metabonet_forecasting/training/loop.py" \
    || die "mid-epoch checkpointing is missing from training/loop.py"
  local free_gb; free_gb=$(df -BG --output=avail "$REPO_ROOT" | tail -1 | tr -dc '0-9')
  (( free_gb > 50 )) || die "only ${free_gb} GB free on disk"
  log "preflight OK: fix present, scheme=all present, checkpointing present, ${free_gb} GB free"
}

run_alive() { pgrep -f -- "--name $1 " >/dev/null 2>&1; }

train_all() {
  preflight
  # common.sh has ALREADY set GPU_FREE_MB=12000 by now; a ${GPU_FREE_MB:-...}
  # default would silently keep it (that is how c96u_mambaformer OOMed next to
  # an inference job). A ctx1152 run holds ~22.5 GiB: take only an empty card.
  GPU_FREE_MB="${FINAL_FREE_MB:-30000}"; JOBS_PER_GPU=1
  gpu_autoconfig "$GPU_FREE_MB"
  queue_init
  local b name
  for b in "${BB[@]}"; do
    name="${TAG}_$b"
    if run_alive "$name"; then warn "$name is already running -- not starting a second copy"; continue; fi
    [[ -f "$RUNS_DIR/$name/last.pt" ]] && log "$name: last.pt found, will RESUME" \
                                       || log "$name: fresh start"
    train "$name" "model/$b" \
      "seed=0" \
      "data.context=1152" \
      "data.target_mode=log_ratio" \
      "data.future_modalities=['insulin']" \
      "data.use_source_identity=false" \
      "data.split.scheme=all" \
      "data.split.val_frac=$VAL_FRAC" \
      "data.split.seed=0" \
      "data.split.val_exclude=['IOBP2']" \
      "data.internal_holdout.enabled=false" \
      "data.sampling.max_train_windows=null" \
      "data.sampling.max_val_windows=150000" \
      "data.sampling.max_test_windows=400000" \
      "data.loader.batch_size=$(bs_of "$b")" \
      "data.loader.num_workers=8" \
      "model.d_model=128" \
      "model.fusion.name=film" \
      "model.head.name=horizon_cross" \
      "model.head.chain=dist" \
      "model.head.quantiles=$Q" \
      "model.head.law.name=student_t" \
      "model.head.law.k=$(k_of "$b")" \
      "training.loss.name=zone_t" \
      "training.loss.horizon_weights=[1,1,1.5,2]" \
      "training.epochs=$EPOCHS" \
      "training.patience=100" \
      "training.min_delta=0.0" \
      "training.select_metric=val_zone_a" \
      "training.resume=true" \
      "training.lr=$(lr_of "$b")" \
      "training.warmup_steps=4000" \
      "training.checkpoint_fractions=[0.25,0.5,0.75]" \
      "training.max_nonfinite_batches=50" \
      "training.predict_val=true"
  done
  queue_wait
}

# Build live + annual parquets. SNAPSHOT=epEE_end|epEE_qK uses a mid-run
# checkpoint instead of best.pt, via a frozen copy so training is never raced.
submit_all() {
  local b src name
  for b in "${BB[@]}"; do
    src="$RUNS_DIR/${TAG}_$b"; name="${TAG}_$b"
    if [[ -n "${SNAPSHOT:-}" ]]; then
      [[ -f "$src/ckpt/$SNAPSHOT.pt" ]] || { warn "no $src/ckpt/$SNAPSHOT.pt, skipping $b"; continue; }
      name="${TAG}_${b}__$SNAPSHOT"
      mkdir -p "$RUNS_DIR/$name"
      cp "$src/ckpt/$SNAPSHOT.pt" "$RUNS_DIR/$name/best.pt"
      cp "$src/config.resolved.yaml" "$src/standardizer.json" "$src/manifest.json" "$RUNS_DIR/$name/"
    fi
    [[ -f "$RUNS_DIR/$name/best.pt" ]] || { warn "no checkpoint for $name yet, skipping"; continue; }
    log "building live + annual submissions for $name"
    COMPETITION=both "$REPO_ROOT/scripts/90_make_submission.sh" "$name"
  done
}

# Equal weights, NO recalibration: these models trained on the live-board
# patients, so nothing may be fitted on the live answer key.
ensemble_all() {
  local sfx="${SNAPSHOT:+__$SNAPSHOT}" L=() A=() b
  for b in "${BB[@]}"; do
    [[ -f "$SUB/live_${TAG}_${b}${sfx}.parquet"   ]] && L+=("$SUB/live_${TAG}_${b}${sfx}.parquet")
    [[ -f "$SUB/annual_${TAG}_${b}${sfx}.parquet" ]] && A+=("$SUB/annual_${TAG}_${b}${sfx}.parquet")
  done
  (( ${#L[@]} == ${#BB[@]} && ${#A[@]} == ${#BB[@]} )) \
    || die "need live AND annual parquets for all of: ${BB[*]} (have ${#L[@]} live, ${#A[@]} annual). Run STAGE=submit first."
  local ol="$SUB/live_${TAG}_ens${sfx}.parquet" oa="$SUB/annual_${TAG}_ens${sfx}.parquet"
  $PY -m metabonet_forecasting.submission.recalibrate \
    --live "${L[@]}" --annual "${A[@]}" \
    --targets "$REPO_ROOT/toolkit/data/live_leaderboard/targets.parquet" \
    --out-live "$ol" --out-annual "$oa" --no-recal
  for f in "$ol" "$oa"; do
    ( cd "$REPO_ROOT/toolkit" && $PY run.py "$f" --competition "$([[ $f == */live_* ]] && echo live || echo annual)" ) \
      || die "toolkit rejected $f"
  done
  log "READY TO UPLOAD BY HAND:"
  log "  live   -> $ol   (its live score is IN-SAMPLE: these models trained on those patients)"
  log "  annual -> $oa"
}

case "$STAGE" in
  train)    train_all ;;
  submit)   submit_all ;;
  ensemble) ensemble_all ;;
  all)      train_all; submit_all; ensemble_all ;;
  *) die "STAGE must be train | submit | ensemble | all" ;;
esac
