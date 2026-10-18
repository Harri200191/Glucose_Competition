#!/usr/bin/env bash
# The c96s recipe with the objective swapped to the exact expected-zone-A loss,
# trained on the FULL data with the annual-shaped cohort included.
#
#   STAGE=preflight scripts/run_c96t.sh    # checks only, ~3 min, RUN FIRST
#   STAGE=train scripts/run_c96t.sh
#   STAGE=submit scripts/run_c96t.sh       # COMPETITION defaults to both
#   STAGE=ensemble scripts/run_c96t.sh
#   BACKBONES="mamba2" EPOCHS=6 DRY_RUN=1 ...
#
# ===========================================================================
# WHAT THIS CHANGES FROM run_c96s.sh, AND WHY
#
# ONE thing: the training objective. Everything else is c96s's recipe, because
# c96s IS the production configuration and a second difference would make the
# comparison uninterpretable.
#
#     training.loss.name=hybrid  ->  zone_t
#     model.head.law.name        ->  student_t
#
# DTS zone A is an indicator of an interval: gradient zero almost everywhere,
# undefined on the boundary. It cannot be a loss. Its EXPECTATION under a
# Student-t predictive law can, and is closed form:
#
#     P(zone A) = F_nu((c + r - mu)/sigma) - F_nu((c - r - mu)/sigma)
#
# with c = -0.020411 and r = 0.202733 the zone-A band in log space, divided by
# the target standardiser's scale. No surrogate, no temperature. The t is not an
# arbitrary family: fitted to this project's own pooled log-residual it
# reproduces the OBSERVED zone-A rate to 0.27-1.17 points at every horizon,
# where a Gaussian is wrong by up to 8.71. Full derivation in
# docs/LOSS_OBJECTIVE_STUDY.md.
#
# MEASURED (results/loss, 19 arms x 2 seeds on the 640k distribution-faithful
# mini-corpus; test se on a 2-seed mean is 0.136, holdout se is 0.391 -- the
# holdout is 216k anchors but only 89 PATIENTS, so patient variance dominates
# and test is the low-noise ruler at this scale):
#
#     arm        test    holdout     what it is
#     mix3      +0.52     +0.51      student_t, K=3 mixture
#     tlog      +0.46     +0.60      student_t, -log P instead of 1-P
#     tpoint    +0.42     +0.41      student_t + Huber on the decision
#     sharp     +0.42     +0.49      student_t + sharpness penalty on log sigma
#     t         +0.40     +0.54      student_t, K=1
#     welsch    +0.34     +0.36      redescending point loss (NOT used here, see below)
#     ox        +0.15     +0.20      the c96s delta, as a control
#     ema       -0.01     -0.00      weight averaging
#     rmsesel   -0.11     +0.18      selecting best.pt on val RMSE not zone A
#
# The five student_t variants are indistinguishable from each other (+0.40 to
# +0.52 spans 0.12 against se 0.136) and are treated as ONE result. What matters
# is that it STACKS with ensembling rather than being absorbed by it:
#
#     ctrl 2-seed ensemble                              80.087 test / 79.723 hold
#     student_t family ensemble (5 variants x 2 seeds)  80.648      / 80.472
#     baseline family ensemble (ctrl+ox+band, 6 runs)   80.361      / 80.107
#                                                       ------------------------
#     objective gain that survives ensembling            +0.29      / +0.37
#
# AND IT PASSES THE CALIBRATION GATE, which is the thing that would have killed
# it. An arm whose zone A rises while its distribution gets worse has narrowed
# sigma to game the band, and is poison in an ensemble. Measured at h=120 on the
# holdout: the fan goes 57.9 -> 46.5 mg/dL (SHARPER) while q50 coverage goes
# 0.411 -> 0.468 (CLOSER to the nominal 0.500). It sharpened and re-centred at
# the same time. The oracle ceiling falls 98.4 -> 97.2 precisely BECAUSE the fan
# is tighter, not because the point was gamed.
#
# WHY NOT THE REDESCENDING LOSSES (welsch/tukey), WHICH ALSO GAINED. They are
# POINT losses, so nothing supervises the quantile ladder and it drifts: the
# measured q10-q90 fan at h=120 is 439.5 mg/dL against the control's 57.9. The
# point forecast is fine; the ladder is unusable, which rules those runs out of
# submission/decide.py, fan-based recalibration, and quantile-aware ensembling.
# Offered to a blend they added +0.04. Not worth a second axis.
#
# PER-BACKBONE VARIANT DIVERSITY, AND WHY IT IS FREE. The 5-variant ensemble beat
# a 2-seed single-variant ensemble by +0.12 for the same number of runs. So each
# backbone gets a DIFFERENT student_t variant rather than all three getting the
# same one. Same compute, more ensemble diversity.
#
# ===========================================================================
# THE TWO THINGS THIS RUN IS EXPLICITLY CONFIGURED FOR
#
# 1. THE ANNUAL-SHAPED COHORT IS IN TRAINING. internal_holdout.enabled=false, so
#    all ~660,000 IOBP2 anchors -- including the 89 patients normally held out --
#    are trained on. IOBP2 is the only source matching the annual competition set
#    (Dexcom G6 + iLet AID, no carb logging, median age 17), and the annual store
#    is 2,228 anchors over exactly 89 IOBP2-shaped patients.
#
#    *** THEREFORE ANY INTERNAL-HOLDOUT SCORE FROM THIS RUN IS MEANINGLESS. ***
#    It is memorisation, not generalisation. The preflight prints the number of
#    IOBP2 anchors that entered training so this is on the record, and the report
#    refuses to print a holdout score at all.
#
# 2. FULL DATA. max_train_windows=null. The preflight asserts it, because a cap
#    is the one setting that would quietly turn this into a different experiment.
#
#    A TENSION WORTH STATING. docs/experiments/trend_study/ measured ss_iobp2 --
#    IOBP2 up-weighted to ~52% -- at +13.75 on the annual-shaped ruler, by far
#    the largest effect anyone has found here. It does NOT transfer to full
#    scale, and the arithmetic is the reason: a boolean anchor mask cannot
#    duplicate a row, so a share is reached only by DISCARDING the other cohorts.
#    There are ~660k IOBP2 anchors against a ~10.6M post-drop pool, so
#
#        IOBP2 at  6% (natural)   ->  10.6M windows   <- this script
#        IOBP2 at 20%             ->   3.3M windows   (69% of the data thrown away)
#        IOBP2 at 55%             ->   1.2M windows   (89% thrown away)
#
#    ss_iobp2's +13.75 was measured at a FIXED 640k budget where every arm paid
#    the same cap, so up-weighting was free. Here it is not free, and no one has
#    measured the mix against the data-volume loss at this scale. IOBP2_BOOST=<pct>
#    exists to run that experiment deliberately; it is OFF by default and the
#    preflight prints exactly how many windows it would cost.
#
# ===========================================================================
# CARRIED OVER FROM run_c96s.sh, WITH ITS MEASUREMENT
#
#   context 1152, chain=dist, horizon_weights, future insulin, 7 quantiles,
#   film fusion, horizon_cross head, per-backbone lr and batch size,
#   official split / val_frac 0.12 / seed 0
#
# NOT CARRIED OVER, AND WHY
#
#   use_source_identity        +0.460 on live, and UNUSABLE for annual. Verified:
#                              the training and live stores share 14 source_file
#                              levels in the same order, but the annual store has
#                              exactly ONE level, 'annual_competition_secret_
#                              holdout_set', which the embedding has never seen.
#                              SRCID defaults to false here BECAUSE this run is
#                              aimed at the annual set. SRCID=true makes it a
#                              live-only model and the preflight will refuse to
#                              let it also target annual.
#
#   training.loss.delta        the ox_half vector is a Huber knob. Under zone_t
#                              the point is driven by the band term, and delta is
#                              read only when point_weight>0 (the tpoint variant).
#
#   min_delta=0.02             c96s inherited it, and EXPERIMENT_LEDGER.md sec.4
#                              traces a +1.68 result that evaporated to +0.25 to
#                              this setting stopping two IDENTICAL configs at 7
#                              and 15 epochs. Set to 0.0 here. This is a
#                              deliberate deviation from c96s; it means a c96t
#                              run may train longer than its c96s twin.
#
#   d_model=256                the c96s/c96w capacity pair is still running and
#                              has not answered. Epoch-matched val RMSE so far:
#                              transformerpp d128 27.88 best vs d256 28.71 best;
#                              mamba2 d128 28.55/28.39 vs d256 28.37/28.18 at
#                              epochs 1-2. Mixed and too early, so this uses 128
#                              -- the incumbent, and half the compute. Revisit
#                              when c96w finishes.
#
# ===========================================================================
# WHAT c96s TAUGHT US, AND WHY THE COHORT DROP IS OFF BY DEFAULT
#
# c96s applied three changes the mini-corpus screen liked (srcid +0.460, cohort
# drop +0.486, ox_half delta +0.280; combo predicted +0.616). Only
# c96s_transformerpp finished. Measured against its c96_transformerpp twin, same
# store, same context, same width, same 395,129 test anchors:
#
#     plain test split      83.903 -> 81.596    -2.31
#     test@livemix          81.210 -> 81.077    -0.13
#
# The -2.31 is almost entirely an ARTIFACT: the plain test split still carries
# 11.6% of its anchors in DCLP3/DCLP5/Flair, cohorts c96s deliberately never
# trained on. On the board-shaped ruler the three changes bought -0.13, i.e.
# nothing. Never score a cohort-drop arm on the plain split.
#
# But the drop did real damage beyond scoring, and this is the part that matters:
#
#   1. AT FULL SCALE A DROP IS A DATA LOSS, NOT A REMIX. The screen measured
#      droprare at a FIXED 640k budget, where removing a cohort meant the sampler
#      SUBSTITUTED more of the kept ones -- same volume, better mix. Uncapped,
#      the same config just deletes 13.4% of the training set with nothing
#      replacing it. Every mix result from the capped mini-corpus has this
#      property, which is also why IOBP2_BOOST below is off.
#
#   2. IT DROPS FROM TRAIN BUT NOT FROM VAL, so val_rmse -- the thing early
#      stopping watches -- is measured partly on cohorts the model was told to
#      ignore. c96s_transformerpp starts 1.2 mg/dL behind at epoch 1 and never
#      recovers, and with min_delta=0.02 it early-stopped on its EPOCH 3
#      checkpoint (29.03 28.45 27.88 28.21 27.94 28.00) while its c96 twin was
#      still descending when it hit the epoch cap (27.83 27.18 26.59 26.59 26.42).
#
# So this script is c96's recipe plus the objective, NOT c96s's. The incumbent to
# beat is c96_mambaformer at 83.963 test / 81.311 livemix, and the c96 four-model
# ensemble at 85.466 live.
#
# DROP_ABSENT=true restores the c96s behaviour. If you use it, also cap the data
# so the drop is a remix rather than a deletion, and expect the val metric to be
# measured on ignored cohorts.
#
# RESUME. Every run writes last.pt at the end of each epoch -- weights,
# optimiser, LR-schedule position, AMP scaler, early-stopping counters, EMA,
# epoch number -- and re-running the same command picks up where it stopped.
# This exists because three of the four c96s/c96w runs were killed mid-epoch
# after 80+ hours and lost everything: best.pt alone cannot continue a run, since
# restarting from it resets the cosine schedule to step 0 and throws away the
# optimiser moments. A config fingerprint is stored with it and a mismatch RAISES
# rather than silently continuing a different experiment from someone else's
# weights. Changing only training.epochs is allowed (extending a run is a
# legitimate resume) and warns, because the cosine then decays over a new horizon.
#
#     STAGE=train scripts/run_c96t.sh     # re-run after any interruption
#     ... --no-resume                     # force a fresh start
#
# EPOCHS. c96_transformerpp was STILL IMPROVING at epoch 5 when its cap hit, so
# simply training longer may be worth more than anything in this script. EPOCHS
# is 10 here with min_delta=0.0, so runs stop when they genuinely plateau.
#
# ===========================================================================
# COST. Median h/epoch observed on ctx1152 at d_model=128, already net of the
# cohort drop, plus ~3.6% for the distributional loss (measured: 6.6 ms against
# a 187 ms step, flat in K):
#
#     transformerpp 17.8   mambaformer 24.1   mamba2 14.2   h/epoch
#     10 epochs  ->  7.4 / 10.0 / 5.9 days, ~10 days wall on three GPUs
#
# The c96s and c96w runs are already holding GPUs. Check the box before starting.
# EPOCHS is an upper bound; patience stops earlier when val zone A plateaus.
# ===========================================================================

set -euo pipefail
_USER_STORE="${STORE:-}"
source "$(dirname "$0")/lib/common.sh"
STORE="${_USER_STORE:-$REPO_ROOT/artifacts/store/ctx1152}"

STAGE="${STAGE:-all}"
CONTEXT="${CONTEXT:-1152}"
EPOCHS="${EPOCHS:-10}"
PATIENCE="${PATIENCE:-3}"
MIN_DELTA="${MIN_DELTA:-0.0}"
SEED="${SEED:-0}"
NEED_MB="${NEED_MB:-24000}"
RUNS_DIR="${RUNS_DIR:-$REPO_ROOT/runs}"
SUB_DIR="${SUB_DIR:-$REPO_ROOT/artifacts/submission}"
TAG="${TAG:-c96t}"
OUT="${OUT:-$REPO_ROOT/results/$TAG}"
D_MODEL="${D_MODEL:-128}"
SRCID="${SRCID:-false}"                      # false = annual-safe. See the header.
COMPETITION="${COMPETITION:-both}"           # annual | live | both
HW="${HW:-[1,1,1.5,2]}"
LR_SCALE="${LR_SCALE:-1.0}"
BACKBONES="${BACKBONES:-transformerpp mambaformer mamba2}"
QUANTS="[0.05,0.1,0.25,0.5,0.75,0.9,0.95]"
ABSENT="['DCLP3','DCLP5','Flair','T1D-UOM','OhioT1DM']"
DROP_ABSENT="${DROP_ABSENT:-false}"   # OFF by default -- see "WHAT c96s TAUGHT US"
WEIGHT_STEP="${WEIGHT_STEP:-0.1}"
IOBP2_BOOST="${IOBP2_BOOST:-0}"              # 0 = off. See the header before using.
SELECT_METRIC="${SELECT_METRIC:-val_zone_a}"

lr_of() { case "$1" in transformerpp) echo 0.0007;; mambaformer) echo 0.0005;;
                       mamba2) echo 0.0004;; moderntcn) echo 0.0014;;
                       *) die "no lr for $1";; esac; }
bs_of() { case "$1" in transformerpp) echo 144;; mambaformer) echo 72;;
                       mamba2) echo 72;; moderntcn) echo 144;;
                       *) die "no batch size for $1";; esac; }

# Each backbone gets a DIFFERENT student_t variant. The five variants are
# statistically indistinguishable (0.12 spread against se 0.136), so nothing is
# lost by spreading them, and the 5-variant ensemble beat a single-variant one by
# +0.12 for the same run count.
law_of() {
  case "$1" in
    transformerpp) echo "model.head.law.k=3" ;;                                  # mix3, +0.52
    mambaformer)   echo "model.head.law.k=1" ;;                                  # t,    +0.40
    mamba2)        echo "model.head.law.k=1 training.loss.zone_obj=logprob" ;;   # tlog, +0.46
    moderntcn)     echo "model.head.law.k=1 training.loss.sharpness_weight=0.1" ;;  # sharp, +0.42
    *) die "no law variant for $1" ;;
  esac
}
variant_of() {
  case "$1" in transformerpp) echo mix3;; mambaformer) echo t;;
               mamba2) echo tlog;; moderntcn) echo sharp;; *) echo "?";; esac
}

mkdir -p "$OUT"
read -r -a BB <<< "$BACKBONES"

preflight() {
  log "tag=$TAG  d_model=$D_MODEL  epochs<=$EPOCHS  srcid=$SRCID  competition=$COMPETITION"
  log "objective: zone_t (exact expected zone A under a Student-t)"
  for b in "${BB[@]}"; do log "  $b -> $(variant_of "$b")  [$(law_of "$b")]"; done
  log "store: $STORE"
  [[ -f "$STORE/manifest.json" ]] || die "no store at $STORE"
  local have
  have=$($PY -c "import json;print(json.load(open('$STORE/manifest.json'))['config']['min_history_steps'])")
  (( have >= CONTEXT )) || die "store guarantees $have steps, need $CONTEXT"
  log "  min_history_steps=$have >= $CONTEXT  OK"

  # The srcid/annual incompatibility, checked rather than trusted.
  if [[ "$SRCID" == "true" ]]; then
    [[ "$COMPETITION" == "live" ]] || die \
"SRCID=true feeds the cohort to the model as an embedding, and the ANNUAL store has
  exactly one source_file level ('annual_competition_secret_holdout_set') that the
  embedding has never seen. A model trained this way must not be scored there.
  Either set SRCID=false (annual-safe, the default) or COMPETITION=live."
    log "  source identity ON -- checking the live store shares the training vocabulary"
    $PY - "$STORE" "$REPO_ROOT/artifacts/store/submission_live" <<'PYEOF' || die "cohort vocabularies differ"
import json, sys
a = json.load(open(sys.argv[1] + "/manifest.json"))["levels"]["source_file"]
b = json.load(open(sys.argv[2] + "/manifest.json"))["levels"]["source_file"]
print(f"    train {len(a)} levels, live {len(b)} levels -- "
      + ("identical, same order  OK" if a == b else "MISMATCH"))
sys.exit(0 if a == b else 1)
PYEOF
    warn "  srcid=true is LIVE ONLY."
  else
    log "  source identity OFF -- annual-safe"
  fi

  log "  parameter count at d_model=$D_MODEL"
  $PY "$REPO_ROOT/scripts/analysis/_check_capacity.py" "$D_MODEL" "$SRCID" "${BB[@]}" \
    || die "could not build the models"

  # The two things this run is configured for, asserted rather than assumed:
  # the annual-shaped cohort is in training, and the data is not capped.
  log "checking the training set: full data, every cohort, IOBP2 present"
  $PY - "$STORE" "$IOBP2_BOOST" "$DROP_ABSENT" <<'PYEOF' || die "training-set preflight failed"
import sys, warnings, numpy as np
warnings.filterwarnings("ignore")
from metabonet_forecasting.data.store import SequenceStore
from metabonet_forecasting.data import splits as S
st = SequenceStore(sys.argv[1]); boost = float(sys.argv[2])
drop = sys.argv[3] == "true"
subj = st.anchors["subject"].to_numpy()
# anchors carry an integer `cohort` code; store.cohort_names() is its vocabulary.
coh = np.asarray(st.cohort_names())[st.anchors["cohort"].to_numpy()]
sp = S.make_split(st, "official", val_frac=0.12, seed=0)
tr = sp["train"]; t0 = tr.sum()
key, mn = S.mix_strata(st, "cohort", 120)
if drop:
    tr2, _ = S.stratified_subsample(
        tr, key, mn, target=None,
        drop=['DCLP3','DCLP5','Flair','T1D-UOM','OhioT1DM'],
        max_items=None, seed=0, subjects=subj)
else:
    tr2 = tr   # every cohort, uncapped: the c96 recipe
kept = set(np.unique(subj[tr2]).tolist()); rem = set(np.unique(subj[tr & ~tr2]).tolist())
n_io = int((coh[tr2] == "IOBP2").sum())
print(f"    train {t0:,} -> {tr2.sum():,}  ({100*tr2.sum()/t0:.1f}% kept, uncapped, "
      + ("5 cohorts dropped" if drop else "ALL cohorts kept") + ")")
names = np.unique(coh[tr2])
print(f"    cohorts in training: {len(names)}  {', '.join(sorted(names))}")
print(f"    patients kept {len(kept):,}  removed whole {len(rem):,}  in BOTH {len(kept & rem)}")
print(f"    val {sp['val'].sum():,} and test {sp['test'].sum():,} untouched")
assert not (kept & rem), "a patient was cut in half -- stop"
print(f"    IOBP2 IN TRAINING: {n_io:,} anchors = {100*n_io/tr2.sum():.2f}% of the training set")
assert n_io > 0, "IOBP2 absent from training -- internal_holdout.enabled must be false"
print(f"    (the 89 normally-held-out patients are INCLUDED; no holdout score is valid)")
if boost > 0:
    cap = int(n_io / (boost / 100.0))
    print(f"    IOBP2_BOOST={boost:g}% would need max_train_windows<={cap:,} "
          f"-- discarding {100*(1-cap/tr2.sum()):.0f}% of the training data")
PYEOF
  log "preflight OK"
}

train_all() {
  gpu_autoconfig "$NEED_MB"; queue_init
  for b in "${BB[@]}"; do
    local name="${TAG}_${b}"
    if [[ -f "$RUNS_DIR/$name/metrics.json" && "${SKIP_EXISTING:-1}" == "1" ]]; then
      log "skip $name (already finished; SKIP_EXISTING=0 to force)"; continue
    fi
    local lr bs law mix_args
    lr=$($PY -c "print(round($(lr_of "$b") * $LR_SCALE, 8))"); bs=$(bs_of "$b")
    read -r -a law <<< "$(law_of "$b")"
    # Uncapped by default. A boost needs a cap, because a boolean mask can only
    # down-sample -- the header does that arithmetic.
    mix_args=("data.sampling.max_train_windows=null")
    if [[ "$DROP_ABSENT" == "true" ]]; then
      mix_args+=("data.sampling.mix.by=cohort" "data.sampling.mix.drop=$ABSENT")
    fi
    if [[ "$IOBP2_BOOST" != "0" ]]; then
      warn "IOBP2_BOOST=$IOBP2_BOOST%: this CAPS the training set. Not full data."
      mix_args=("data.sampling.max_train_windows=${BOOST_WINDOWS:?set BOOST_WINDOWS with IOBP2_BOOST}")
    fi
    log "queueing $name  $(variant_of "$b")  d_model=$D_MODEL  lr $lr  bs $bs  epochs<=$EPOCHS"
    queue_run "$name" $PY -m metabonet_forecasting.training.train \
      --config "model/$b" --name "$name" --runs-dir "$RUNS_DIR" \
      --set \
        "seed=$SEED" \
        "data.store=$STORE" \
        "data.context=$CONTEXT" \
        "data.target_mode=log_ratio" \
        "data.future_modalities=['insulin']" \
        "data.use_source_identity=$SRCID" \
        "data.split.scheme=official" \
        "data.split.val_frac=0.12" \
        "data.split.seed=0" \
        "data.internal_holdout.enabled=false" \
        "${mix_args[@]}" \
        "data.sampling.max_val_windows=150000" \
        "data.loader.batch_size=$bs" \
        "data.loader.num_workers=8" \
        "model.d_model=$D_MODEL" \
        "model.fusion.name=film" \
        "model.head.name=horizon_cross" \
        "model.head.chain=dist" \
        "model.head.quantiles=$QUANTS" \
        "model.head.law.name=student_t" \
        "${law[@]}" \
        "training.loss.name=zone_t" \
        "training.loss.horizon_weights=$HW" \
        "training.epochs=$EPOCHS" \
        "training.patience=$PATIENCE" \
        "training.min_delta=$MIN_DELTA" \
        "training.select_metric=$SELECT_METRIC" \
        "training.resume=true" \
        "training.lr=$lr" \
        "training.predict_val=true"
  done
  queue_wait
}

submit_all() {
  for b in "${BB[@]}"; do
    local name="${TAG}_${b}"
    [[ -f "$RUNS_DIR/$name/best.pt" ]] || { warn "no checkpoint for $name, skipping"; continue; }
    log "submission predictions for $name (COMPETITION=$COMPETITION, ~75 min each)"
    COMPETITION="$COMPETITION" "$REPO_ROOT/scripts/90_make_submission.sh" "$name"
  done
}

ensemble_all() {
  local targets="$REPO_ROOT/toolkit/data/live_leaderboard/targets.parquet"
  local mem=()
  for b in "${BB[@]}"; do
    [[ -f "$SUB_DIR/live_${TAG}_${b}.parquet" ]] && mem+=("$SUB_DIR/live_${TAG}_${b}.parquet")
  done
  if (( ${#mem[@]} )); then
    log "live ensemble over ${#mem[@]} members (--weight-step $WEIGHT_STEP)"
    $PY -m metabonet_forecasting.submission.recalibrate \
      --live "${mem[@]}" --targets "$targets" \
      --out-live "$SUB_DIR/live_${TAG}_ens_affw.parquet" \
      --affine --fit-weights --weight-step "$WEIGHT_STEP" | tee "$OUT/ensemble.txt"
    log "scoring against the current best"
    $PY "$REPO_ROOT/scripts/analysis/score_live.py" "${mem[@]}" \
      "$SUB_DIR/live_${TAG}_ens_affw.parquet" \
      "$SUB_DIR/live_c96_ens4_affw.parquet" | tee "$OUT/FINAL.txt"
  else
    warn "no live parquets; run STAGE=submit first (or COMPETITION included live)"
  fi

  # The annual set has no public answer key, so weights and per-bin corrections
  # cannot be fitted against it. recalibrate.py's multipliers TRANSFER (+0.363
  # measured on held-out SUBJECTS, and 0.006-0.058 when carried across
  # populations), so the live-fitted correction is applied, not refitted.
  local amem=()
  for b in "${BB[@]}"; do
    [[ -f "$SUB_DIR/annual_${TAG}_${b}.parquet" ]] && amem+=("$SUB_DIR/annual_${TAG}_${b}.parquet")
  done
  if (( ${#amem[@]} )); then
    log "annual ensemble over ${#amem[@]} members (weights TRANSFERRED from live, never fitted here)"
    $PY -m metabonet_forecasting.submission.recalibrate \
      --live "${mem[@]}" --annual "${amem[@]}" --targets "$targets" \
      --out-live "$SUB_DIR/live_${TAG}_ens_affw.parquet" \
      --out-annual "$SUB_DIR/annual_${TAG}_ens_affw.parquet" \
      --affine --fit-weights --weight-step "$WEIGHT_STEP" | tee -a "$OUT/ensemble.txt"
  fi

  cat <<EOF | tee -a "$OUT/FINAL.txt"

HOW TO READ THIS
  live_c96_ens4_affw = 85.466 is the number to beat on the live board.

  Read the SINGLE models before the ensemble. If ${TAG}_<b> does not beat its
  c96s_<b> twin on its own, the objective change did not carry to full scale and
  the ensemble is hiding it. That is the whole point of this run.

  NOTE the two arms are NOT a clean pair: c96s has srcid=true (+0.460, live
  only) and min_delta=0.02, this run has srcid=false and min_delta=0.0. On the
  LIVE board c96s therefore starts ~0.5 ahead on a setting this run cannot use.
  The honest comparison is SRCID=true COMPETITION=live scripts/run_c96t.sh, or
  compare on the annual submission where neither can use srcid.

  *** DO NOT REPORT AN INTERNAL-HOLDOUT SCORE. *** internal_holdout.enabled is
  false, so the 89 annual-shaped patients are in training by design. Any score
  on them is memorisation. The measured size of that inflation is +8.7 points
  (docs/EXPERIMENT_LEDGER.md sec.3), so it will look like a triumph and mean
  nothing.

  Expected from the mini-corpus, which is what this is testing at scale:
  +0.4 per single model, +0.29 surviving into the ensemble. Against ~0.6
  run-to-run noise at full scale, a single model's gain is NOT separable from
  noise -- the ensemble comparison and the live board are.
EOF
  log "done -> $OUT/FINAL.txt"
}

case "$STAGE" in
  preflight) preflight ;;
  train)     preflight; train_all ;;
  submit)    submit_all ;;
  ensemble)  ensemble_all ;;
  all)       preflight; train_all; submit_all; ensemble_all ;;
  *) die "STAGE must be preflight|train|submit|ensemble|all" ;;
esac
