#!/usr/bin/env bash
# 96-hour context, everything in the training pool, competition splits only.
#
#   scripts/run_ctx1152.sh                 # store check -> 4 backbones -> ensemble -> submission
#   STAGE=store  scripts/run_ctx1152.sh    # build the 1152-step store, then stop
#   STAGE=train  scripts/run_ctx1152.sh    # train only
#   STAGE=report scripts/run_ctx1152.sh    # ensemble + score + build the live parquet
#
# ===========================================================================
# WHAT THIS RUN IS
#
# A SUBMISSION run, not a benchmark run. Three things follow from that and all
# three are deliberate:
#
#   1. context 1152 steps = 96 h. Context is the only intervention in this
#      project that repeatedly beat the 0.6 noise floor: 24 h -> 48 h -> 72 h
#      gave 80.16 -> 81.06 -> 81.56 on a fixed anchor set, and the 72 h ensemble
#      beat the 24 h one by +1.63 on matched anchors (+2.38 at 120 min). It was
#      still climbing at 72 h, so this asks for one more step.
#
#   2. THE INTERNAL HOLDOUT IS IN TRAINING, and so is all of IOBP2. Every other
#      script in this repo goes to some length to keep them out. This one does
#      the opposite on purpose, because a submission should be trained on every
#      row it is allowed to see.
#
#   3. splits are the competition's own (`split.scheme=official`) and nothing
#      else. No internal holdout carve-out, so the official test split is
#      byte-identical to the one the organisers define.
#
# WHAT THIS COSTS YOU, AND IT IS NOT NEGOTIABLE
#
#   ***  THE INTERNAL HOLDOUT NUMBER FROM THIS RUN IS MEANINGLESS.  ***
#
# Those 89 patients are in the training set. Any holdout score this run produces
# is measuring memorisation. We have quantified exactly this: with IOBP2 in
# training the same backbone scored +9.93 on the IOBP2 holdout but only +1.24 on
# an independent test split -- about 8.7 points of pure domain memorisation.
#
# So this run has exactly TWO honest readouts:
#   * the official test split, printed below;
#   * the live leaderboard, whose targets are public (scripts/analysis/score_live.py).
# Do not put this run's holdout number in a table next to a clean run's. The
# integrity check every other script runs is SKIPPED here, because it would fail
# by design; that skip is the reason this banner exists.
#
# THE RECIPE: the b72 recipe plus chain=dist.
#   log_ratio target        zone A is a +/-20% RELATIVE band
#   horizon weights         30 min is near its ceiling, 120 min has the headroom
#   horizon_cross head      per-horizon cross-attention over history and doses
#   chain=dist              the predicted DISTRIBUTION (not the point) feeds the
#                           next horizon. Measured +0.39 at full data with the
#                           gain concentrated at 90/120 min, where this run's
#                           context increase also lands.
#   future insulin          worth +12.45 zone A, the single largest effect we
#                           have measured for anything
#   4-backbone ensemble     ~+0.7, the most reproducible gain after context
#   NO recentring           costs -0.89 on log_ratio runs
#
# THE STORE MUST BE REBUILT. 96 h = 1152 steps and windows.py refuses any
# context above the store's guaranteed history; ctx864 guarantees 864. The build
# needs ~14 GB of disk and a large transient chunk of RAM -- an earlier attempt
# was OOM-killed while another user held 493 GB, so check `free -g` first and
# re-run STAGE=store if it dies. It is idempotent.
# ===========================================================================

set -euo pipefail
_USER_STORE="${STORE:-}"
source "$(dirname "$0")/lib/common.sh"
STORE="${_USER_STORE:-$REPO_ROOT/artifacts/store/ctx1152}"

CONTEXT="${CONTEXT:-1152}"          # 96 h at the 5-minute grid
NEED_MB="${NEED_MB:-20000}"
EPOCHS="${EPOCHS:-5}"
PATIENCE="${PATIENCE:-3}"
STAGE="${STAGE:-all}"
RUNS_DIR="${RUNS_DIR:-$REPO_ROOT/runs}"
OUT="${OUT:-$REPO_ROOT/results/ctx1152}"
TAG="${TAG:-c96}"
SEED="${SEED:-0}"
WINDOWS="${WINDOWS:-null}"          # null = every anchor in the store
HW="${HW:-[1,1,1.5,2]}"
CHAIN="${CHAIN:-dist}"
TEMPLATE="${TEMPLATE:-$REPO_ROOT/artifacts/submission/live_template.parquet}"
LIVE_STORE="${LIVE_STORE:-$REPO_ROOT/artifacts/store/submission_live}"
TARGETS="${TARGETS:-$REPO_ROOT/toolkit/data/live_leaderboard/targets.parquet}"
QUANTS="[0.05,0.1,0.25,0.5,0.75,0.9,0.95]"

# ---------------------------------------------------------------- the store
build_store() {
  log "building $STORE (min-history 1152 = 96 h, dirty history allowed)"
  log "  free RAM now: $(free -g | awk '/^Mem:/{print $7}') GB. An earlier 864-step"
  log "  build was OOM-killed at ~60% while another user held 493 GB."
  $PY -m metabonet_forecasting.data.prepare \
    --corpus "$REPO_ROOT/data/metabonet_public.parquet" --out "$STORE" \
    --min-history-steps 1152 --anchor-stride 6 \
    --max-cgm-gap-min 30 --interp-gap-min 30 --allow-dirty-history \
    || die "store build failed (most likely OOM). Check 'free -g' and re-run:
  STAGE=store $0"
  log "store built -> $STORE"
}

if [[ "$STAGE" == "store" ]]; then build_store; exit 0; fi

if [[ ! -f "$STORE/manifest.json" ]]; then
  warn "no store at $STORE -- 96-hour windows need 1152 steps of guaranteed"
  warn "history and the code refuses anything longer than the store promises."
  build_store
fi

# A store that exists but was built for a shorter context is the quiet failure
# mode here: training would abort 20 minutes in, after the data load.
_have=$($PY -c "import json;print(json.load(open('$STORE/manifest.json'))['config']['min_history_steps'])")
(( _have >= CONTEXT )) || die \
"store $STORE guarantees only $_have steps of history but this run needs $CONTEXT.
Rebuild it with:  STAGE=store $0"

# ------------------------------------------------------------------- config
# NOTE the two lines that make this a submission run rather than a benchmark
# run. They are the whole difference, and they are why the holdout number this
# produces must not be compared with anything else in results/.
CORE=(
  "seed=$SEED"
  "data.store=$STORE"
  "data.context=$CONTEXT"
  "data.target_mode=log_ratio"
  "data.future_modalities=['insulin']"
  "data.use_source_identity=false"
  "data.split.scheme=official"          # the competition's own split, nothing else
  "data.internal_holdout.enabled=false" # <- holdout patients TRAIN here
  "data.sampling.max_train_windows=$WINDOWS"
  "data.sampling.max_val_windows=150000"
  "data.loader.num_workers=8"
  "model.fusion.name=film"
  "model.head.name=horizon_cross"
  "model.head.chain=$CHAIN"
  "model.head.quantiles=$QUANTS"
  "training.loss.name=hybrid"
  "training.loss.quantiles=$QUANTS"
  "training.loss.horizon_weights=$HW"
  "training.loss.alpha=0.5"
  "training.epochs=$EPOCHS"
  "training.patience=$PATIENCE"
  "training.min_delta=0.02"
  "training.predict_val=true"
)

# backbone : batch : lr -- ranked by their 72-hour scores. Batch shrinks again:
# 1152 steps is 4x the sequence of the 288-step runs and 1.33x the 864 ones.
DEFAULT_MEMBERS="model/mamba2:72:0.0008 model/transformerpp:144:0.0007 model/mambaformer:72:0.0010 model/moderntcn:144:0.0014"
read -r -a MEMBERS <<< "${MEMBERS_SPEC:-${MEMBERS:-$DEFAULT_MEMBERS}}"
NAMES=(); for s in "${MEMBERS[@]}"; do c="${s%%:*}"; NAMES+=("${TAG}_${c##*/}"); done
mkdir -p "$OUT"

# ------------------------------------------------------------- cost notice
$PY - "$STORE" "$CONTEXT" "$EPOCHS" "${#MEMBERS[@]}" <<'PYEOF'
import json, sys
store, ctx, ep, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
m = json.load(open(f"{store}/manifest.json"))
anch = m.get("n_anchors", 0)
print(f"\n  store {store}")
print(f"  {anch:,} anchors, ALL of them in the training pool")
print(f"  context {ctx} steps = {ctx*5/60:.0f} h")
print(f"  measured at 864: 2.3-3.4 h/epoch/backbone on 15.3M anchors")
print(f"  this run scales that by ~1.33 (context) x ~1.41 (data) x {ep}/4 (epochs)")
print(f"  -> roughly 22-32 h per backbone, {n} in parallel, so ~32 h wall clock.")
print(f"     Early stopping (patience 3) usually cuts it. WINDOWS= or EPOCHS= trade down.\n")
PYEOF

cat <<'EOF'
  ============================================================
   THE INTERNAL HOLDOUT IS IN TRAINING IN THIS RUN.
   Its holdout score measures memorisation and must NOT be
   compared with any other run in results/. The honest
   readouts are the official test split and the live board.
  ============================================================

EOF

# ------------------------------------------------------------------- train
if [[ "$STAGE" == "all" || "$STAGE" == "train" ]]; then
  gpu_autoconfig "$NEED_MB"; queue_init
  for spec in "${MEMBERS[@]}"; do
    IFS=: read -r cfg bs lr <<< "$spec"
    n="${TAG}_${cfg##*/}"
    log "queueing $n  (${cfg##*/}, context $CONTEXT, chain=$CHAIN, batch $bs, lr $lr)"
    train "$n" "$cfg" "${CORE[@]}" "data.loader.batch_size=$bs" "training.lr=$lr"
  done
  queue_wait
  [[ "$STAGE" == "train" ]] && { log "STAGE=train: trained only. Score with STAGE=report $0"; exit 0; }
fi

# ------------------------------------------------------------------ report
PRESENT=()
for n in "${NAMES[@]}"; do
  [[ -f "$RUNS_DIR/$n/metrics.json" ]] && PRESENT+=("$RUNS_DIR/$n")
done
(( ${#PRESENT[@]} )) || die "no member finished; check $LOG_DIR"
log "members finished: ${#PRESENT[@]}/${#NAMES[@]}"

# Official test split, the only in-repo number from this run worth reading.
log "official test split (the honest readout)"
$PY - "$OUT" "${PRESENT[@]}" <<'PYEOF' | tee "$OUT/test_scores.txt"
import sys, warnings, csv, os
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
out, runs = sys.argv[1], sys.argv[2:]
rows = []
for r in runs:
    p = os.path.join(r, "metrics.csv")
    if not os.path.exists(p): continue
    d = pd.read_csv(p)
    d = d[(d.split == "test") & (d.stratum == "all") & (d["slice"] == "overall")]
    if d.empty: continue
    rec = {"run": os.path.basename(r)}
    vals = []
    for h in (30, 60, 90, 120):
        v = d[d.horizon == h]
        if v.empty: continue
        rec[f"test@{h}"] = round(float(v.iloc[0]["dts_A"]), 2); vals.append(rec[f"test@{h}"])
    if len(vals) == 4: rec["test_mean"] = round(float(np.mean(vals)), 3)
    rows.append(rec)
df = pd.DataFrame(rows).sort_values("test_mean", ascending=False)
pd.set_option("display.width", 200)
print(df.to_string(index=False))
df.to_csv(os.path.join(out, "test_scores.csv"), index=False)
PYEOF

# --------------------------------------------------------------- submission
# One live parquet per member, then a plain mean ensemble. No recalibration
# here: recentring measured -0.89 on log_ratio runs, and fitting multipliers on
# the live answer key is a separate decision that should be made deliberately
# (metabonet_forecasting.submission.recalibrate) rather than folded in silently.
if [[ -f "$TEMPLATE" && -d "$LIVE_STORE" ]]; then
  LIVE=()
  for d in "${PRESENT[@]}"; do
    n=$(basename "$d"); f="$REPO_ROOT/artifacts/submission/live_${n}.parquet"
    if [[ ! -f "$f" ]]; then
      log "live predictions for $n"
      # pad_history=True inside predict.py left-pads template rows with less
      # than $CONTEXT of history -- which at 1152 is a LOT of rows. The coverage
      # and score printed below are what say whether that hurt.
      $PY -m metabonet_forecasting.submission.predict \
        --run "$d" --store "$LIVE_STORE" --template "$TEMPLATE" \
        --out "$f" --batch-size 128 || { warn "live predict failed for $n"; continue; }
    fi
    LIVE+=("$f")
  done
  if (( ${#LIVE[@]} )); then
    ENSF="$REPO_ROOT/artifacts/submission/live_${TAG}_ens.parquet"
    log "ensembling ${#LIVE[@]} member(s) -> $ENSF"
    # --targets is required by the CLI even under --no-recal (it is the ensemble
    # path, not the fitting path, that we want here).
    $PY -m metabonet_forecasting.submission.recalibrate \
      --live "${LIVE[@]}" --no-recal --targets "$TARGETS" --out-live "$ENSF" >/dev/null \
      || warn "ensemble step failed"
    [[ -f "$TARGETS" && -f "$ENSF" ]] && {
      log "scoring against the live answer key"
      $PY "$REPO_ROOT/scripts/analysis/score_live.py" "${LIVE[@]}" "$ENSF" \
        | tee "$OUT/live_scores.txt"; }
    log "SUBMIT THIS FILE -> $ENSF"
  fi
else
  warn "no live template or store; skipping the submission build"
fi
log "done -> $OUT"
