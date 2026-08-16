#!/usr/bin/env bash
# Measure what "knowing the future" is worth -- and why that number is not a
# forecasting result.
#
#   scripts/35_future_covariates.sh                       # one model, 4 settings
#   BACKBONE=tcn FUSION=film scripts/35_future_covariates.sh
#   RUN_BASELINES=1 scripts/35_future_covariates.sh       # + GBDT, the leaderboard recipe
#
#   ARMS=both WITH_CAUSAL=0 scripts/35_future_covariates.sh   # just the +both arm
#   ARMS="insulin carbs" scripts/35_future_covariates.sh      # skip the combined arm
#
# Runs the same architecture four times by default -- causal, +future insulin,
# +future carbs, +both -- with everything else held fixed, then prints the
# deltas. ARMS selects which exposed arms to run; WITH_CAUSAL=0 drops the causal
# reference (only sensible when you already have a matched one, since the whole
# point is the difference between two runs that differ in one thing).
#
# What the numbers mean. A forecast answers "given what is known now, what will
# glucose be?". Reading insulin or carbohydrate values from after the anchor
# answers a different question: "given that this dose is delivered, what will
# glucose be?". Both are legitimate; they are not comparable, and only the first
# one is a forecast.
#
# In this corpus the distinction is not academic. 85.8% of rows are AID
# (closed-loop, Loop alone is 61.2%), where delivery at t+k is computed by a
# control algorithm from the CGM at t+k. Future insulin there is not a plan the
# patient made -- it is the label, transformed. Expect the insulin arm to gain
# far more than the carbohydrate arm, expect the gain to be largest at 30 min
# where the controller's reaction is most tightly coupled to the outcome, and
# expect it to be concentrated in AID cohorts. That pattern *is* the result: it
# is evidence of leakage, not of a better model.
#
# Carbohydrate is the more interesting arm scientifically. A meal announced
# before eating is genuinely known at inference, so a gain there is partly real
# -- but this corpus timestamps when the entry was logged, not when it was
# announced, so even that arm mixes the two.
#
# Anything produced here belongs in a separate table from the benchmark, clearly
# labelled. Never merge it into the headline comparison.

source "$(dirname "$0")/lib/common.sh"
require_store

BACKBONE="${BACKBONE:-mamba2}"
FUSION="${FUSION:-film}"
SEEDS="${SEEDS:-0}"
CONTEXT="${CONTEXT:-288}"
RUN_BASELINES="${RUN_BASELINES:-0}"
TAG="${TAG:-e9}"
ARMS="${ARMS:-insulin carbs both}"
WITH_CAUSAL="${WITH_CAUSAL:-1}"

for arm in $ARMS; do
  case "$arm" in
    insulin|carbs|both) ;;
    *) die "ARMS must be drawn from: insulin carbs both (got '$arm')" ;;
  esac
done

log "future-covariate study: ${BACKBONE}/${FUSION} at context ${CONTEXT}, seeds [${SEEDS}]"
log "arms: [${ARMS}]  causal reference: $([[ "$WITH_CAUSAL" == "1" ]] && echo yes || echo no)"

queue_init
for seed in $SEEDS; do
  base=("seed=${seed}" "data.context=${CONTEXT}" "model.fusion.name=${FUSION}")
  name="${TAG}_${BACKBONE}_${FUSION}_s${seed}"

  # The causal reference. No FUTURE tag, so this is the ordinary benchmark run
  # and will be skipped if an identical one already exists.
  if [[ "$WITH_CAUSAL" == "1" ]]; then
    FUTURE=none  _FUTURE_SET=""  _FUTURE_TAG="" \
      train "$name" "model/${BACKBONE}" "${base[@]}"
  fi

  for setting in $ARMS; do
    case "$setting" in
      insulin) fset="data.future_modalities=['insulin']";            ftag="_futins" ;;
      carbs)   fset="data.future_modalities=['carbs']";              ftag="_futcarb" ;;
      both)    fset="data.future_modalities=['insulin','carbs']";    ftag="_futboth" ;;
    esac
    _FUTURE_SET="$fset" _FUTURE_TAG="$ftag" \
      train "$name" "model/${BACKBONE}" "${base[@]}"
  done
done

if [[ "$RUN_BASELINES" == "1" ]]; then
  log "gradient-boosted trees, same four settings -- the cheapest way to buy a score"
  for setting in none insulin carbs both; do
    case "$setting" in
      none)    fset="";                                            ftag="" ;;
      insulin) fset="data.future_modalities=['insulin']";          ftag="_futins" ;;
      carbs)   fset="data.future_modalities=['carbs']";            ftag="_futcarb" ;;
      both)    fset="data.future_modalities=['insulin','carbs']";  ftag="_futboth" ;;
    esac
    _FUTURE_SET="$fset" _FUTURE_TAG="$ftag" \
      baseline "${TAG}_gbdt" experiment/e1_baselines gbdt "data.context=${CONTEXT}"
  done
fi

queue_wait
summarise "${TAG}_*"

log "what the future covariates bought, per horizon:"
$PY - "$RESULTS_DIR/all_runs.csv" <<'PY' || warn "aggregate first"
import sys, pandas as pd
df = pd.read_csv(sys.argv[1])
sel = df[(df["stratum"] == "all") & (df["slice"] == "overall")]
if sel.empty or "future" not in sel.columns:
    print("no scored runs yet"); raise SystemExit
piv = sel.pivot_table(index=["backbone", "fusion", "horizon"], columns="future",
                      values=["rmse", "dts_A"], aggfunc="mean").round(2)
print(piv.to_string())
print()
if ("rmse", "none") in piv.columns:
    for col in [c for c in piv.columns if c[0] == "rmse" and c[1] != "none"]:
        delta = (piv[col] - piv[("rmse", "none")]).round(2)
        print(f"RMSE change vs causal, {col[1]}:")
        print(delta.to_string(), "\n")
PY

log "Read this as a leakage measurement. Also compare the AID-heavy sources"
log "(Loop, ReplaceBG, IOBP2) against the rest in metrics.json -> by_source."
