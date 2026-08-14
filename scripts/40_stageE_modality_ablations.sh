#!/usr/bin/env bash
# Stage E / experiment E6 -- modality ablations.
#
#   scripts/40_stageE_modality_ablations.sh
#
#   cgm                       the floor: how much is autocorrelation alone?
#   cgm+insulin
#   cgm+carbs
#   cgm+insulin+carbs         the core benchmark set
#   +activity                 heart rate and steps, where recorded
#
# The interesting reading is not the aggregate. Scoring already splits every run
# by event context, so after this stage compare the *postprandial* and
# *post-correction* strata across these five settings: if carbohydrate input
# helps anywhere, it helps in the two hours after a logged meal, and if it does
# not help there, the aggregate improvement was something else.
#
# Activity is reported separately and never folded into the core benchmark: it
# exists in a minority of cohorts, so its runs are scored on a different
# population.

source "$(dirname "$0")/lib/common.sh"
require_store

BACKBONE="${BACKBONE:-mamba2}"
FUSION="${FUSION:-film}"
SEEDS="${SEEDS:-0 1 2}"
CONTEXT="${CONTEXT:-288}"

queue_init
for seed in $SEEDS; do
  base=("seed=${seed}" "data.context=${CONTEXT}" "model.fusion.name=${FUSION}")

  train "e6_${BACKBONE}_cgm_s${seed}"          "model/${BACKBONE}" "${base[@]}" "data.modalities=['cgm']"
  train "e6_${BACKBONE}_cgm-insulin_s${seed}"  "model/${BACKBONE}" "${base[@]}" "data.modalities=['cgm','insulin']"
  train "e6_${BACKBONE}_cgm-carbs_s${seed}"    "model/${BACKBONE}" "${base[@]}" "data.modalities=['cgm','carbs']"
  train "e6_${BACKBONE}_core_s${seed}"         "model/${BACKBONE}" "${base[@]}" "data.modalities=['cgm','insulin','carbs']"
  train "e6_${BACKBONE}_activity_s${seed}"     "model/${BACKBONE}" "${base[@]}" "data.modalities=['cgm','insulin','carbs','activity']"
done
queue_wait

summarise "e6_*"

log "event-context view of the ablation (postprandial and post-bolus strata):"
$PY - "$RESULTS_DIR/all_runs.csv" <<'PY' || warn "aggregate first"
import sys, pandas as pd
df = pd.read_csv(sys.argv[1])
sel = df[(df["stratum"] == "all")
         & (df["slice"].isin(["overall", "event:postprandial",
                              "event:post_correction_bolus", "event:quiet_no_logged_event"]))]
if sel.empty:
    print("no stage E runs scored yet"); raise SystemExit
piv = sel.pivot_table(index=["modalities", "horizon"], columns="slice",
                      values="rmse", aggfunc="mean").round(2)
print(piv.to_string())
PY
