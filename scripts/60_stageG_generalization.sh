#!/usr/bin/env bash
# Stage G / experiment E7 -- generalisation and transportability.
#
#   scripts/60_stageG_generalization.sh
#   SOURCES="OhioT1DM Loop" scripts/60_stageG_generalization.sh
#
# Three things, in increasing order of difficulty:
#
# 1. novel-patient vs known-patient. Free: the official split already contains
#    both, and every run is scored on the two strata separately. This step only
#    prints the comparison.
# 2. leave-one-source-dataset-out. Train without a study, test on it. This is the
#    transportability question -- can the model be moved to a cohort it has never
#    seen, with different devices, protocols and missingness patterns?
# 3. the cohort-shortcut sensitivity analysis: the same model with source-dataset
#    identity added as a static covariate. If accuracy jumps, part of what the
#    primary model looks like it knows about physiology is really knowledge of
#    which study a patient came from.
#
# Subgroup analyses (age, device, source) belong *after* the primary comparison
# is frozen; the scored metrics already carry per-source breakdowns for that.

#
# Planned future actions (off by default):
#   FUTURE=insulin|carbs|both  expose post-anchor insulin/carbs to the model.
#   Runs get a _futins/_futcarb/_futboth name suffix and are a DIFFERENT task
#   from the causal benchmark -- see scripts/lib/common.sh.

source "$(dirname "$0")/lib/common.sh"
require_store

BACKBONE="${BACKBONE:-mamba2}"
SEEDS="${SEEDS:-0}"
CONTEXT="${CONTEXT:-288}"
SOURCES="${SOURCES:-$($PY -c "
import json;print(' '.join(json.load(open('$STORE/manifest.json'))['levels']['source_file'][:4]))")}"

log "leave-one-source-out over: $SOURCES"

queue_init
for seed in $SEEDS; do
  for source in $SOURCES; do
    train "e7_${BACKBONE}_loso_${source}_s${seed}" "model/${BACKBONE}" \
      "seed=${seed}" "data.context=${CONTEXT}" model.fusion.name=film \
      data.split.scheme=loso "data.split.holdout_source=${source}"
  done
  train "e7_${BACKBONE}_sourceid_s${seed}" "model/${BACKBONE}" \
    "seed=${seed}" "data.context=${CONTEXT}" model.fusion.name=film \
    data.use_source_identity=true
done
queue_wait

summarise "e7_*"

log "novel-patient vs known-patient, across every scored run:"
$PY - "$RUNS_DIR" <<'PY' || warn "no scored runs yet"
import json, sys
from pathlib import Path
rows = []
for m in sorted(Path(sys.argv[1]).glob("*/metrics.json")):
    blob = json.loads(m.read_text())
    for h in blob["horizons_min"]:
        row = {"run": blob["run"], "horizon": h}
        for stratum in ("all", "novel_patient", "known_patient"):
            block = blob["strata"].get(stratum, {})
            overall = block.get(str(h), {}).get("overall", {}) if "n" not in block else {}
            row[stratum] = round(overall.get("rmse", float("nan")), 2)
        rows.append(row)
if not rows:
    print("nothing scored yet"); raise SystemExit
import pandas as pd
df = pd.DataFrame(rows)
df["gap_novel_minus_known"] = (df["novel_patient"] - df["known_patient"]).round(2)
print(df.to_string(index=False))
PY
