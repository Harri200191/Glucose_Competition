#!/usr/bin/env bash
# Re-score every run and rebuild the results tables.
#
#   scripts/70_evaluate_all.sh                  # rescore + aggregate
#   scripts/70_evaluate_all.sh --bootstrap 1000 # + patient-level 95% CIs
#
# Scoring reads predictions_test.parquet, so this is safe to re-run after fixing
# a metric, adding a stratum, or changing how a table is laid out. No model is
# reloaded and no GPU is touched.
#
# The bootstrap is the expensive part (patient resampling over every stratum and
# horizon); 1000 replicates over ~50 runs takes tens of minutes on CPU.

#
# Planned future actions (off by default):
#   FUTURE=insulin|carbs|both  expose post-anchor insulin/carbs to the model.
#   Runs get a _futins/_futcarb/_futboth name suffix and are a DIFFERENT task
#   from the causal benchmark -- see scripts/lib/common.sh.

source "$(dirname "$0")/lib/common.sh"

BOOT=0
if [[ "${1:-}" == "--bootstrap" ]]; then BOOT="${2:-1000}"; fi

runs=("$RUNS_DIR"/*/)
(( ${#runs[@]} > 0 )) || die "no runs under $RUNS_DIR"

log "scoring ${#runs[@]} run directories (bootstrap=$BOOT)"
$PY -m metabonet_forecasting.evaluation.evaluate "${runs[@]}" --bootstrap "$BOOT" \
  2>&1 | tee "$LOG_DIR/evaluate_all.log"

summarise

log "primary table (test, all patients):"
$PY - "$RESULTS_DIR/primary_table.csv" <<'PY'
import sys, pandas as pd
df = pd.read_csv(sys.argv[1])
cols = [c for c in ("run", "backbone", "fusion", "context", "modalities", "seed",
                    "rmse_30", "rmse_60", "rmse_90", "rmse_120",
                    "mard_60", "dts_A_60")
        if c in df.columns]
print(df[cols].to_string(index=False))
PY
log "written: $RESULTS_DIR/{all_runs.csv,primary_table.csv,seed_summary.csv,REPORT.md}"
