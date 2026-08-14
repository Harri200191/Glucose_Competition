#!/usr/bin/env bash
# Build every figure from results/all_runs.csv.
#
#   scripts/80_make_figures.sh
#   BEST_RUN=e4_mamba2_film_s0 scripts/80_make_figures.sh
#
#   context_curve_rmse.png     performance vs history length, per backbone (E3)
#   fusion_ladder_rmse_h60.png the four fusion rungs, per backbone (E4/E5)
#   stratum_heatmap_rmse.png   error by glycaemic range, event context and trend
#   error_grid_h*.png          forecasts against reference with the DTS borders
#
# Figures are only as honest as the runs behind them: a curve with one seed per
# point has no error bar, and the plotting code draws what it is given.

source "$(dirname "$0")/lib/common.sh"

FIG_DIR="${FIG_DIR:-$REPO_ROOT/figures}"
mkdir -p "$FIG_DIR"
[[ -f "$RESULTS_DIR/all_runs.csv" ]] || die \
  "no results table; run scripts/70_evaluate_all.sh first"

BEST_RUN="${BEST_RUN:-$($PY - "$RESULTS_DIR/primary_table.csv" <<'PY'
import sys, pandas as pd
df = pd.read_csv(sys.argv[1])
col = "rmse_60" if "rmse_60" in df.columns else df.filter(like="rmse_").columns[0]
print(df.sort_values(col).iloc[0]["run"])
PY
)}"
log "best run by 60-minute RMSE: $BEST_RUN"

for horizon in 30 60 120; do
  $PY -m metabonet_forecasting.evaluation.plots "$RESULTS_DIR/all_runs.csv" \
    --out "$FIG_DIR" --metric rmse --horizon "$horizon" \
    --predictions "$RUNS_DIR/$BEST_RUN/predictions_test.parquet" \
    || warn "figures skipped (see the message above; metrics are unaffected)"
done

$PY -m metabonet_forecasting.evaluation.plots "$RESULTS_DIR/all_runs.csv" \
  --out "$FIG_DIR" --metric mard --horizon 60 || true

log "figures in $FIG_DIR:"
ls -1 "$FIG_DIR"
