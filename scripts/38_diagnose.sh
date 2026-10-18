#!/usr/bin/env bash
# Error diagnostics for one run: where it is wrong, and under what conditions.
#
#   scripts/38_diagnose.sh runs/my_run
#   scripts/38_diagnose.sh runs/my_run internal_holdout
#
# Writes to <run>/figures/<split>/:
#   error_grid.png        all horizons on the DTS grid, zones A-E drawn
#   zones_by_glucose.png  zone mix across the reference range -- read this first
#   error_profile.png     bias / MAE / zone-A by glucose, trend, bolus, hour
#   traces.png            real history + what happened + what the model said
#   zone_table.csv        the per-horizon zone percentages
#
# Figures need a matplotlib that works under numpy 2, which the repo interpreter
# does not have. This reuses the venv 37_refresh_reports.sh builds.

source "$(dirname "$0")/lib/common.sh"

RUN="${1:?usage: scripts/38_diagnose.sh <run-dir> [split]}"
SPLIT="${2:-test}"
[[ -d "$RUN" ]] || die "no such run directory: $RUN"
[[ -f "$RUN/predictions_${SPLIT}.parquet" ]] || \
  die "$RUN has no predictions_${SPLIT}.parquet"

VENV="${PLOT_VENV:-$HOME/.cache/metabonet-plots}"
if [[ ! -x "$VENV/bin/python" ]]; then
  log "building the plotting venv at $VENV (one time)"
  python -m venv --system-site-packages "$VENV" >/dev/null 2>&1 || \
    die "could not create $VENV"
  "$VENV/bin/pip" install -q "matplotlib>=3.9" || die "matplotlib install failed"
fi

log "diagnosing $RUN [$SPLIT]"
PYTHONPATH="$REPO_ROOT" "$VENV/bin/python" -m metabonet_forecasting.evaluation.diagnose \
  "$RUN" --split "$SPLIT" "${@:3}"
