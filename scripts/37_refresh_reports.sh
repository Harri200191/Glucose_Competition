#!/usr/bin/env bash
# Re-score every finished run and rebuild all figures and write-ups.
#
#   scripts/37_refresh_reports.sh
#
# Safe to run at any time, including while training is still going: it touches no
# GPU and only reads finished runs. Anything still training simply shows as
# PENDING and appears next time.
#
# Figures need a matplotlib that works under numpy 2, which the repo interpreter
# does not have. This builds one once in ~/.cache and reuses it.

source "$(dirname "$0")/lib/common.sh"

VENV="${PLOT_VENV:-$HOME/.cache/metabonet-plots}"
if [[ ! -x "$VENV/bin/python" ]]; then
  log "building the plotting venv at $VENV (one time)"
  python -m venv --system-site-packages "$VENV" >/dev/null 2>&1 || \
    die "could not create $VENV"
  "$VENV/bin/pip" install -q "matplotlib>=3.9" markdown xhtml2pdf pypdf \
    || warn "some plotting packages failed to install"
fi
PLOT_PY="$VENV/bin/python"

log "1/3  scoring every run in $RUNS_DIR"
# Pattern "*" on purpose: a narrow pattern REPLACES the global table with only the
# matching runs, which silently drops the rest of the study.
$PY -m metabonet_forecasting.evaluation.aggregate "$RUNS_DIR" \
  --out "$RESULTS_DIR" --pattern "*" >/dev/null 2>&1 \
  || warn "aggregation reported a problem"

log "2/3  per-experiment write-ups and figures"
"$PLOT_PY" scripts/analysis/followup_report.py 2>&1 | grep -E "wrote|!" || true

log "3/3  the combined results figures"
"$PLOT_PY" scripts/analysis/results_figures.py 2>&1 | grep -E "wrote|!" || true

n=$(ls -d "$RUNS_DIR"/f*/ 2>/dev/null | while read -r d; do
      [[ -f "$d/metrics.json" ]] && echo x; done | wc -l)
log "done — $n suite runs scored"
log "  docs/RESULTS.md              headline results and figures"
log "  docs/experiments/*.md        one write-up per experiment"
log "  artifacts/figures/results/   combined figures"
log "  artifacts/figures/followup/  per-experiment figures"
