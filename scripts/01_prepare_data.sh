#!/usr/bin/env bash
# Stage A: build the sequence store that every experiment reads.
#
#   scripts/01_prepare_data.sh              # full corpus  -> artifacts/store/main
#   scripts/01_prepare_data.sh --smoke      # 3 row groups -> artifacts/store/smoke
#
# Runs once. It is the "freeze the dataset version" step: the store's
# manifest.json records the config, the counts, the level tables and the code's
# schema version, and every run copies that record into its own manifest.
#
# Cost on the full corpus: roughly 30-60 minutes and ~15 GB of disk, dominated by
# the parquet scan and the per-segment IIR filters. Peak RAM is around 40 GB.

#
# Planned future actions (off by default):
#   FUTURE=insulin|carbs|both  expose post-anchor insulin/carbs to the model.
#   Runs get a _futins/_futcarb/_futboth name suffix and are a DIFFERENT task
#   from the causal benchmark -- see scripts/lib/common.sh.

source "$(dirname "$0")/lib/common.sh"

MODE="${1:-full}"
CORPUS="${CORPUS:-$REPO_ROOT/data/metabonet_public.parquet}"
[[ -f "$CORPUS" ]] || die "corpus not found: $CORPUS"

case "$MODE" in
  --smoke|smoke)
    OUT="${SMOKE_STORE:-$REPO_ROOT/artifacts/store/smoke}"
    log "smoke build -> $OUT (3 row groups, ~570k rows)"
    $PY -m metabonet_forecasting.data.prepare \
      --corpus "$CORPUS" --out "$OUT" \
      --row-groups 3 \
      --min-history-steps 288 \
      --anchor-stride 6 \
      2>&1 | tee "$LOG_DIR/prepare_smoke.log"
    ;;
  full|*)
    OUT="$STORE"
    log "full build -> $OUT"
    log "this reads 154.8M rows x 17 columns; expect 30-60 minutes"
    $PY -m metabonet_forecasting.data.prepare \
      --corpus "$CORPUS" --out "$OUT" \
      --min-history-steps 288 \
      --anchor-stride 6 \
      --max-cgm-gap-min 30 \
      --interp-gap-min 30 \
      --min-segment-min 360 \
      2>&1 | tee "$LOG_DIR/prepare_main.log"
    ;;
esac

log "store summary:"
$PY - "$OUT" <<'PY'
import json, sys
m = json.load(open(f"{sys.argv[1]}/manifest.json"))
s = m["stats"]
print(json.dumps({
    "rows": m["n_rows"], "anchors": m["n_anchors"], "subjects": m["n_subjects"],
    "segments": s.get("n_segments"), "channels": len(m["channels"]),
    "sources": m["levels"].get("source_file"),
    "anchor_funnel": s.get("anchor_filter_counts"),
    "targets_per_horizon": s.get("targets_valid_per_horizon"),
    "build_seconds": m["build_seconds"],
}, indent=2))
PY
