#!/usr/bin/env bash
# Turn a finished run into competition submission parquets.
#
#   scripts/90_make_submission.sh RUN_NAME                # both competitions
#   COMPETITION=annual scripts/90_make_submission.sh RUN_NAME
#   COMPETITION=live   scripts/90_make_submission.sh RUN_NAME
#
# Two targets, and they are not the same task:
#
#   annual  2,228 rows against a secret holdout (data/annual_competition_secret_
#           holdout_set_public.parquet). CGM at +30/+60/+90/+120 is masked to -1
#           for every one of them, so the labels really are held back. Scored
#           server-side; the toolkit can only check the format.
#   live    2,648,987 rows drawn from the public corpus. Its targets ship with
#           the toolkit, so this one can be scored locally -- treat it as a
#           dress rehearsal, not as a held-out result.
#
# The submission store is built with `prepare.py --anchors-from <template>`, so
# the windows come out of the same cleaning, segmentation and exposure code that
# built the training data. Do not replace this with a bespoke feature builder:
# an inference path that differs slightly from the training path is the standard
# way to ship a model that looked good locally and scores badly on submission.
#
# Note on future-covariate runs (_futins/_futcarb/_futboth). The organisers
# masked future CGM and shipped future insulin, so those runs are legal here --
# but read scripts/35_future_covariates.sh before treating the score as a
# forecasting result. Also note the annual holdout has NO carbohydrate data at
# all (100% NaN), so the carbohydrate half of a _futboth model contributes
# nothing there.

source "$(dirname "$0")/lib/common.sh"

RUN_NAME="${1:-}"
[[ -n "$RUN_NAME" ]] || die "usage: scripts/90_make_submission.sh RUN_NAME"
RUN_DIR="$RUNS_DIR/$RUN_NAME"
[[ -f "$RUN_DIR/best.pt" ]] || die "no checkpoint at $RUN_DIR/best.pt"

COMPETITION="${COMPETITION:-both}"
SUB_DIR="${SUB_DIR:-$REPO_ROOT/artifacts/submission}"
CORPUS="${CORPUS:-$REPO_ROOT/data/metabonet_public.parquet}"
HOLDOUT="${HOLDOUT:-$REPO_ROOT/data/annual_competition_secret_holdout_set_public.parquet}"
mkdir -p "$SUB_DIR"

make_one() {
  local name="$1" source_file="$2" template="$3" store="$4"
  [[ -f "$template" ]] || die "template not found: $template"
  [[ -f "$source_file" ]] || die "source data not found: $source_file"

  if [[ ! -f "$store/manifest.json" ]]; then
    log "building $name submission store (anchors = template rows)"
    $PY -m metabonet_forecasting.data.prepare \
      --corpus "$source_file" --out "$store" \
      --anchors-from "$template" \
      --min-segment-min 30 2>&1 | tee "$LOG_DIR/prepare_submission_${name}.log"
  else
    log "reusing $name submission store at $store"
  fi

  local out="$SUB_DIR/${name}_${RUN_NAME}.parquet"
  log "predicting $name -> $out"
  $PY -m metabonet_forecasting.submission.predict \
    --run "$RUN_DIR" --store "$store" --template "$template" --out "$out" \
    2>&1 | tee "$LOG_DIR/submission_${name}_${RUN_NAME}.log"

  if [[ -d "$REPO_ROOT/toolkit" ]]; then
    log "validating with the organisers' toolkit"
    ( cd "$REPO_ROOT/toolkit" && $PY run.py "$out" --competition "$name" \
        ${SCORE_HORIZON:+--horizon "$SCORE_HORIZON"} ) || warn "toolkit reported a problem"
  fi
  echo "$out"
}

case "$COMPETITION" in
  annual|both)
    make_one annual "$HOLDOUT" "$SUB_DIR/annual_template.parquet" \
      "$REPO_ROOT/artifacts/store/submission_annual" ;;&
  live|both)
    SCORE_HORIZON="${SCORE_HORIZON:-all}" \
    make_one live "$CORPUS" "$SUB_DIR/live_template.parquet" \
      "$REPO_ROOT/artifacts/store/submission_live" ;;
  *) die "COMPETITION must be annual, live or both" ;;
esac

log "submissions in $SUB_DIR:"
ls -1 "$SUB_DIR"/*_"$RUN_NAME".parquet 2>/dev/null || true
