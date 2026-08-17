#!/usr/bin/env bash
# The stage-A test suite: leakage, horizon alignment, masks, patient separation,
# the SSD kernel, the DTS grid against the official scorer, and one end-to-end run.
#
#   scripts/03_run_tests.sh            # everything (~2-4 min, CPU only)
#   scripts/03_run_tests.sh leakage    # one file
#
# These build their own synthetic corpus, so they need neither the real data nor
# a GPU. Run them before every sweep and after every change to data/ or models/.

#
# Planned future actions (off by default):
#   FUTURE=insulin|carbs|both  expose post-anchor insulin/carbs to the model.
#   Runs get a _futins/_futcarb/_futboth name suffix and are a DIFFERENT task
#   from the causal benchmark -- see scripts/lib/common.sh.

source "$(dirname "$0")/lib/common.sh"

TARGET="${1:-}"
if [[ -n "$TARGET" ]]; then
  FILES=("$REPO_ROOT/tests/test_${TARGET}.py")
else
  FILES=("$REPO_ROOT/tests")
fi

log "pytest ${FILES[*]}"
$PY -m pytest "${FILES[@]}" -q --durations=10 "${@:2}"
