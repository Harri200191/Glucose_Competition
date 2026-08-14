#!/usr/bin/env bash
# The whole programme, in dependency order.
#
#   scripts/99_run_everything.sh              # everything, days of compute
#   DRY_RUN=1 scripts/99_run_everything.sh    # print the plan, run nothing
#   SEEDS="0" scripts/99_run_everything.sh    # one seed: roughly a third of the cost
#   STAGES="B C D" scripts/99_run_everything.sh
#
# Order matters and follows the plan's milestones: nothing downstream is
# meaningful until the tests pass and the baselines are on the board.
#
# Rough cost on 8x V100 with the defaults (3 seeds, 2M training windows):
#   prepare    ~1 h, once
#   tests      ~4 min
#   stage B    ~4 h   (GBDT dominates)
#   stage C    ~12 h  (15 runs)
#   stage D    ~40 h  (63 runs)   <- the expensive one; BACKBONES=mamba2 cuts it to ~14 h
#   stage E    ~15 h  (15 runs)
#   stage F    ~20 h  (30 runs, short contexts are cheap)
#   stage G    ~8 h   (10 runs)
# Reduce with SEEDS, EXTRA_SET="data.sampling.max_train_windows=500000", or by
# running the stage scripts individually.

source "$(dirname "$0")/lib/common.sh"

STAGES="${STAGES:-prep test B C D E F G}"
log "stages: $STAGES"

for stage in $STAGES; do
  case "$stage" in
    prep)
      [[ -f "$STORE/manifest.json" ]] \
        && log "store already built at $STORE, skipping prepare" \
        || bash "$REPO_ROOT/scripts/01_prepare_data.sh"
      ;;
    test) bash "$REPO_ROOT/scripts/03_run_tests.sh" ;;
    B)    bash "$REPO_ROOT/scripts/10_stageB_baselines.sh" ;;
    C)    bash "$REPO_ROOT/scripts/20_stageC_backbones.sh" ;;
    D)    bash "$REPO_ROOT/scripts/30_stageD_fusion_ablations.sh" ;;
    E)    bash "$REPO_ROOT/scripts/40_stageE_modality_ablations.sh" ;;
    F)    bash "$REPO_ROOT/scripts/50_stageF_context_length.sh" ;;
    G)    bash "$REPO_ROOT/scripts/60_stageG_generalization.sh" ;;
    *)    warn "unknown stage $stage" ;;
  esac
done

bash "$REPO_ROOT/scripts/70_evaluate_all.sh" --bootstrap "${BOOTSTRAP:-1000}"
bash "$REPO_ROOT/scripts/80_make_figures.sh"

log "everything finished."
log "  results: $RESULTS_DIR/REPORT.md"
log "  figures: $REPO_ROOT/figures"
log "  runs:    $RUNS_DIR"
