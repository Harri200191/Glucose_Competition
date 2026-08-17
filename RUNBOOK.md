# MetaboNet Multimodal Glucose Forecasting — Runbook

Everything in this repository, and every command needed to run it.

This is a rebuild of the project as the *benchmark and architecture study* described in
`MetaboNet_Multimodal_Glucose_Forecasting_Project_Plan.pdf`, not as a competition
submission pipeline. The question it is built to answer:

> How should CGM, insulin, carbohydrate, activity and static information be **encoded and
> fused** so that sequence models learn transferable glucose dynamics rather than
> short-term autocorrelation or dataset-specific shortcuts?

The old competition code (`exp/`, `toolkit/`, `notebooks/`, `docs/`) is gone from this
branch and lives on `master`. Nothing on disk was deleted: `data/`, `build/`, `runs/` and
`logs/` are untracked, and the new pipeline writes to `artifacts/` so it cannot collide
with the old build.

---

## 1. Quick start

```bash
cd /home/hrehman5/glucose
git checkout metabonet-forecasting-v2

scripts/00_setup_env.sh            # what is installed, what is missing, GPUs, corpus
scripts/01_prepare_data.sh --smoke # 3 row groups -> artifacts/store/smoke   (~10 s)
scripts/03_run_tests.sh            # 65 tests, CPU only                      (~6 s)
scripts/02_smoke_test.sh           # all 6 architectures + baselines, tiny   (~1 min)

scripts/01_prepare_data.sh         # the full corpus -> artifacts/store/main
scripts/10_stageB_baselines.sh     # then the stages below, in order
```

If `scripts/02_smoke_test.sh` prints a results table, the whole system works end to end:
data → windows → model → training → predictions → metrics → aggregation.

---

## 2. Layout

```
metabonet_forecasting/
  common/        config (YAML + includes + dotted overrides), registries, seeding, run logging
  data/
    constants.py   corpus facts, channel contract, modality groups, excluded columns + reasons
    prepare.py     corpus parquet -> sequence store (the only heavy, once-per-project step)
    store.py       memmap reader for a built store
    splits.py      official / subject k-fold / leave-one-source-out + novel vs known patient
    transforms.py  normalisation fitted on training subjects only; NaN policies
    windows.py     THE common window dataset every model consumes
    dataset.py     config -> store + split + standardizer + dataloaders
  models/
    base.py        the common API: batch dict -> ForecastOutput(pred_norm [B, H])
    encoders.py    CGM (causal conv), insulin/carb/activity (MLP + causal conv), static, time
    fusion.py      concat | gated | film   (+ raw_concat handled in frontend.py)
    frontend.py    encoders -> fusion -> backbone -> head, shared by TCN/PatchTST/Mamba-2/LSTM
    tcn.py patchtst.py mamba2.py recurrent.py    temporal backbones
    tft.py tide.py                               native-form architectures
    heads.py       one multi-horizon head for every backbone
  training/        losses (masked), callbacks (early stop, checkpoint, schedule), loop, train.py
  evaluation/      metrics + DTS/Clarke/Parkes grids, clinical strata, patient bootstrap,
                   plots, evaluate.py (score a run), aggregate.py (tables across runs)
  baselines/       persistence, linear trend, ridge, GBDT — on identical windows
configs/           base.yaml + model/ fusion/ data/ experiment/
scripts/           every command below, as a runnable file
tests/             leakage, windows, splits, metrics, models, end-to-end
artifacts/store/   built datasets (gitignored)
runs/ results/ figures/ logs/   outputs (gitignored)
```

---

## 3. Stage A — build and freeze the dataset

```bash
scripts/01_prepare_data.sh --smoke     # small store for development
scripts/01_prepare_data.sh             # artifacts/store/main, the real one
```

Direct invocation, if you want different knobs:

```bash
python -m metabonet_forecasting.data.prepare \
  --corpus data/metabonet_public.parquet \
  --out artifacts/store/main \
  --min-history-steps 288 \      # every anchor is guaranteed 24 h of history
  --anchor-stride 6 \            # one anchor per 30 min
  --max-cgm-gap-min 30 \         # no CGM gap longer than this inside the window
  --interp-gap-min 30 \          # gaps up to this are interpolated FOR FEATURES ONLY
  --min-segment-min 360
# scale control: --row-groups N, --max-subjects N, --cohorts StudyA,StudyB
```

**What the store is.** One `float32` memmap of the whole 5-minute lattice
(`channels.npy`, `[n_rows, 23]`) plus index tables. Windows are gathered lazily by row
index, because materialising a 24 h window per anchor would cost terabytes.

**Decisions baked into it**, each recorded in `manifest.json`:

| Step | Rule | Why |
|---|---|---|
| Columns | 17 of 37 read; the other 20 excluded with a written reason each | `insulin` is exactly `basal + bolus`; five columns are >99 % null and their *presence* identifies the study; free text and trial bookkeeping are study labels in disguise |
| Order | sort by (subject, time) before anything | two ShanghaiT1DM subjects are two wear periods concatenated, one stored backwards in time |
| Segments | cut wherever consecutive rows are not exactly 5 min apart; drop segments under 6 h | a subject is not one time series |
| Cleaning | CGM outside [10, 600] → NaN; negative doses → 0; the 40/400 clamps kept | those clamps are real reported readings |
| Interpolation | CGM gaps ≤ 30 min filled linearly, inside a segment, **features only** | recovers most events while fabricating little; forward-fill would forge flat plateaus |
| Targets | read from the raw column, never the filled one; must be inside the segment; must be on the same side of the official train/test boundary as the anchor | an interpolated label is a fabricated label, and 166–169 subjects are split *temporally* |
| Exposures | causal IIR per segment: bolus τ ∈ {30, 60, 120, 240} min, basal τ ∈ {60, 240}, carbs τ ∈ {20, 45, 90}, plus time-since-event | inductive bias, **not** a validated pharmacokinetic model — that is why every one of them is ablatable |
| Anchors | ≥ 24 h history, real CGM at the anchor, no gap > 30 min in the window, stride 6 | r(CGM_t, CGM_t+5min) ≈ 0.99, so consecutive anchors are near-duplicates |

Nothing is normalised at this stage. Normalisation is fitted per run, on training
subjects only, and saved to `standardizer.json`.

---

## 4. Stage A — the test suite

```bash
scripts/03_run_tests.sh              # all 65 tests, ~6 s, CPU, no real data needed
scripts/03_run_tests.sh leakage      # one file
```

The tests build their own synthetic corpus with planted structure (a short gap that must
be interpolated, a long one that must not, a subject stored out of order, a subject split
across the train/test boundary) and run the real `prepare.build_store` over it.

What they pin down:

- **Causality**: perturb inputs after step *t*, assert outputs at *t* do not move — for the
  causal convolution, the CGM and driver encoders, TCN, LSTM, Mamba-2, and across
  PatchTST's patches.
- **Horizon alignment**: `targets[i, j]` equals the raw CGM exactly *h* minutes later, and
  is always a real reading.
- **Statistical leakage**: two disjoint anchor sets give different normalisation constants
  (so the fit really is per-split).
- **Patient separation**: train/val subject-disjoint, val never touches test subjects,
  leave-one-source-out really holds out the study, and the guard *raises*.
- **The DTS error grid** matches the organisers' vendored scorer point for point.
- **The SSD scan** behind Mamba-2 matches an explicit step-by-step recurrence to 1e-8.
- **End to end**: a real training run leaves behind a complete, traceable artifact.

---

## 5. Stages B–G — the experiments

Every stage is one script. All of them accept `SEEDS`, `CONTEXT`, `GPUS`, `JOBS_PER_GPU`,
`GPU_FREE_MB`, `EXTRA_SET`, `DRY_RUN=1`, and skip runs that are already scored
(`SKIP_EXISTING=0` to force). Jobs wait for a GPU that genuinely has room rather than
piling onto a busy one — see §9.

```bash
# B / E1 — the floor: persistence, linear trend, ridge, GBDT, simple LSTM
scripts/10_stageB_baselines.sh

# C / E2 — controlled backbone comparison, raw concatenation held fixed
scripts/20_stageC_backbones.sh
SEEDS="0" scripts/20_stageC_backbones.sh

# D / E4+E5 — the fusion ladder (the core contribution) on each backbone
scripts/30_stageD_fusion_ablations.sh
BACKBONES="mamba2" scripts/30_stageD_fusion_ablations.sh     # much cheaper

# E / E6 — modality ablations, read in event context
scripts/40_stageE_modality_ablations.sh

# F / E3 — context length: 2/4/8/12/24 h per backbone
scripts/50_stageF_context_length.sh
CONTEXTS="24 96 288" SEEDS="0" scripts/50_stageF_context_length.sh

# G / E7 — novel vs known patient, leave-one-source-out, cohort-shortcut sensitivity
scripts/60_stageG_generalization.sh
SOURCES="OhioT1DM Loop" scripts/60_stageG_generalization.sh

# scoring and figures, re-runnable at any time
scripts/70_evaluate_all.sh --bootstrap 1000
scripts/80_make_figures.sh

# or the whole programme in dependency order
DRY_RUN=1 scripts/99_run_everything.sh     # print the plan first
scripts/99_run_everything.sh
STAGES="B C D" SEEDS="0" scripts/99_run_everything.sh
```

The fusion ladder, which is the experiment the project exists for:

| Rung | `model.fusion.name` | Adds |
|---|---|---|
| 1 | `raw_concat` | nothing — normalised channels → one linear → backbone |
| 2 | `concat` | modality-specific shallow encoders |
| 3 | `gated` | learned per-modality gates |
| 4 | `film` | CGM-centred conditioning: `Ē_G = γ(c) ⊙ E_G + β(c)` |
| 5–7 | `film` minus event features / minus masks / plus learned missing embedding | which component carries the result |

Rung 4 − rung 1 is the headline claim. Rung 4 − rungs 2 and 3 is what separates a claim
about *conditioning* from a claim about *encoding*. FiLM is initialised as the identity
(`γ = 1 + Δ`, `Δ = 0`), so any modulation is a departure the optimiser had to pay for.

---

## 5b. Planned future actions (off by default)

Every script accepts `FUTURE`, which lets the model read insulin and/or
carbohydrate values from **after** the forecast anchor:

```bash
FUTURE=insulin scripts/20_stageC_backbones.sh    # future insulin schedule visible
FUTURE=carbs   scripts/30_stageD_fusion_ablations.sh
FUTURE=both    scripts/40_stageE_modality_ablations.sh
FUTURE_STEPS=6 FUTURE=both scripts/...           # only the next 30 min, instead of 120

scripts/35_future_covariates.sh                  # the dedicated 4-arm comparison
RUN_BASELINES=1 scripts/35_future_covariates.sh  # + GBDT, the leaderboard recipe
```

`FUTURE=none` is the default everywhere and the benchmark; runs that opt in get a
`_futins` / `_futcarb` / `_futboth` name suffix so they can never be confused with
their causal twin, and `future` becomes a column in `results/all_runs.csv` so the
comparison is a group-by.

**This changes the task, it does not improve the model.** A forecast answers
"given what is known now, what will glucose be?"; reading post-anchor doses
answers "given that this dose is delivered, what will glucose be?". Both are
legitimate, neither is comparable to the other, and only the first is a forecast.

In this corpus the distinction is not academic: **85.8 % of rows are AID**
(closed-loop; Loop alone is 61.2 %), where delivery at t+k is computed by a
control algorithm from the CGM at t+k. Future insulin there is not a plan a
patient made — it is the label, transformed. Measured on the smoke slice, giving
a GBDT nothing but future dose and carbohydrate sums moves DTS zone A from
84.3 → 86.3 at 30 min and 47.7 → 57.0 at 120 min. A gain that *grows* with
horizon is the signature: a genuine planned-action signal would not behave that
way, but a controller's reaction to the very glucose you are predicting does.

Only the raw per-slot values and their masks are exposed, never the exponential
exposures (those accumulate the past and have no forward definition), and the
block never crosses the end of the anchor's segment. Anything produced this way
belongs in its own table, clearly labelled.

## 6. Running one thing by hand

```bash
python -m metabonet_forecasting.training.train \
  --config configs/model/mamba2.yaml \
  --name mamba2_film_ctx288_s0 \
  --set data.context=288 \
        model.fusion.name=film \
        "data.modalities=['cgm','insulin','carbs']" \
        seed=0 \
        training.epochs=20

python -m metabonet_forecasting.baselines.run_baselines \
  --config configs/experiment/e1_baselines.yaml --model gbdt

python -m metabonet_forecasting.evaluation.evaluate runs/mamba2_film_ctx288_s0 --bootstrap 1000
python -m metabonet_forecasting.evaluation.aggregate runs --out results
python -m metabonet_forecasting.evaluation.aggregate runs \
  --compare mamba2_film_ctx288_s0 mamba2_raw_concat_ctx288_s0 --horizon 120
```

Config resolution: a file may `include:` others, they deep-merge in order, the including
file wins, and `--set key.path=value` wins over everything. Both the file list and the
override list are written into the run's manifest, so a run is reconstructible from its own
output directory.

Knobs worth knowing:

| Override | Effect |
|---|---|
| `data.context=24…288` | history length in 5-minute steps |
| `data.horizons_min=[30,60,120]` | which horizons to train and score |
| `data.modalities=['cgm','insulin']` | modality ablation |
| `data.use_event_features=false` | drop decay kernels and time-since-event |
| `data.use_masks=false` | drop observation masks |
| `data.missing_policy=zero\|ffill\|learned` | NaN policy |
| `data.use_source_identity=true` | cohort id as a static (sensitivity analysis only) |
| `data.split.scheme=loso data.split.holdout_source=OhioT1DM` | transportability arm |
| `data.sampling.max_train_windows=500000` | the main cost dial |
| `training.epochs`, `training.max_steps_per_epoch` | the other cost dials |

---

## 7. What a run produces

```
runs/<name>/
  config.resolved.yaml     the exact config after includes and overrides
  manifest.json            git commit + dirty flag, package versions, GPU, seed,
                           store manifest excerpt, model description, window counts
  standardizer.json        the frozen normalisation, fitted on training subjects only
  train_log.csv            per-epoch train/val loss, RMSE, MAE, MARD, lr, seconds
  events.jsonl             append-only structured log
  best.pt                  best checkpoint by validation RMSE in mg/dL
  predictions_test.parquet per-anchor forecasts + the metadata every stratum needs
  metrics.json             nested: stratum -> horizon -> slice -> metrics (+ bootstrap)
  metrics.csv              the same, flat, for aggregation
```

Predictions are stored per anchor, so **evaluation never requires retraining**: add a
stratum, fix a metric, change a table, re-run `scripts/70_evaluate_all.sh`.

Scored automatically for every run, at every horizon:

- **Strata**: `all`, `novel_patient`, `known_patient`, `leaderboard` (the four cohorts the
  public leaderboard drops: no carbohydrate data, or MDI-only).
- **Slices**: glycaemic range (hypo L2 <54, hypo L1 54–70, in range, hyper L1, hyper L2
  >250, by the *true* future value); event context (postprandial ≤120 min after a logged
  carbohydrate entry, post-correction ≤120 min after a bolus, both, neither); trend
  (falling / steady / rising by the 30-minute change at the anchor).
- **Metrics**: RMSE, MAE, MARD, signed bias, ISO 15/15 and 20/20 agreement, DTS, Clarke
  and Parkes error-grid zone percentages, and subject-level RMSE distribution.
- **Per source dataset**, for the cohort-shortcut and transportability analyses.

Statistics follow plan §8.3: patient-level bootstrap, not window-level. `--bootstrap 1000`
adds 95 % CIs; `--compare A B` resamples the *same* subjects for both models and reports
the paired difference, its interval, a bootstrap p-value, and the share of patients on
which A wins.

---

## 7b. Competition submissions

```bash
scripts/90_make_submission.sh e9_mamba2_film_s0_futboth        # both competitions
COMPETITION=annual scripts/90_make_submission.sh <RUN_NAME>    # just the annual one
```

Submissions go through `prepare.py --anchors-from <template>`, which builds a
store whose anchors are exactly the template's rows, in the template's order,
using the same cleaning, segmentation and exposure code as the training data.
That is deliberate: an inference path that computes features slightly differently
from the training path is the standard way to ship a model that looked good
locally and scored badly on submission.

`submission/predict.py` then loads the run's own frozen `standardizer.json` and
`best.pt`, refuses to write a partially filled horizon column, leaves untrained
horizons entirely NaN, and checks the row order against the template before
saving. Two traps it handles that are easy to miss:

* **categorical vocabularies.** Each store numbers `gender` / `modality` /
  `source_file` in first-appearance order *within its own file*. The holdout has
  two genders, the corpus has four — so the same integer means different things,
  and an embedding trained on one indexed with the other's codes is wrong
  silently whenever the counts happen to match. `align_categoricals` re-maps by
  name into the training store's index space; a level the corpus never saw (the
  holdout's synthetic `source_file`) becomes the reserved unknown embedding.
* **short history.** A template row can sit near the start of a recording.
  `WindowDataset(pad_history=True)` left-pads with zeros and an unobserved mask,
  so the window says "nothing here" instead of reaching into another patient.
  Training never sets that flag.

The two competitions are different tasks:

| | annual | live leaderboard |
|---|---|---|
| rows | 2,228 | 2,648,987 |
| source | separate secret holdout file | the public corpus |
| labels | **masked** — CGM at +30/60/90/120 is `-1` for 100% of anchors | ship with the toolkit |
| scoring | server-side | local |

**On future covariates and the rules.** The organisers masked future *CGM* in the
holdout and left future *insulin* fully present (100% of anchors). Using it is
therefore permitted by their own data design, which is how a 92% zone-A score at
30 minutes appears on the leaderboard — our `_futboth` run reproduces exactly
that. It is still not a forecasting number; see §5b. One quirk worth knowing:
the annual holdout contains **no carbohydrate data at all** (100% NaN), so the
carbohydrate half of a `_futboth` model contributes nothing there.

## 8. Model zoo

| Config | Tests | Notes |
|---|---|---|
| `configs/model/tcn.yaml` | are multiscale local patterns enough? | strictly causal; depth auto-sized so the receptive field covers the context, and the field is reported |
| `configs/model/patchtst.yaml` | does patch attention benefit long context? | patches the **fused latent**, per the plan, not raw channels — a documented deviation from channel-independence; patches are trailing-aligned so the last one ends at the anchor |
| `configs/model/mamba2.yaml` | does selective state help long context and events? | pure-PyTorch chunked SSD scan (tested against an explicit recurrence); runs in fp32 under AMP because a decay product over hundreds of steps is exactly what fp16 loses |
| `configs/model/tft.yaml` | does native variable selection win? | native form, no common fusion in front of it; encoder-only, because nothing about the future is known at forecast time |
| `configs/model/tide.yaml` | can a dense encoder-decoder match sequence models? | efficiency control; keeps the global linear residual from the lookback |
| `configs/model/lstm.yaml` | how much is "a sequence model was used at all"? | stage-B reference |

All of them consume the identical batch from `data/windows.py` and emit `[B, H]` in model
space. Targets are the **delta from the last CGM reading**, standardised per horizon, and
the head is zero-initialised — so every model starts at the persistence forecast and has to
earn any departure from it.

---

## 9. Environment

Verified working in the existing conda base env: Python 3.9, torch 2.6 + CUDA 12.4,
numpy 2.0.2, pandas 2.3.3, 8 × V100-32GB, 96 cores, 754 GB RAM.

```bash
scripts/00_setup_env.sh              # report
scripts/00_setup_env.sh --install    # pip install -r requirements.txt
```

Three environment facts worth knowing before they cost you an afternoon:

1. **`mamba_ssm` is not installed, and does not need to be.** Its CUDA kernels want
   Ampere-class hardware; these are Volta. `models/mamba2.py` implements the SSD
   recurrence in pure PyTorch and that is the path the tests cover. If you later run on
   A100/H100: `pip install mamba-ssm causal-conv1d` and set
   `model.backbone.use_kernels=true` for a speedup with identical mathematics.
2. **matplotlib 3.5.1 in this env is broken under numpy 2** (`numpy.core.multiarray failed
   to import`). Metrics, tables and every experiment are unaffected — only figures. Fix
   with `pip install -U 'matplotlib>=3.8'`. The plotting code imports matplotlib lazily and
   the figure script degrades to a warning rather than killing a sweep. (The figures were
   verified by rendering them under the `multi` env, which has matplotlib 3.11.)
3. **`numexpr`/`bottleneck` print an `_ARRAY_API not found` traceback at import.** Cosmetic
   noise from the same numpy-2 mismatch, emitted by pandas' optional dependencies. Nothing
   downstream uses them. `python -W ignore` hides most of it.
4. **This box is shared and heavily loaded** (48 users, load average >20 during
   development). Timings vary by orders of magnitude between runs; the same pyarrow call
   was measured at 0.04 s and at 31 s minutes apart. Prefer running long jobs detached
   (`setsid nohup ... &`) and judge progress from the phase log, not from wall clock.

GPU scheduling: `scripts/lib/common.sh` does **admission control**, not round-robin. A run
starts only on a device that is free at that moment — fewer than `JOBS_PER_GPU` of our own
jobs on it *and* at least `GPU_FREE_MB` MiB reported free by `nvidia-smi`. If nothing
qualifies the queue polls every `GPU_POLL_S` seconds until something does, whether that is
one of our jobs exiting or one of the other 47 users'. A device that has just been handed a
job is skipped for `GPU_WARMUP_S` seconds, because torch takes tens of seconds to allocate
and the card would otherwise look free twice. One log file per run under `logs/<date>/`.

| Variable | Default | Meaning |
|---|---|---|
| `GPUS` | `CUDA_VISIBLE_DEVICES`, else every device | physical nvidia-smi indices |
| `JOBS_PER_GPU` | 1 | how many of *our* runs may share a device |
| `GPU_FREE_MB` | 12000 | free memory required to admit a run (Mamba-2 @ 288 measures 10.2–10.8 GiB) |
| `GPU_POLL_S` | 30 | re-check interval while blocked |
| `GPU_WARMUP_S` | 90 | hold-off after dispatch, so allocation shows up first |
| `GPU_WAIT_MAX_S` | 0 | give up after this long; 0 = wait forever |

```bash
GPUS="4,5,6,7" JOBS_PER_GPU=2 scripts/30_stageD_fusion_ablations.sh
GPU_FREE_MB=20000 scripts/20_stageC_backbones.sh     # a heavier model
GPU_WAIT_MAX_S=7200 scripts/50_stageF_context_length.sh  # fail rather than queue overnight
```

Re-running a stage is always safe: anything already scored is skipped (`SKIP_EXISTING=0` to
force), so a sweep that gives up on a full box picks up where it left off.

---

## 10. Cost, and how to shrink it

Measured on this machine: Mamba-2 at 24 h context, batch 256, ≈1,300 windows/s on one
V100 → about 25 minutes per epoch over 2M training windows.

The store build is a one-off, and `prepare.py` logs every phase with its duration so a
long build is never silent. Measured on a 5-row-group slice (4.39M rows, ~3 % of the
corpus), before and after the optimisations that this profiling motivated:

| Phase | before | after | what changed |
|---|---|---|---|
| read | ~95 s | see below | destination arrays preallocated from the parquet footer instead of concatenating per-group blocks |
| sort | 63 s | skipped when already ordered | the reorder is skipped outright if the corpus already arrives subject-time sorted |
| segment + clean | <1 s | <1 s | |
| channels | 38 s | **2.9 s** | one contiguous memmap write per segment instead of ~13 column-strided passes over the file |
| anchors + targets | 44 s | **0.6 s** | |
| tables | >4 min | **0.3 s** | per-subject statics by boundary reduction instead of a row-wise pandas groupby |

The read phase is the one number not to trust. On this shared box (48 users, load average
>20) the *identical* pyarrow call was measured at 0.04 s and at 31 s minutes apart, and
Python's cyclic collector made it worse still — with the collector on, successive row
groups cost 0.04 s, 9 s, 35 s, 64 s as the live heap grew, so the build now runs with it
disabled (`no_cyclic_gc` in `prepare.py`, with the reasoning written out there).

Plan for the full corpus (154.8M rows, ~35× this slice) to take **under an hour of real
work but potentially several hours of wall clock** depending on what else is running, and
~15 GB on disk with tens of GB of RAM at peak. Run it detached and watch the phase log:

```bash
setsid nohup scripts/01_prepare_data.sh > logs/prepare.log 2>&1 &
tail -f logs/prepare.log
```

| Stage | Runs (3 seeds) | Rough cost |
|---|---|---|
| prepare | 1 | once, tens of minutes |
| B baselines | 5 | ~4 h (GBDT dominates) |
| C backbones | 15 | ~12 h |
| D fusion ladder | 63 | ~40 h → ~14 h with `BACKBONES=mamba2` |
| E modalities | 15 | ~15 h |
| F context length | 30 | ~20 h (short contexts are cheap) |
| G generalisation | 10 | ~8 h |

To cut it: `SEEDS="0"` (a third), `EXTRA_SET="data.sampling.max_train_windows=500000"` (a
quarter), or run one stage at a time. Seed variability is reported in
`results/seed_summary.csv`, so dropping to one seed is a stated limitation rather than a
hidden one.

---

## 11. What has been verified, and what has not

Run and confirmed working:

- **80/80 tests pass** (`scripts/03_run_tests.sh`, ~8 s, CPU only). This includes the
  14 ablation switches the stage scripts turn, each built and run end to end.
- Smoke store built from the real corpus: 2.29M rows, 214,721 anchors, 62 subjects,
  4 source datasets, 23 channels.
- The optimised builder produces **byte-identical output** to the original: same row,
  anchor and subject counts, same anchor funnel, same per-horizon target counts.
- `scripts/02_smoke_test.sh`: all six architectures plus persistence, linear trend and
  ridge trained, predicted and scored across 8 GPUs; GBDT verified separately; results
  aggregated into one table. Re-running it reproduced every metric exactly.
- A 24 h-context Mamba-2 run on GPU with AMP, end to end including scoring
  (~1,300 windows/s).
- Patient-level bootstrap CIs, per-source breakdowns, novel/known-patient strata,
  paired model comparison.
- All four figures rendered (under an env with a working matplotlib).

Sanity of the smoke numbers, for orientation only (30 optimiser steps on 3 % of the
corpus): persistence at 30/60/120 min gives RMSE 24.4 / 40.6 / 57.0 mg/dL, and the GBDT
baseline 20.7 / 33.7 / 46.0. Those are the right order of magnitude for this task, which
is the point of running them — not evidence about any architecture.

Not run, deliberately — these are the experiments themselves, and they are hours to days
of compute:

- the full-corpus store build;
- any stage B–G sweep to convergence.

No accuracy claim in this repository has been established yet. The smoke numbers exist
only to show the plumbing works, and every script that produces them says so.

---

## 12. Deviations from the plan document

Stated here because they change what the results mean:

1. **PatchTST patches the fused latent sequence**, not raw channels independently. The plan
   asks for this (§6) so the backbone comparison stays controlled; it does mean this is not
   the paper's channel-independent PatchTST. The `raw_concat` setting is the closest
   available approximation to the original input treatment.
2. **TFT and TiDE are encoder-only.** Both architectures natively consume *known future*
   covariates; in glucose forecasting nothing about the future is known at inference —
   future insulin and meals are precisely what is unknown. Their decoders are conditioned
   on a learned per-horizon embedding instead.
3. **90 minutes is carried alongside 30/60/120.** The plan names three primary horizons;
   the public benchmark also scores 90, and dropping it from the store would make results
   non-comparable. It is built, and reported as secondary.
4. **Activity/physiology is off by default.** Heart rate and steps exist in a minority of
   cohorts (>99 % null corpus-wide), so as the plan says they are an optional extension:
   `data.modalities=[...,'activity']` enables them, and those runs are scored on a
   different population, which is why they are reported separately.
5. **Source-dataset identity is excluded from the primary model** and available only via
   `data.use_source_identity=true` for the sensitivity analysis, because missingness
   patterns nearly determine the cohort and the shortcut is worth measuring rather than
   absorbing.

---

## 13. Extending it

- **A new backbone**: subclass `TemporalBackbone` ( `[B, L, d] -> [B, L', d]`, causal),
  decorate with `@register_backbone("name")`, import it in `models/__init__.py`, add
  `configs/model/name.yaml`. It inherits the front end, the head, the training loop and
  the whole evaluation stack. Add it to the causality test.
- **A new fusion**: subclass `FusionBase`, decorate `@register_fusion("name")`, add a
  `configs/fusion/name.yaml`, and it becomes another rung of the ladder.
- **A new metric or stratum**: add it in `evaluation/metrics.py` or
  `clinical_metrics.py`, then `scripts/70_evaluate_all.sh` — no retraining.

Two rules the code enforces rather than suggests: training never branches on which model
it holds, and the split guard raises instead of warning.
