# How to run the experiments

Everything is driven by three commands: **build once**, **train many**, **collect**.

```bash
cd /home/hrehman5/glucose
PY=~/anaconda3/envs/ecg_cpc/bin/python
```

All commands below assume `$PY` and that you are in the repo root.

---

## 0 · Environment

The `ecg_cpc` conda env, with these versions verified working on the 8× Tesla V100s:

| Package | Version | GPU |
|---|---|---|
| python | 3.13.5 | |
| torch | 2.7.1+cu126 | ✅ 8 devices |
| xgboost | **2.1.4** | ✅ |
| catboost | 1.2.10 | ✅ |
| lightgbm | 4.7.0 | CPU |
| tabpfn | **2.2.1** | ✅ |
| scikit-learn | 1.6.1 | CPU |

Two version pins that are **not optional**:

- **xgboost must be 2.1.4.** V100s are compute capability SM 70, and xgboost ≥ 3.1
  dropped that target — 3.3.0 fails with `This program was not compiled for SM 70`.
- **tabpfn must be 2.2.1.** The 8.x line is licence-gated: it refuses to download
  weights without an interactive licence acceptance and a `TABPFN_TOKEN`. 2.2.1
  downloads from HuggingFace directly.

If you need to recreate them:

```bash
$PY -m pip install "xgboost==2.1.4" "tabpfn==2.2.1" catboost lightgbm
```

---

## 1 · Build the dataset (once, ~20–35 min)

```bash
$PY -m exp.build_dataset --out build/main --stride 6
```

Produces in `build/main/`:

| File | Size (approx.) | What |
|---|---|---|
| `features.npy` | ~5 GB | float32 `[n_samples, 113]` |
| `targets.npy` | ~160 MB | float32 `[n_samples, 4]` for 30/60/90/120 min |
| `meta.parquet` | ~300 MB | subject, cohort, date, `is_test`, `seg_id`, `row_idx` |
| `channels.npy` | ~3 GB | full-resolution 5-channel series for `tcn` / `gru` |
| `manifest.json` | ~60 KB | config, column decisions, cleaning log, attrition counts |

Useful variants:

```bash
# quick smoke build (40 subjects, ~90 s)
$PY -m exp.build_dataset --out build/smoke --subjects 40 --stride 12

# denser sampling: 6x more training rows, 6x the build cost and disk
$PY -m exp.build_dataset --out build/dense --stride 1

# ablation: no cohort identity leaking in through source_file
$PY -m exp.build_dataset --out build/nocohort --stride 6 --no-cohort-feature

# ablation: no sensor-health / missingness features
$PY -m exp.build_dataset --out build/nomiss --stride 6 --no-missingness-features

# looser gap tolerance -> bigger, dirtier training set
$PY -m exp.build_dataset --out build/loose --stride 6 --max-gap-min 60
```

Check what you built before training on it:

```bash
$PY -c "
import json; m=json.load(open('build/main/manifest.json'))
print(m['n_samples'],'samples,',m['n_features'],'features,',m['n_subjects'],'subjects')
print('attrition:', m['stats'])
"
```

---

## 2 · Single experiment

```bash
$PY -m exp.train --data build/main --model xgboost --horizon 30 --split official --gpu 0
```

| Flag | Meaning |
|---|---|
| `--model` | one of the 16 below |
| `--horizon` | `30` \| `60` \| `90` \| `120` |
| `--split` | `official` \| `subject` \| `cohort` |
| `--fold N` | for `--split subject` (default 5 folds) |
| `--holdout-cohort NAME` | required for `--split cohort` |
| `--gpu N` | `-1` for CPU |
| `--max-train N` | cap training rows |
| `--max-eval N` | cap eval rows |
| `--set k=v ...` | override any model kwarg, JSON-parsed |

Examples:

```bash
# deeper trees, more of them
$PY -m exp.train --data build/main --model xgboost --horizon 60 \
    --set max_depth=10 n_estimators=5000 learning_rate=0.02

# leave-one-study-out: does the model know physiology or just the cohort?
$PY -m exp.train --data build/main --model xgboost --horizon 30 \
    --split cohort --holdout-cohort Loop

# a sequence model on GPU 3, longer window
$PY -m exp.train --data build/main --model tcn --horizon 30 --gpu 3 \
    --set epochs=40 seq_len=144 seq_stride=2

# TabPFN with more bagged contexts
$PY -m exp.train --data build/main --model tabpfn --horizon 30 --gpu 0 \
    --set n_subsets=8 subset_size=10000 max_eval_rows=200000
```

### The 16 models

| Name | Family | Notes |
|---|---|---|
| `persistence` | baseline | predict "glucose stays put"; the number to beat |
| `linear_trend` | baseline | extrapolate 15-min slope; EDA says it is *worse* |
| `xgboost` | GBDT | GPU, L1 on log-target |
| `xgboost_l2` | GBDT | squared error control |
| `lightgbm` | GBDT | CPU, thread-capped at 16 |
| `catboost` | GBDT | CPU for MAE loss (no GPU MAE kernel) |
| `catboost_gpu` | GBDT | GPU, RMSE loss |
| `histgb` | GBDT | sklearn, dependency-free control |
| `randomforest` | forest | needs imputation |
| `extratrees` | forest | needs imputation |
| `tabpfn` | PFN | bagged 10k-row contexts |
| `mlp` | NN | tabular |
| `resmlp` | NN | tabular, residual blocks |
| `ft_transformer` | NN | tabular, feature-tokeniser transformer |
| `tcn` | NN | **sequence**, dilated causal convs |
| `gru` | NN | **sequence**, GRU encoder |

`tcn` and `gru` require `channels.npy` — do not build with `--no-channels`.

---

## 3 · Full sweeps

### 3.1 The headline run — everything, all four horizons

```bash
$PY -m exp.run_all --data build/main --split official --gpu 0 --skip-done \
    2>&1 | tee runs/sweep_official.log
```

14 models × 4 horizons = 56 runs. `--skip-done` makes it resumable.

### 3.2 Preview the plan without running it

```bash
$PY -m exp.run_all --data build/main --split official --dry-run
```

### 3.3 Just the tree models, quickly

```bash
$PY -m exp.run_all --data build/main --split official \
    --models persistence xgboost lightgbm catboost histgb --horizons 30 60 90 120
```

### 3.4 Just the neural nets, on their own GPU

```bash
$PY -m exp.run_all --data build/main --split official --gpu 1 \
    --models mlp resmlp ft_transformer tcn gru --horizons 30 60 90 120
```

### 3.5 Subject-level cross-validation (model selection)

```bash
$PY -m exp.run_all --data build/main --split subject --folds 0 1 2 3 4 \
    --models xgboost catboost mlp tcn --horizons 30
```

### 3.6 Leave-one-cohort-out (the honest generalisation check)

```bash
$PY -m exp.run_all --data build/main --split cohort --horizons 30 \
    --models xgboost mlp \
    --cohorts Loop ReplaceBG IOBP2 PEDAP CTR3 BrisT1D AZT1D HUPA-UCM
```

Compare these MARDs against §3.5. A large gap means the model is exploiting cohort
identity rather than physiology (see `DATA_PIPELINE.md` §1.3).

### 3.7 Run several GPUs in parallel

Different model sets, different GPUs, same data — they only read `build/main`:

```bash
$PY -m exp.run_all --data build/main --gpu 0 --models xgboost xgboost_l2 catboost_gpu &
$PY -m exp.run_all --data build/main --gpu 1 --models mlp resmlp ft_transformer &
$PY -m exp.run_all --data build/main --gpu 2 --models tcn gru &
$PY -m exp.run_all --data build/main --gpu 3 --models tabpfn &
wait
$PY -c "from exp.run_all import collect; collect()"
```

---

## 4 · Collect results

Every sweep ends by calling this, but you can run it any time:

```bash
$PY -c "from exp.run_all import collect; collect()"
```

Writes `runs/results.csv` and prints a table per (split, horizon), sorted by MARD.

Columns: `MARD`, `RMSE`, `MAE`, `clarke_A`, `clarke_AB`, `parkes_A`, `parkes_AB`,
`MARD_hypo`, `MARD_subj_mean`, `MARD_subj_p90`, `MARD_leaderboard`, `fit_s`.

Per-run detail lives in `runs/<tag>/result.json`, including a per-cohort breakdown
and the model's own notes (objective, best iteration, imputation config, device).

---

## 5 · Suggested order

```bash
# 1. smoke test the whole path first -- ~10 min, catches everything structural
$PY -m exp.build_dataset --out build/smoke --subjects 40 --stride 12
$PY -m exp.run_all --data build/smoke --split subject --horizons 30 --runs-dir runs/smoke

# 2. build the real dataset
$PY -m exp.build_dataset --out build/main --stride 6

# 3. baselines first -- if these don't reproduce the EDA numbers, stop and debug
$PY -m exp.run_all --data build/main --models persistence linear_trend

# 4. the main sweep
$PY -m exp.run_all --data build/main --split official --skip-done | tee runs/sweep.log

# 5. the generalisation check
$PY -m exp.run_all --data build/main --split cohort --horizons 30 \
    --models xgboost mlp --cohorts Loop ReplaceBG IOBP2 PEDAP

# 6. read the table
$PY -c "from exp.run_all import collect; collect()"
```

**Step 3 is the important one.** `persistence` must land near the EDA's measured
MARD of **14.03 / 22.71 / 28.41 / 32.32 %** at 30/60/90/120 min, and `linear_trend`
must come out *worse* at every horizon. If either is off, the feature/target
alignment is broken and no other number means anything.

### What the smoke test actually produced

Every model below was run end-to-end on `build/smoke` (40 subjects, 172,509 samples,
107,853 training rows, 30-min horizon, subject split). These are **not** the numbers
to expect from the full corpus — 40 subjects is far too few to separate the models —
but they confirm every code path works and every learner beats the baseline.

| model | family | MARD % | RMSE | Clarke-A % | fit s |
|---|---|---|---|---|---|
| `mlp` | nn | **10.214** | 19.59 | 87.03 | 9 |
| `resmlp` | nn | 10.271 | 19.70 | 86.94 | 14 |
| `ft_transformer` | nn | 10.297 | 19.75 | 86.76 | 136 |
| `tcn` | nn | 10.312 | 19.54 | 86.75 | 41 |
| `catboost` | gbdt | 10.316 | 19.54 | 86.77 | 27 |
| `lightgbm` | gbdt | 10.318 | **19.43** | 86.72 | 12 |
| `xgboost` | gbdt | 10.318 | 19.50 | 86.73 | 8 |
| `tabpfn` | pfn | 10.318 | 19.59 | 86.80 | 16 |
| `histgb` | gbdt | 10.358 | 19.48 | 86.64 | 157 |
| `gru` | nn | 10.390 | 19.68 | 86.41 | 22 |
| `randomforest` | forest | 10.441 | 19.86 | 86.45 | 11 |
| `extratrees` | forest | 10.486 | 19.93 | 86.24 | 8 |
| `persistence` | baseline | 13.133 | 24.77 | 79.43 | 0 |
| `linear_trend` | baseline | 14.731 | 28.81 | 76.91 | 0 |

Sanity checks this confirms: every learner beats persistence by ~2.8 MARD points,
`linear_trend` is *worse* than persistence exactly as EDA 9.6 predicted, and the
neural nets are competitive with the GBDTs only because of the residual
parameterisation (see `DATA_PIPELINE.md` §11 — without it the TCN scored 14.03 %,
i.e. worse than doing nothing).

Note `tabpfn` matched the GBDTs while seeing only **37 % of the training rows**
(4 bagged contexts × 10,000 rows, 100 of 113 features). That ratio falls as the
dataset grows — on the full corpus it will see 40,000 of ~10 M rows, i.e. 0.4 % — so
treat its full-corpus number as a small-context result, not a like-for-like
comparison. `result.json → model_notes` records the exact fraction for every run.

**GPU memory:** TabPFN attends every query row against its whole context, so peak
memory is `O(chunk × context × n_estimators)`. The prediction chunk is 10,000, which
fits comfortably on a 32 GB V100; at 50,000 it peaked at 31 GB and OOM'd anything
sharing the device. If you run TabPFN alongside another job, give it its own GPU.

---

## 6 · Expected runtimes (154 M-row corpus, stride 6)

Rough, on this box (96 cores, V100):

| Stage | Time |
|---|---|
| `build_dataset --stride 6` | 20–35 min |
| `persistence` / `linear_trend` | seconds |
| `xgboost` (GPU) | 5–20 min per horizon |
| `lightgbm` / `histgb` | 15–45 min |
| `catboost` (CPU, MAE) | 30–90 min |
| `randomforest` (capped 3 M rows) | 30–60 min |
| `mlp` / `resmlp` | 10–30 min |
| `ft_transformer` | 30–60 min |
| `tcn` / `gru` | 45–120 min |
| `tabpfn` | 10–30 min |

The full 56-run official sweep is roughly a day on one GPU, or ~6 hours spread
across four as in §3.7.

---

## 7 · Before you submit anything

The error-grid numbers here are **Clarke** and **Parkes**, not the official
**DTS / Surveillance Error Grid** — that one needs the organisers' toolkit, which is
not vendored in this repo. `exp/metrics.py::dts_zones()` raises deliberately rather
than returning a plausible wrong number. Re-score with the official scorer before
trusting any zone-A figure. See `DATA_PIPELINE.md` §11.
