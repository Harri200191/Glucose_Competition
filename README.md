# MetaboNet Glucose Prediction Challenge — XGBoost baseline

Forecast CGM glucose **30 / 60 / 90 / 120 minutes ahead**. Scored on **MARD** and
**DTS error-grid A-zone %** as two independent tracks (RMSE only breaks ties).

- Challenge: <https://metabo-net.org/glucose-prediction-challenge>
- Leaderboard: <https://metabonetglucose-leaderboard.hf.space/>
- Official toolkit: <https://github.com/replicahealth/glucose-prediction-submission-toolkit>

## Layout

```
data/metabonet_public.parquet   corpus you downloaded — 154,842,077 rows x 37 cols (1.36 GB)
toolkit/                        official metrics.py + run.py (vendored from the toolkit repo)
toolkit_data/                   official template + targets, pulled via GitHub's LFS media endpoint
gpc/data.py                     re-gridding + windowed feature construction
gpc/run_xgboost.py              training, scoring, submission writing
runs/                           models, submission parquets, results.json
```

`toolkit_data/` was fetched without `git-lfs` via
`https://media.githubusercontent.com/media/<owner>/<repo>/main/<path>` — a plain
`git clone` without LFS installed yields pointer stubs instead.

## The data

`is_test` splits the corpus: 132.8M train rows / 1,725 subjects, 22.1M test rows /
351 subjects. The subject key is **`(source_file, id)`** — `id` alone is reused
across source datasets. 169 subjects are split *temporally* (train = earlier
window, test = later), so a random row split leaks; always split by subject.

The live leaderboard scores a **filtered** subset of the test split — 2,648,987 rows
across 279 subjects and 9 source datasets. Every sample is guaranteed 24h of history
with no CGM gap > 30 min, no insulin 0/NaN run > 120 min, and at least one carb or
meal event in the window. `DCLP3`, `DCLP5`, `Flair` (no carb data) and `T1D-UOM`
(MDI only) are dropped entirely. Practical consequence: **a CGM-only model is
leaving signal on the table by construction** — the test set was filtered precisely
so multimodal models get credit.

The annual-competition template is only **2,228 rows** across 89 subjects, under a
synthetic `source_file` of `annual_competition_secret_holdout_set`. Its history is
*not* in the public corpus — you need the separate 22.7 MB holdout download.

## Features (102)

Every subject is re-gridded onto a dense 5-minute lattice first, so lags are
time-correct rather than row-correct. Missing values are left as NaN — XGBoost
learns a default split direction, which beats any imputation we'd invent.

| Block | Detail |
|---|---|
| CGM lags | 17 taps, 0–288 steps (now → 24h back), dense over the last hour |
| CGM deltas | differences + per-minute rates over 1/2/3/6/12/24 steps |
| CGM rolling | mean/std/min/max over 30m, 1h, 2h, 4h, 6h, 24h |
| Dose sums | bolus / basal / insulin / carbs / steps summed over the same 6 windows |
| Decay | exponential IOB and COB proxies (halflife 55 / 45 min) |
| Recency | steps since last carb event, last CGM reading, last bolus |
| Activity | heart-rate means, workout intensity, calories, meal count |
| Calendar | time-of-day sin/cos, day-of-week |
| Static | age, weight, height, age at diagnosis, gender, source dataset |

## Run

```bash
python -m gpc.run_xgboost --gpu 0 --workers 48 --predict-annual
```

Useful knobs: `--train-stride` (5-min slots between training samples; 6 = 30 min),
`--max-train-rows`, `--val-frac` (held out **by subject** for early stopping),
`--max-depth`, `--eta`, `--gpu -1` for CPU.

Outputs `runs/submission_live.parquet` in exact template row order, plus
`runs/results.json` and one saved model per horizon.

Verify before submitting:

```bash
cd toolkit && python run.py ../runs/submission_live.parquet --competition live --horizon all
```

## Submission rules that bite

- Exact template rows, columns, **and row order**.
- All four `pred_*` columns must exist. Fully populate **at least one**; any horizon
  you skip must be *entirely* NaN. Partially-filled columns are rejected outright.
- Annual competition: pick up to 3 saved submissions for final evaluation; during
  the competition they are identified by file MD5 only.
