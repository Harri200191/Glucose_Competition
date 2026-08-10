# Data pipeline — every manipulation, and why

Complete record of what happens between `data/metabonet_public.parquet` and the
matrix the models actually see. Every decision cites the section of `data.ipynb`
that justifies it.

**Nothing in this document is a default.** Each choice is either forced by a
property of the corpus or is a judgement call, and the judgement calls are marked
**[CHOICE]** with the alternative that was rejected.

Machine-readable twin: `exp/clean.py::COLUMN_DECISIONS` is dumped verbatim into
`build/<name>/manifest.json` on every build, so any result traces back to the exact
treatment that produced it.

---

## 0 · Source

| | |
|---|---|
| File | `data/metabonet_public.parquet` |
| Rows | 154,842,077 |
| Columns | 37 |
| Subjects | 1,907, keyed on `(source_file, id)` |
| Source cohorts | 14 |
| Date range | 2011-11-17 → 2027-07-14 (shifted; the max is in the future) |
| On disk / resident | 1.36 GB / 60.6 GB |

Only **14 of 37 columns are read** (`exp/clean.py::LOAD_COLUMNS`). This cuts the
load from ~61 GB / 30 s to ~11 GB / 17 s. The other 23 are excluded before they
ever reach memory — see §1.

---

## 1 · Column decisions (all 37)

### 1.1 Kept as features (12 source columns → 113 engineered features)

| Column | Treatment |
|---|---|
| `CGM` | Target **and** the dominant history channel. See §3, §4, §5. |
| `basal` | Units per 5-min slot. Window sums + exponential decay. Never used raw. |
| `bolus` | Units. Window sums + IOB decay + recency. Never used raw. |
| `carbs` | Grams. Window sums + COB decay + recency. Never used raw. |
| `age` | Static, passed through. |
| `age_of_diagnosis` | Static; also derives `diabetes_duration`. |
| `height` | **Converted feet → metres.** |
| `weight` | **Converted pounds → kilograms.** |
| `gender` | Normalised (4 stored levels → 3 real ones). |
| `insulin_delivery_modality` | AID / SAP / MDI, 3 levels, kept as-is. |
| `source_file` | Categorical feature **[CHOICE]**, toggled by `--no-cohort-feature`. |
| `date` | Never a feature. Derives time-of-day, day-of-week, elapsed-in-segment. |

### 1.2 Dropped, with reasons (23 columns)

Grouped by why, not alphabetically — the reason is the point.

**Redundant (1)**

| Column | Reason |
|---|---|
| `insulin` | **Exactly** `basal + bolus`: max residual 3.6e-15 over 134,035,414 rows where all three exist (EDA 10.1). Keeping it triple-counts one signal and gives tree splitters three correlated ways to say the same thing. |

**Statistically nonexistent (5)** — >99 % null, and present in only one or two cohorts, so their *presence* identifies the study (EDA 3.3)

| Column | Null % | Extra reason |
|---|---|---|
| `air_temp` | 99.94 | One cohort. In Fahrenheit. |
| `galvanic_skin_response` | 99.91 | φ = 0.999 co-missing with `skin_temp` (EDA 3.4). 1,071 literal zeros = sensor-off stored as data. |
| `skin_temp` | 99.91 | Same 1,071 zeros. In Fahrenheit. |
| `heartrate` | 99.33 | Wearable cohorts only. |
| `steps` | 99.38 | Wearable cohorts only. |

**Degenerate or physically impossible (4)**

| Column | Reason |
|---|---|
| `workout_intensity` | **219 non-null values in 154,842,077 rows** (EDA 5.1). |
| `workout_duration` | 99.94 % null; 11 rows exceed a day, max 8,218 min = 5.7 days. |
| `calories_burned` | 76.4 % null, and 7,127 slots exceed 500 kcal per 5 minutes, max 21,726 (EDA 12). |
| `is_pregnant` | 33 % null, `True` for only 10 subjects. A near-perfect single-cohort indicator. |

**Free text (4)**

| Column | Reason |
|---|---|
| `meal_label` | 99.94 % null, 2,385 distinct strings. `carbs` already quantifies the same events. |
| `workout_label` | 99.93 % null, 80 strings. |
| `misc_notes` | 99.97 % null. |
| `ethnicity` | 30 comma-joined multi-select combinations, not a categorical (EDA 6.4). Strongly cohort-correlated. Excluded to avoid encoding study identity, and because a raw ethnicity feature in a clinical model is a fairness hazard that needs a deliberate justification this task does not have. |

**Study identity dressed as therapy context (5)** — each is ~90 % co-missing with cohort (EDA 3.4) and changes mid-subject (EDA 8.5)

| Column | Reason |
|---|---|
| `cgm_device` | Sensor model. Era variable, mostly a cohort proxy. |
| `insulin_delivery_device` | Pump model. Same. |
| `insulin_delivery_algorithm` | Control-IQ / Loop / 670G — nearly collinear with modality + cohort. |
| `insulin_type_basal` | 20 unnormalised levels collapsing to far fewer; nearly all rapid-acting analogues with equivalent PK. Adds identity, not pharmacology. |
| `insulin_type_bolus` | 17 levels, same collapse. |

**Trial bookkeeping (3)** — exist only for the randomised studies, so they *are* the study label

| Column | Null % |
|---|---|
| `treatment_group` | 64.7 |
| `randomization_date` | 77.9 |
| `extension_date` | 94.2 |

**Split metadata (1)**

| Column | Reason |
|---|---|
| `subject_split_across_traintest` | Describes the evaluation design. Feeding it to a model leaks how it will be scored. |

### 1.3 The one that deserves an argument: `source_file`

**[CHOICE] Included by default; `--no-cohort-feature` turns it off.**

For: it is genuinely available at inference time — the test rows carry it — so using
it is legal and the organisers' own baseline uses "source dataset" as a static
feature.

Against: EDA 3.3 showed the null pattern alone nearly determines the cohort (12 of 14
cohorts have a unique >50 %-null signature). Cohorts differ in population — paediatric,
pregnancy, MDI, closed-loop — so a model can score well by learning "which study is
this, what is its mean glucose" rather than physiology.

Resolution: keep it, but make the shortcut **measurable** rather than invisible. Run
the same model under `--split cohort` (leave-one-study-out) and compare. If the gap
between the subject-split and cohort-split MARD is large, the model is leaning on
identity. Also build a `--no-cohort-feature` dataset and diff.

---

## 2 · Row-level cleaning

`exp/clean.py::clean_frame`. Counts land in `manifest.json → stats.clean_log`.
**No step here imputes anything** — imputation is a modelling decision and lives in
the model adapters (§9), where it can differ per learner and be recorded.

| # | Rule | Rationale |
|---|---|---|
| 1 | `CGM` outside **[10, 600] mg/dL → NaN** | Physically impossible; a sensor fault, not a reading. The corpus minimum is 1.80. **The 40 and 400 clamp values are deliberately kept** — they are real readings the device chose to report (EDA 9.2). |
| 2 | `bolus < 0 → 0` | 4 rows, values −3 and −1 (EDA 12). A negative dose is not physical. |
| 3 | `basal < 0 → 0`, `carbs < 0 → 0` | Defensive; none observed. |
| 4 | `height_m = height × 0.3048`, `weight_kg = weight × 0.45359237` | The corpus mixes imperial and metric without declaring it (EDA 5.7). Raw `height`/`weight` are then **dropped** so nothing downstream can use the ambiguous units. |
| 5 | `bmi = weight_kg / height_m²`; BMI outside **[10, 60] → NaN** | Derived. Out-of-range means one of the inputs is wrong, and NaN is honest where a fabricated value is not. |
| 6 | `diabetes_duration = age − age_of_diagnosis`; negatives → NaN | Zero violations observed (EDA verify), so this is a guard, not a fix. |
| 7 | `gender`: `Non-binary` and `Non Binary` → one level | Two spellings of one level (EDA 6.3). 4 stored → 3 real. |
| 8 | Dose/carb NaN **left as NaN here** | Semantics differ by use: for a *window sum* the neutral element is 0 (applied in §5), but for a *presence* feature NaN is information. Applying one rule globally would destroy the other. |

---

## 3 · Segmentation

`exp/segments.py`. This is the step most likely to be skipped, and skipping it
silently corrupts every lag feature.

### 3.1 Two different things, kept apart

**Segment** — a maximal run of rows that are the same subject *and* exactly 5 minutes
apart. A property of the lattice, not of the signal.

**CGM gap** — a run of NaN in the target *inside* a segment. Does **not** break the
segment, because insulin and carbs keep recording through it.

Conflating them is the mistake this module exists to prevent.

### 3.2 Why it matters here

EDA 7.1: of 154,840,168 adjacent within-subject row pairs, **154,840,166 are exactly
300 s apart**. The lattice is dense and already sorted. But there are exactly two
violations, both in `ShanghaiT1DM`:

| Subject | What is stored |
|---|---|
| `1002` | Two wear periods concatenated; +118.4-day jump between them |
| `1006` | Two wear periods concatenated, **stored out of chronological order** — a −39.7-day jump |

A plain `df.shift(1)` on subject `1006` computes a "5-minute delta" across 39.7 days
running **backwards**. Segmentation cuts there, so it cannot happen. 1,907 subjects
→ **1,909 segments**.

### 3.3 Rules

| Rule | Value | Effect |
|---|---|---|
| Cut where consecutive rows are not exactly 5 min apart | always | 1,909 segments |
| Drop segments shorter than `min_segment_min` | **6 h** (72 steps) | Too short to ever carry the 24 h history an anchor needs |
| Every lag / rolling window is NaN if it crosses a segment edge | always | `exp/features.py::blocked_*` |

Verified in `manifest.json → stats.segments`, including
`all_segments_time_monotone`, which must be `true`.

---

## 4 · CGM interpolation — features only, never targets

**[CHOICE] Linear interpolation of CGM gaps ≤ 30 min (6 slots), inside a segment
only.** Longer gaps stay NaN.

Justified by EDA 7.4: 83 % of the 1,147,210 gaps are ≤ 30 min but they hold only
**6.6 % of the lost time**. So a short-gap-only rule recovers most *events* while
fabricating almost none of the missing signal. Interpolating everything would invent
the overwhelming majority of it; the largest single gap is 636 days.

Guards:

- A gap is only filled if it has a real reading on **both** sides **within the same
  segment**. A gap touching a segment edge is never filled.
- The filled series (`cgm_f`) is used **only to build features**.
- **The prediction target is always a real reading.** Targets are read from the raw
  column, so an interpolated slot yields NaN and the sample is dropped (§6).

Rejected alternative: forward-fill. It creates flat plateaus that the `cgm_diff_*`
and `cgm_rate_*` features read as "glucose is perfectly stable", which is the
opposite of what a sensor dropout means.

---

## 5 · Sample (anchor) selection

A *sample* is one lattice row used as the "now" of a prediction. Filters apply in
this order, and each count is recorded in `manifest.json → stats`:

| # | Filter | Why |
|---|---|---|
| 1 | ≥ 24 h of history within the segment | Matches the leaderboard filter (EDA 13); the longest feature window is 288 steps |
| 2 | The anchor row itself has a **real** CGM reading | You cannot forecast from an imputed present |
| 3 | **No CGM gap > 30 min anywhere in the 24 h history** | The leaderboard's own quality bar (EDA 13) |
| 4 | At least one of the four horizons has a **real** target | A sample with no label is not a sample |
| 5 | **Stride**: keep one anchor every `--stride` steps (default 6 = 30 min) | **[CHOICE]** EDA 9.6: r(CGM_t, CGM_t+5min) ≈ 0.99, so consecutive anchors are near-duplicates. Striding cuts cost ~6× at negligible information loss. Stride is on *position within the segment*, so the phase is stable rather than drifting. |
| 6 | Optional global cap (`--max-samples`) | Only for smoke tests |

Filter 3 is the aggressive one. It is deliberate: it makes the local training
distribution resemble the scored distribution. Relax with `--max-gap-min` if you want
a larger, dirtier training set.

---

## 6 · Targets

For horizon *h* ∈ {30, 60, 90, 120} min = *k* ∈ {6, 12, 18, 24} steps:

```
y_h[i] = CGM_raw[i + k]      if rows i..i+k are in the same segment
         NaN                 otherwise
```

- Read from **`CGM_raw`**, never `cgm_f` — an interpolated label is a fabricated label.
- All four horizons are stored per sample (`targets.npy`, shape `[n, 4]`). A run
  selects one column and drops rows where it is NaN, so the four horizons have
  slightly different sample counts.
- One model per horizon. No multi-output sharing. **[CHOICE]** — simpler, and it lets
  each horizon early-stop on its own validation curve.

---

## 7 · Feature engineering — all 113

`exp/features.py`. Every window is invalidated where it crosses a segment edge.
Units: CGM mg/dL, insulin units, carbs grams, time minutes.

### 7.1 CGM level (18)

`cgm_now`, then `cgm_lag_k` for k ∈ {1, 2, 3, 4, 5, 6, 8, 10, 12, 18, 24, 36, 48, 72,
96, 144, 288} — dense over the last hour, thinning out to 24 h back. Taken from the
short-gap-interpolated series.

### 7.2 CGM dynamics (13)

- `cgm_diff_k = cgm_now − cgm_lag_k` for k ∈ {1, 2, 3, 6, 12, 24}
- `cgm_rate_k = cgm_diff_k / (5k)` — mg/dL per minute
- `cgm_accel = (cgm_now − lag_1) − (lag_1 − lag_2)` — second difference

### 7.3 CGM shape (36)

- Rolling `mean`, `std`, `min`, `max`, `range` over w ∈ {6, 12, 24, 48, 72, 288}
  (30 min, 1 h, 2 h, 4 h, 6 h, 24 h). Windows tolerate up to 50 % NaN
  (`min_frac=0.5`) and return NaN below that.
- `cgm_ewm_{3,6,12}` — exponentially weighted means, half-life in steps, reset at
  segment starts.
- `cgm_tir_frac_288`, `cgm_hypo_frac_288`, `cgm_hyper_frac_288` — fraction of the last
  24 h in 70–180, below 70, above 180. Per-subject glycaemic context (EDA 9.3).

### 7.4 Dose and carbohydrate (27)

EDA 5.4 showed `bolus` is 93 % exact zeros and `carbs` 98 % — spike trains, not
continuous signals. **No instantaneous dose value is ever a feature.** Instead:

- `{bolus,basal,carbs}_sum_w` for the six windows — 18 features. NaN is treated as 0
  **inside the sum only** (the neutral element), while the window is still invalidated
  at segment edges.
- `iob` — insulin-on-board proxy: first-order IIR filter `y[i] = α·y[i−1] + bolus[i]`
  with α from a **55-minute half-life**, reset at every segment start.
- `cob` — same on `carbs` with a **45-minute half-life**.
- `basal_ob` — same on `basal` with the insulin half-life.
- `min_since_bolus`, `min_since_carb` — recency.
- `last_bolus_units`, `last_carb_grams` — magnitude of the most recent event.
- `n_carb_events_288`, `n_bolus_events_288` — event counts in 24 h.

Justified directly by EDA 10.2: the measured carbohydrate response peaks **45–75 min**
after the event and decays over 3–4 h. The decay kernels encode that shape; the
instantaneous column cannot.

Implementation note: the IIR filters run per segment via `scipy.signal.lfilter`. The
closed-form `α^i · cumsum(x / α^i)` overflows once *i* reaches the tens of thousands,
and segments here reach 152,982 rows.

### 7.5 Sensor health (4) — `--no-missingness-features` to disable

`cgm_missing_frac_12`, `cgm_missing_frac_288`, `min_since_real_cgm`,
`max_cgm_gap_288`.

**[CHOICE] Kept, with a caveat.** These are legitimate signal — a model should know
its input is stale. But EDA 3.3 showed missingness is cohort-structured, so they
partially encode study identity. They are grouped separately in the manifest so a
`--no-missingness-features` build can measure their contribution.

### 7.6 Calendar (6)

`tod_sin`, `tod_cos`, `dow_sin`, `dow_cos`, `is_weekend`, `hours_into_segment`.

Time-of-day is encoded **cyclically** so 23:55 sits next to 00:00. EDA 9.4 measured a
15–25 mg/dL diurnal swing — comparable to a 30-minute prediction error, so this is
load-bearing.

**Absolute calendar time is never a feature.** EDA 7.2: timestamps are shifted per
study and the maximum is in the future, so cross-cohort calendar alignment is fiction.

### 7.7 Static (9)

Numeric: `age`, `age_of_diagnosis`, `height_m`, `weight_kg`, `bmi`,
`diabetes_duration`.
Categorical: `gender`, `insulin_delivery_modality`, `source_file`.

Categoricals are stored as **integer codes plus a level table** in the manifest, so
each model adapter picks its own encoding (native categorical for LightGBM/CatBoost,
ordinal-as-numeric for XGBoost, embedding-free scaled input for the NNs) without the
dataset committing to one.

---

## 8 · Sequence channels (for `tcn` / `gru`)

Materialising a 24 h window per sample would cost 288 × 5 × 4 B × 10 M ≈ **57 GB**.
Instead `channels.npy` stores five channels at **full length** (n_rows × 5 × 4 B
≈ 3 GB) and the loader gathers windows lazily through `row_idx`.

Fixed transforms, applied once at build time:

| Channel | Transform |
|---|---|
| `cgm` | `cgm_f / 100 − 1.5` (roughly zero-centred) |
| `cgm_present` | `1.0` where the reading is **real**, `0.0` where interpolated or missing |
| `bolus` | `log1p(x)` |
| `basal` | `log1p(x)` |
| `carbs` | `log1p(x)` |

`log1p` because these are spike trains with heavy tails (EDA 5.4: skew 15.8 for
bolus, 14.1 for carbs). Default window: 96 steps at stride 3 = **24 h at 15-min
resolution**.

---

## 9 · NaN policy — per model, not global

The dataset ships NaN **as NaN**. That is the whole point: EDA 3.3 showed missingness
is cohort-structured, so any globally imputed value is a fabricated cohort-specific
constant, and worse, it destroys the information that the value was missing.

| Model | Policy |
|---|---|
| XGBoost, LightGBM, CatBoost, HistGB | **Native.** Learn a default split direction. Nothing imputed. |
| TabPFN | Native (internal preprocessor). |
| RandomForest, ExtraTrees | **Median imputation + up to 40 missingness indicators.** sklearn forests cannot take NaN. |
| Neural nets | Same imputation, then **median/IQR scaling** clipped to ±8. |

Every imputer is **fitted on train only** and its configuration is written to
`result.json → model_notes.imputation`. Median/IQR rather than mean/sd because EDA 5.2
found skew in the tens to hundreds, where mean/sd is dominated by the tail.

---

## 10 · Splits

EDA 11 forces this. `exp/splits.py`.

| Scheme | What it does | Use for |
|---|---|---|
| `official` | The competition's `is_test` flag, plus a **subject-disjoint** validation slice carved from train | The headline number |
| `subject` | Group K-fold on `(source_file, id)` within the train portion | Model selection |
| `cohort` | Leave-one-study-out | Detecting the cohort shortcut |

Non-negotiables enforced in code (`train.py` raises, it does not warn):

1. **Never split by row.** r(CGM_t, CGM_t+5min) ≈ 0.99 — a random row split is
   near-total leakage.
2. **Train and validation never share a subject.** Checked every run.
3. The `official` split *does* have train/test subject overlap — 166–169 subjects
   sit on both sides, split **temporally** (their earlier window is train). This is
   the competition's own design, so it is allowed there and only there.

`leaderboard_mask()` additionally drops `DCLP3`, `DCLP5`, `Flair`, `T1D-UOM` to
approximate the actually-scored slice (EDA 13). Reported as a separate line, because
persistence alone scores 13.52 % there vs 14.03 % corpus-wide — the slices are not
interchangeable.

---

## 11 · Metrics

`exp/metrics.py`. Primary: **MARD**. Secondary: RMSE, MAE, ISO-style 15/15 and 20/20
agreement, MARD restricted to hypo (<70) and hyper (>250), bias, and error-grid
zone fractions.

Also reported **per subject** (mean, median, p90, worst) and **per cohort**. EDA 8.3:
`Loop` is 61 % of rows, so a row-weighted score is largely a score on `Loop`.

> ### Error-grid caveat — read before quoting a zone-A number
> The official track uses the **DTS / Surveillance Error Grid**, whose zones come
> from a published continuous risk surface shipped with the organisers' toolkit.
> That toolkit is not vendored here. What is implemented is **Clarke** (classical,
> unambiguous) and **Parkes/Consensus Type 1** (from the published boundary vertices
> in Pfützner et al. 2013). Both are reasonable proxies; neither is the official
> scorer. `dts_zones()` raises `NotImplementedError` on purpose rather than being
> silently wrong. **Re-score with the official toolkit before submitting.**

### Training objective

**[CHOICE] Every model regresses `log(glucose)` under an L1-type loss.**

`d|log p − log t| / dp ∝ 1/p`, which is the same `1/t` weighting MARD applies. So the
training signal matches the scoring metric, and capacity goes to the hypoglycaemic
end — which EDA 9.2 showed dominates MARD despite being ~4 % of readings. Squared
error on the raw scale remains available (`xgboost_l2`, `catboost_gpu`) as a control.

---

## 12 · What is deliberately NOT done

| Not done | Why |
|---|---|
| Mean/median imputation in the dataset | Missingness is cohort-structured (EDA 3.3); imputing fabricates a study-specific constant and erases real signal. Pushed to the model layer instead. |
| Forward-filling CGM | Creates flat plateaus that the rate features misread as "stable". |
| Interpolating long gaps | Would fabricate ~93 % of the missing time (EDA 7.4). |
| Outlier removal on CGM 40/400 spikes | They are real device outputs, not errors. The censoring is *recorded* (EDA 9.2) rather than deleted. |
| Rank-gauss / quantile transforms | Tree models are invariant to monotone transforms; for the NNs it is a leak hazard unless fitted on train only, and median/IQR scaling already handles the skew. |
| Resampling to a regular grid | The lattice is already dense and exactly 5-minutely (EDA 7.1). A resample step would be a no-op that hides the two real discontinuities. |
| Per-subject target normalisation | Test subjects include unseen people; there is no legitimate per-subject statistic at inference for them. |
| Using `insulin`, `ethnicity`, device/algorithm columns | §1.2 and §1.3. |

---

## 13 · Reproducibility

Each build writes `build/<name>/manifest.json` containing the full config, the
complete `COLUMN_DECISIONS` table, the cleaning log with counts, the segmentation
audit, the anchor-attrition counts at every filter stage, all 113 feature names with
their groups and NaN fractions, and the categorical level tables.

Each run writes `runs/<tag>/result.json` containing the metrics, the model's own
notes (objective, best iteration, imputation config, device, training history), and
an embedded copy of the data manifest's identifying fields. A result is therefore
self-describing: you never have to guess which data version produced it.

Seeds are threaded through config → split → model. Sources of residual
non-determinism: GPU kernel reduction order (XGBoost `hist` on CUDA, cuDNN), and
CatBoost's GPU trainer.

---

## 14 · Known limitations

1. **DTS error grid is not implemented** (§11). Clarke and Parkes are proxies.
2. **The leaderboard subset is approximated, not reproduced.** The EDA's
   reconstruction reached 9,975,130 rows / 220 subjects against the official
   2,648,987 rows / 279 subjects — an upper bound. The organisers also constrain
   insulin outages and use their own sampling stride.
3. **`source_file` as a feature is a live risk** (§1.3). Mitigated by measurement
   (`--split cohort`), not by assumption.
4. **TabPFN sees only a bagged subsample** of training data (§9 of the model docs);
   its MARD is not directly comparable to a model trained on all rows. Recorded in
   `model_notes.caveat`.
5. **Stride 6 is a cost/fidelity trade** (§5). Justified by autocorrelation, but a
   stride-1 build would be strictly more informative if you can afford it.
6. **The wearable channels are dropped entirely** rather than used within their own
   cohorts. A cohort-specific model could use heart rate and steps; this pipeline
   targets one global model.
