# What this project is, in plain language

This document assumes you know nothing about this repository, nothing about the
machine-learning terms in it, and nothing about what I did. It explains all of
it, in order, from the medical problem to the final result.

It is long. Read it in parts. Every technical term is defined the first time it
appears, and there is a glossary at the end.

---

# Table of contents

1. [The medical problem](#1-the-medical-problem)
2. [What "forecasting glucose" actually means](#2-what-forecasting-glucose-actually-means)
3. [The data we have](#3-the-data-we-have)
4. [The vocabulary you need](#4-the-vocabulary-you-need)
5. [What the project was supposed to answer](#5-what-the-project-was-supposed-to-answer)
6. [The five model architectures, explained](#6-the-five-model-architectures-explained)
7. [The four fusion strategies, explained](#7-the-four-fusion-strategies-explained)
8. [What I actually built](#8-what-i-actually-built)
9. [The experiments and what they found](#9-the-experiments-and-what-they-found)
10. [The big discovery: future insulin](#10-the-big-discovery-future-insulin)
11. [The competition submissions](#11-the-competition-submissions)
12. [What it all means](#12-what-it-all-means)
13. [Where everything lives](#13-where-everything-lives)
14. [Glossary](#14-glossary)

---

# 1. The medical problem

## Type 1 diabetes

In type 1 diabetes the pancreas stops making **insulin**. Insulin is the hormone
that lets your cells absorb glucose (sugar) from the blood. Without it, glucose
builds up in the blood instead of being used.

So a person with type 1 diabetes has to inject or infuse insulin themselves,
for the rest of their life, and they have to get the amount right:

- **Too little insulin** → blood glucose stays high (**hyperglycaemia**). Over
  years this damages eyes, kidneys, nerves and blood vessels.
- **Too much insulin** → blood glucose crashes (**hypoglycaemia**). This is
  dangerous *immediately* — confusion, seizure, unconsciousness, and it can kill
  within hours.

The target range is roughly **70–180 mg/dL** (milligrams per decilitre). Below 70
is hypoglycaemia; below 54 is severe. Above 180 is hyperglycaemia; above 250 is
severe.

Note the asymmetry: high glucose harms you over decades, low glucose can harm you
this afternoon. That asymmetry matters later.

## What raises and lowers glucose

- **Carbohydrate** (food) raises it, starting ~15 minutes after eating and
  peaking around an hour later.
- **Insulin** lowers it, starting ~15 minutes after a dose and acting for 3–5
  hours.
- **Exercise** usually lowers it, sometimes for many hours afterwards.
- Stress, illness, sleep, hormones, and the injection site all shift it too.

The practical problem is that insulin acts *slowly* and food acts *fast*, so you
are always dosing for a future you cannot see.

## CGM: the device that made this a data problem

A **CGM** (continuous glucose monitor) is a small sensor worn on the skin with a
filament under it. It measures glucose in the interstitial fluid (the fluid
between cells, which tracks blood glucose with a lag) and reports a number
**every 5 minutes**, day and night.

That is 288 readings a day, 105,000 a year, per person. It turned diabetes
management from occasional finger-prick snapshots into a continuous data stream —
which is why machine learning became relevant.

## Three ways people deliver insulin (this becomes very important)

| Mode | What it means | Who decides the dose |
|---|---|---|
| **MDI** | Multiple daily injections — a pen or syringe | The person |
| **SAP** | Sensor-augmented pump — a pump delivering a pre-programmed rate | The person programs it in advance |
| **AID** | Automated insulin delivery, "closed loop", "artificial pancreas" | **An algorithm**, automatically, every 5 minutes |

In an **AID** system the pump reads the CGM, runs a control algorithm, and
adjusts insulin by itself — every five minutes, forever. The person only
announces meals. Commercial examples: Medtronic 670G, Tandem Control-IQ, and the
open-source Loop.

Remember this. It is the single most important fact in this entire document.

---

# 2. What "forecasting glucose" actually means

## The task

Stand at some moment in time. Call it **now**, or *t*. You can see:

- the person's CGM readings for the past 24 hours
- every insulin dose they took in that window
- every carbohydrate entry they logged
- some static facts (age, weight, therapy type)

**Question:** what will their glucose be in 30 minutes? In 60? In 90? In 120?

That's it. That's the whole task. Four numbers.

```
        past (known)                    future (to predict)
  ├──────────────────────────────┤  ├─────────────────────────┤
  24 hours of CGM + insulin + carbs      +30   +60   +90  +120 min
                                 ↑
                               "now"
```

## Why those four horizons

- **30 min** — enough time to eat something if you're heading low.
- **60 min** — enough time for a correction dose to start working.
- **120 min** — roughly the horizon over which a meal fully plays out.
- **90 min** sits between; the public benchmark scores it so we keep it.

## Why it's hard

The future depends on things that haven't happened yet and aren't in the data:
whether they're about to eat, whether they'll go for a walk, whether the
infusion site is absorbing properly. A forecast can only capture the part that
is predictable from the past.

There is therefore a ceiling on how good any honest forecast can be, and part of
this project turned out to be about discovering that some published scores are
above that ceiling.

---

# 3. The data we have

## The file

`data/metabonet_public.parquet` — 1.36 GB, and inside it:

- **154,842,077 rows**
- **1,907 patients**
- **14 separate research studies** ("source datasets" or "cohorts") pooled into
  one common format
- Each row = **one patient, one 5-minute time slot**

A row looks conceptually like this:

| patient | timestamp | CGM | basal | bolus | carbs | age | ... |
|---|---|---|---|---|---|---|---|
| Loop-441 | 2024-03-02 14:25 | 142 | 0.05 | 0.0 | NaN | 34 | |
| Loop-441 | 2024-03-02 14:30 | 145 | 0.05 | 3.2 | 45 | 34 | |

- **CGM** — the glucose reading, mg/dL. This is both the thing we predict and
  our main input.
- **basal** — background insulin delivered in that 5-minute slot (units).
- **bolus** — a discrete dose, usually for a meal or a correction (units).
- **carbs** — grams of carbohydrate the person logged.
- **NaN** — "not a number", i.e. missing. Most cells are missing, because meals
  and boluses are rare events: carbs is empty 98% of the time.

There are 37 columns in total; we use 17 and deliberately discard 20 (explained
in section 8).

## The 14 cohorts are wildly unequal

| Cohort | Share of all rows | Insulin mode |
|---|---|---|
| Loop | **61.2%** | 100% AID |
| ReplaceBG | 13.5% | 0% AID (pump) |
| IOBP2 | 6.2% | 100% AID |
| Flair | 4.5% | 100% AID |
| DCLP5 | 3.9% | 100% AID |
| DCLP3 | 3.9% | 100% AID |
| PEDAP | 3.6% | 100% AID (children) |
| CTR3 | 1.4% | 100% AID |
| 6 others | 1.8% | mixed |

**85.8% of all rows come from people on AID** — closed-loop systems where an
algorithm sets the insulin. Loop alone is nearly two-thirds of the data.

Two consequences:

1. Any average over rows is mostly a statement about Loop patients.
2. For 86% of this data, insulin is a *machine's reaction to glucose*, not a
   human's decision. Hold that thought until section 10.

## The competition

There is a public competition on this data
(<https://metabo-net.org/glucose-prediction-challenge>) with two tracks:

- a **live leaderboard** — 2,648,987 rows to predict, answers publicly available
- an **annual competition** — 2,228 rows against a secret held-back answer set

Your professor's project plan (the PDF) is about building a rigorous research
benchmark on this data, not primarily about winning the competition — but we
ended up producing submissions for both.

---

# 4. The vocabulary you need

Everything in this section is a term I used carelessly in earlier messages.

## Window, anchor, horizon, target

- **Anchor** — the moment we stand at and forecast from. The "now". In the data
  it's one specific row of one specific patient.
- **Window** (or **context**) — how much history the model gets to look at,
  ending at the anchor. We mostly use **288 steps = 288 × 5 min = 24 hours**.
- **Horizon** — how far ahead we predict. 30 / 60 / 90 / 120 minutes.
- **Target** (or **label**) — the true CGM value at the horizon. This is what we
  compare the prediction against. It comes from the data, from rows *after* the
  anchor.

One **sample** = one anchor = (a 24-hour window of inputs, four target values).

We have **16,054,652 anchors** — 16 million training/testing examples.

Why not one anchor per row (154 million)? Because consecutive rows are almost
identical. The correlation between CGM now and CGM five minutes from now is
about 0.99. Ten thousand nearly-identical examples teach a model roughly as much
as one. So we take one anchor every 30 minutes (a **stride** of 6 steps).

## Training, validation, test

You split the data into three parts:

- **Training set** — the model learns from this.
- **Validation set** — used *while* training to decide when to stop, and to pick
  settings. The model never learns from it directly.
- **Test set** — touched only at the very end, to report the final number.

**The critical rule here: split by patient, not by row.** If patient A's Monday
is in training and patient A's Tuesday is in test, the model can memorise
patient A rather than learn physiology, and the test score is inflated. So all
of a patient's data goes to exactly one side.

(The official benchmark makes one deliberate exception: some patients have their
*early* data in training and their *later* data in test. That's a legitimate
design — it measures "how well do we do on a patient we've seen before" — and we
report it as a separate stratum.)

## Leakage

**Leakage** is when information from the future, or from the answer, sneaks into
the model's inputs.

The reason it's the number-one concern in this kind of work: a leaking model
looks *excellent*. You cannot spot it by looking at the accuracy — high accuracy
is exactly the symptom. You can only catch it by auditing where every input came
from.

Classic examples in time-series work:

- Computing an average that includes future values ("centred smoothing")
- Normalising your data using statistics computed over the whole dataset
  including the test set
- Off-by-one errors that let the model see the value it's predicting
- Using a feature that is a consequence of the outcome rather than a cause

Section 10 is about a subtle case of the last one.

## What "training a model" means here

A **model** is a mathematical function with millions of adjustable numbers
(**parameters** or **weights**). Training means:

1. Show it a batch of samples (say 256 windows).
2. It produces predictions.
3. Compare predictions to the true targets → a single number called the
   **loss** (how wrong it was).
4. Compute which direction to nudge every parameter to make the loss smaller
   (backpropagation).
5. Nudge them slightly. Repeat millions of times.

An **epoch** is one pass over the training data. Our runs do up to 20 epochs on
2 million windows, which takes a few hours on one GPU.

A **seed** is the random number that determines the initial random parameter
values and the shuffling order. Same seed = same run exactly. Different seed =
slightly different result, which tells you how much of a difference between two
models is just luck.

## The metrics

**RMSE (root mean squared error)** — average size of the error in mg/dL, with
big errors punished extra. If RMSE is 33, typical predictions are about 33 mg/dL
off. Lower is better.

**MARD (mean absolute relative difference)** — average error as a *percentage of
the true value*. Note this punishes errors at low glucose much harder: being
20 mg/dL wrong when the truth is 40 is a 50% error; being 20 wrong when the truth
is 400 is a 5% error.

**DTS error grid** — the clinically meaningful one, and the competition's
headline metric. Instead of asking "how many mg/dL off were you", it asks **"if a
patient acted on this prediction, how much harm could result?"** Every
prediction lands in a zone:

| Zone | Meaning |
|---|---|
| **A** | No risk — acting on this is fine |
| B | Mild risk |
| C | Moderate risk |
| D | High risk (e.g. you missed a dangerous low) |
| E | Extreme risk (your prediction implies the opposite treatment) |

"DTS-A = 69.4%" means 69.4% of predictions were clinically harmless. This is
better than RMSE for judging usefulness, because a 30 mg/dL error at 250 is
tolerable while the same error at 60 could be lethal.

---

# 5. What the project was supposed to answer

Here is the plan's central idea, stated plainly.

Most people approach this by throwing every column into one table and handing it
to a neural network. The plan argues that's wasteful, because **the signals mean
different things**:

- **CGM** is the *state* of the system — the thing evolving, and the thing we
  predict.
- **Insulin and carbohydrate** are *interventions* — sparse events that get
  injected into the system and change how the state evolves.
- **Age, therapy type** are *context* — they modify baseline behaviour but
  aren't part of the dynamics.

The hypothesis: if you build a model that respects this structure — treating CGM
as the state and insulin/carbs as forces acting on it — it should forecast better
than one that treats all columns as equivalent.

The plan proposes testing this by holding everything else fixed and varying two
things:

1. **The temporal backbone** — the part that processes the time sequence (five
   candidates).
2. **The fusion strategy** — how the different signals get combined (four
   candidates, of increasing structure).

Plus supporting experiments: how much history to use, which signals matter, and
whether conclusions hold for new patients.

The next two sections explain those two lists, because those are the terms I
used without ever defining.

---

# 6. The five model architectures, explained

First: **what is a "backbone"?**

Our model has stages. Something turns the raw numbers into a useful internal
representation; something processes the *time* dimension; something turns the
result into four predicted numbers. The middle part — the bit that reads a
sequence and understands how it evolves over time — is the **backbone**. It's
the engine.

All five backbones below do the same job: take a sequence of 288 vectors (one per
5-minute step) and produce a processed sequence that captures the temporal
patterns. They differ in *how*.

## 6.1 TCN — Temporal Convolutional Network

**The idea:** slide a small window over the sequence and look for local patterns,
then stack layers so each layer sees a wider span.

Think of it like this. Layer 1 looks at 3 consecutive readings at a time — it can
learn "glucose is rising steeply". Layer 2 looks at 3 outputs of layer 1, spaced
2 apart — so it effectively sees 7 original readings. Layer 3 spaces them 4
apart, seeing 15. By layer 8 a single output can "see" 288 steps back.

That spacing trick is called **dilation**, and the total span a layer can see is
its **receptive field**.

- **Strength:** fast, simple, extremely well-understood.
- **Weakness:** its memory is a fixed window. It cannot remember something from
  20 hours ago unless it's deep enough to physically reach that far.
- **Why it's in the study:** it's the strong, boring baseline. If nothing beats
  the TCN, that's a real finding.

## 6.2 PatchTST — a Transformer over patches

**Transformers** are the architecture behind ChatGPT. The core mechanism is
**attention**: each position in the sequence looks at every other position and
decides which ones are relevant. It's very flexible, but comparing every position
to every other position costs O(n²), which is expensive for 288 steps.

**PatchTST's trick:** don't treat each 5-minute reading as a separate position.
Chop the sequence into **patches** of 16 consecutive steps (80 minutes) and treat
each patch as one token. 288 steps becomes 18 tokens. Now attention is cheap.

- **Strength:** attention can connect distant events directly — "that bolus 6
  hours ago is relevant to right now".
- **Weakness:** patching blurs fine detail inside each 80-minute block.
- **Why it's in the study:** it's the current strong Transformer for time series.

## 6.3 Mamba-2 — a selective state-space model

This is the one you asked about specifically.

**The idea:** carry a running **state** — a memory — through the sequence, one
step at a time. At each step, look at the new input and decide (a) how much of
the existing memory to keep, and (b) how much of the new input to write into it.

If that sounds like an RNN/LSTM, it is closely related. The differences that
matter:

1. **"Selective"** means the keep/write decisions are *computed from the input
   itself*. When a bolus arrives, the model can decide "this is important, write
   it into memory strongly and keep it for hours". During a quiet stretch it can
   decide "nothing happening, just decay the memory".
2. **State-space** refers to the mathematical form — it's a linear recurrence
   borrowed from control theory, `h_t = A·h_{t-1} + B·x_t`, which behaves well
   over long sequences.
3. It's **fast**, because the recurrence can be reformulated so a GPU computes
   long stretches in parallel instead of stepping one-at-a-time. That
   reformulation is called the **SSD (state-space duality)** algorithm, and it
   works by splitting the sequence into chunks, doing an attention-like
   computation *inside* each chunk, and a cheap recurrence *between* chunks.

**Why the plan expected it to win:** glucose has exactly the structure Mamba-2
is designed for. A meal at 8am still matters at 11am. A bolus persists for
hours. Selective memory should handle "remember this dose, ignore this noise"
better than a fixed convolution window.

**A practical note:** the official Mamba-2 implementation needs newer NVIDIA GPUs
(Ampere or later). Your machine has V100s, which are older. So I implemented the
SSD algorithm from scratch in plain PyTorch, and wrote a test that checks it
against a slow, obviously-correct step-by-step version — they agree to 1e-8.

## 6.4 TFT — Temporal Fusion Transformer

A more complicated architecture from Google that has its own opinion about
multimodal data built in. Its distinctive features:

- **Variable selection network** — a small network that learns, at each time
  step, which input variables to pay attention to. It can learn "during the
  night, insulin matters more than carbs".
- **Static enrichment** — patient-level facts (age, therapy) are injected as
  context that conditions the whole sequence.
- **Gating** everywhere — learned on/off switches that let the model skip
  components it doesn't need.

**Why it's in the study, and why it's treated differently:** the plan says TFT
must be used in its *native* form, not wrapped in our fusion module — because
variable selection and gating are TFT's own answer to the exact question we're
asking. Putting our fusion in front of it would test nothing.

## 6.5 TiDE — Time-series Dense Encoder

The deliberately simple one. No attention, no recurrence, no convolution over
time. It just **flattens** the whole 24-hour window into one long vector and
feeds it through a stack of ordinary fully-connected layers.

It also keeps a **linear shortcut** straight from the input CGM to the output,
so the "easy" part of the prediction (roughly: glucose will be near where it is
now, continuing its current trend) is handled by a simple linear map, and the
neural part only has to model the correction.

**Why it's in the study:** as a control. If a model with no temporal machinery
at all matches the sophisticated ones, that tells you the sophistication isn't
buying anything.

---

# 7. The four fusion strategies, explained

**What is "fusion"?**

At each 5-minute step we have several different kinds of information:

- CGM value + a flag saying whether it was actually measured
- basal, bolus, and derived insulin features
- carbs and derived carb features
- (optionally) heart rate and steps

**Fusion** is the question: *how do you combine these into the single vector per
time step that the backbone consumes?*

The plan lists four options, each adding structure to the last. This is the
"ladder" I kept referring to.

### Rung 1 — `raw_concat` (raw concatenation)

Just glue the numbers together. At each time step, take all ~20 values, stick
them in one list, multiply by one weight matrix to get the right size, done.

```
[cgm, cgm_mask, basal, bolus, carbs, ...] → one linear layer → vector
```

This is the required baseline. It treats every input as equivalent. If nothing
beats this, the whole "structure matters" thesis fails.

### Rung 2 — `concat` (separate encoders, then concatenate)

Give each modality its own small neural network first, *then* glue the results
together.

```
CGM channels    → small CGM network    ┐
insulin channels → small insulin network ├→ glue → linear → vector
carb channels   → small carb network   ┘
```

Why this might help: the CGM encoder can specialise in reading a smooth
physiological trace; the insulin encoder can specialise in reading sparse spikes.
They have genuinely different statistics.

### Rung 3 — `gated`

Same as rung 2, but each modality also gets a learned **gate** — a number
between 0 and 1, computed from the data, that scales that modality up or down at
each moment.

```
gate_insulin(t) = sigmoid(W · insulin_features(t))
output = [gate_cgm·CGM_enc ; gate_insulin·insulin_enc ; gate_carb·carb_enc]
```

Why this might help: the model can learn to *ignore* carbohydrate when nothing
has been eaten for six hours, and turn it up sharply right after a meal.

### Rung 4 — `FiLM` (the proposed model)

**FiLM** stands for **Feature-wise Linear Modulation**. This is the plan's
preferred design and the one you asked about.

The idea is to stop treating the modalities as equal partners. Instead:

- **CGM is the state.** It gets an embedding, `E_G(t)`.
- **Insulin, carbs, activity, and patient context are drivers.** They get
  combined into a single **context vector** `c(t)`.
- The context doesn't get concatenated. It **modulates** the state — it reaches
  in and rescales and shifts the CGM representation:

```
    Ē_G(t) = γ(c(t)) ⊙ E_G(t) + β(c(t))
```

Where `γ` (gamma, a multiplier) and `β` (beta, an offset) are both computed from
the drivers by small networks, and `⊙` means element-wise multiply.

**In plain words:** insulin and carbohydrate don't add themselves to the glucose
signal — they change *how the glucose signal should be read*. "There's 4 units of
insulin on board, so interpret this flat trace as about to fall."

This is a real structural claim about physiology: interventions modify the
evolution of the state rather than sitting alongside it. That's why the plan
prefers it.

One implementation detail I added: `γ` is parameterised as `1 + δ` where `δ`
starts at zero. So at the start of training FiLM does *nothing at all* — it's the
identity function — and any modulation the model develops is something it had to
learn and "pay for". This makes the comparison against rung 1 fair.

### And the knock-outs

Beyond the four rungs, I also ran FiLM with pieces removed, to see which
components matter:

- **−events**: remove the derived insulin/carb features (decay curves, time
  since last dose), leaving only raw values.
- **−masks**: remove the flags that say "this value was actually measured"
  vs "this is a filled-in gap".
- **+learned missing**: instead of filling gaps with zeros, give the model a
  learned vector that means "missing".

---

# 8. What I actually built

## 8.0 The starting point

When I started, the repo contained a previous effort: a competition-submission
pipeline (`exp/`, `toolkit/`, some notebooks) built around gradient-boosted trees
and tabular features. It worked, but it was built for a different goal — getting
a leaderboard score — not for the controlled scientific comparison the plan
describes.

You asked me to build the repo from scratch in a totally different structure
following the plan. So I made a new git branch (`metabonet-forecasting-v2`), and
the old code is still safely on `master`.

## 8.1 Step one: turn 154 million rows into something usable

**The problem.** To train, you need windows: 24 hours of history per sample. If
you literally materialise every window you'd store 288 time steps × 23 channels ×
4 bytes × 16 million anchors ≈ **4 terabytes**. Not viable.

**The solution — a "sequence store".** Store the timeline *once* as a big
memory-mapped array, and pull windows out of it on demand by row index.

`metabonet_forecasting/data/prepare.py` does this. Given the parquet file, it
produces a directory containing:

| File | What it holds |
|---|---|
| `channels.npy` | The whole 5-minute grid: 154.8M rows × 23 columns of numbers, 14 GB |
| `anchors.parquet` | One row per valid forecast point: where it is, which patient, time of day, etc. |
| `targets.npy` | The four true future CGM values for each anchor |
| `segments.parquet` | The contiguous stretches of recording |
| `subjects.parquet` | One row per patient with their static facts |
| `manifest.json` | Everything about how it was built, for provenance |

Building this takes **87 minutes** for the full corpus and produces **14 GB**.

**What happens inside, and why each step exists:**

**1. Read only 17 of the 37 columns.** The other 20 are excluded with a written
reason each. Some highlights:
   - `insulin` is *exactly* `basal + bolus` — keeping it triple-counts one signal.
   - Five columns (air temperature, skin temperature, etc.) are >99% empty and
     exist only in one or two studies, so their mere *presence* identifies the
     study. Feeding those in lets the model cheat by recognising the cohort.
   - Device model, trial arm, and randomisation date are study identity in
     disguise for the same reason.

**2. Sort by (patient, time).** Sounds trivial, but two patients in the
ShanghaiT1DM study are stored as two recording periods concatenated, one of them
*backwards in time*. Without sorting, a "5-minute difference" is computed across
a 39.7-day jump running in reverse. Sorting first makes that impossible.

**3. Cut into segments.** A segment is a stretch where consecutive rows really
are 5 minutes apart. Wherever there's a jump, we cut. A patient is not one
continuous time series — they take the sensor off, they drop out of the study.
We drop segments shorter than 6 hours.

**4. Clean each row.** CGM outside 10–600 mg/dL is physically impossible, so it
becomes "missing". Negative doses become zero. Height is in feet and weight in
pounds in this dataset (undeclared!) so they're converted to metric.

**5. Fill short CGM gaps.** Sensors drop out. If a gap is ≤30 minutes we fill it
by drawing a straight line between the readings on either side — but **only for
use as input**, never as an answer. A filled-in value is a guess; using a guess
as the thing you're graded against would be fabricating your own exam.

Why linear interpolation and not "carry the last value forward"? Because carrying
forward creates a flat plateau, and any feature that measures change reads a
plateau as "glucose is perfectly stable", which is the opposite of what a sensor
dropout means.

**6. Build derived features.** From the raw insulin and carb columns we compute:
   - **Exponential exposure curves**: how much insulin is still "on board",
     computed at several different decay speeds (30, 60, 120, 240 minutes for
     bolus; 20, 45, 90 for carbs). This is a rough stand-in for how the body
     processes a dose. It is *not* a validated pharmacological model — which is
     exactly why it can be switched off in the ablation.
   - **Time since last event**: minutes since the last bolus, minutes since the
     last carb entry.

   All of these are computed **causally** — each value uses only past data — and
   reset at each segment boundary so nothing bleeds across a gap.

**7. Choose anchors.** A row can be an anchor only if:
   - it has 24 hours of history within the same segment
   - the CGM at the anchor itself is a real measurement, not a filled gap
   - there's no CGM gap longer than 30 minutes anywhere in that history
   - it passes the stride filter (one every 30 minutes)

   154.8M rows → **16,054,652 anchors**.

**8. Precompute the targets** — with three rules: must be a real reading (never
interpolated), must lie inside the same segment, and must be on the same side of
the official train/test boundary as its anchor.

That last rule matters: since some patients are split by *time* across train and
test, an anchor near the boundary could otherwise be trained using an answer from
the test period.

## 8.2 Step two: the one window dataset

`metabonet_forecasting/data/windows.py`.

This is the single most important file for the study's validity. It is the only
thing that produces training examples, and **every model uses it** — the TCN,
PatchTST, Mamba-2, TFT, TiDE, the LSTM, and even the classical baselines like
ridge regression and gradient-boosted trees.

Why that matters: if the Mamba-2 experiment used slightly different features from
the TCN experiment, then any difference between their scores could be the
features rather than the model, and the whole comparison is worthless.

For a given anchor it produces:

```
x_cgm       288 × 2    the CGM trace and its "was this measured?" flag
x_insulin   288 × 9    basal, bolus, their masks, exposure curves, time-since
x_carbs     288 × 6    carbs, mask, exposure curves, time-since
time        288 × 4    time of day and day of week, encoded as sine/cosine
static_num  6          age, diabetes duration, BMI, ...
static_cat  2          gender, therapy type (as category codes)
y           4          the true answers, mg/dL
y_mask      4          which of the four answers actually exist
cgm_now     1          the last real CGM value (used to convert predictions back)
```

Two details worth knowing:

**Time of day is encoded as sine and cosine** rather than as "minutes since
midnight", so that 23:55 and 00:05 are neighbours rather than maximally distant.

**Models predict the *change* from the current value**, not the absolute value.
And the final layer starts at zero, which means an untrained model predicts "no
change" — the persistence forecast. Every model therefore starts from a sensible
baseline and has to earn any improvement.

## 8.3 Step three: normalisation, done safely

Neural networks want inputs on a similar scale. CGM is ~40–400; a bolus is
0–15; time-since-carb is 0–1440. Left raw, the big numbers dominate.

`data/transforms.py` handles this, with two important properties:

1. **The statistics are computed on training patients only**, then frozen to a
   JSON file that ships with the run. If you computed the mean using the test
   set, you'd have leaked information about the test set into training.

2. **Different signal types get different treatment.** CGM gets a robust
   median/spread standardisation. But bolus is 93% exact zeros with a long tail —
   standardising that makes almost every sample identical. Those get a `log(1+x)`
   transform and a high-percentile scale instead.

## 8.4 Step four: the models

`metabonet_forecasting/models/`. The structure mirrors section 6 and 7:

```
encoders.py   the per-modality networks (CGM conv, driver MLPs, static embedding)
fusion.py     concat / gated / FiLM
frontend.py   glues encoders + fusion + backbone + head together
tcn.py patchtst.py mamba2.py recurrent.py    the backbones
tft.py tide.py                               the two native architectures
heads.py      the final layer, shared by everything
```

Everything obeys one interface: take the batch dictionary above, return four
numbers per sample. The training code literally cannot tell which architecture
it's holding — that's enforced by a registry that maps a config name to a class.

## 8.5 Step five: training

`metabonet_forecasting/training/`. One loop, used by every architecture. It:

- computes a **masked Huber loss** (Huber = like squared error but less
  sensitive to outliers; masked = skip horizons that have no true answer)
- uses mixed precision (fp16) for speed on the GPU
- decays the learning rate on a cosine schedule with warmup
- stops early if validation RMSE hasn't improved for 5 epochs
- saves the best checkpoint, the resolved config, the frozen normalisation, a
  provenance record (git commit, package versions, GPU, seed), and per-anchor
  predictions

That last one matters: because the predictions are saved, **scoring never
requires retraining**. Adding a new metric or a new clinical breakdown is a
30-second job over stored files.

## 8.6 Step six: evaluation

`metabonet_forecasting/evaluation/`.

Every run gets scored on:

- **4 patient strata**: everyone / novel patients / previously-seen patients /
  the leaderboard-comparable subset
- **13 clinical slices**: five glucose bands, four event contexts (after a meal,
  after a correction, both, neither), three trend states (rising/steady/falling)
- **4 horizons**
- and a pile of metrics including three different error grids

Statistics are done by **patient-level bootstrap**. Here's why that matters: we
have 380,000 test anchors, but they come from only 351 patients, and anchors from
the same patient are highly correlated. Treating them as 380,000 independent
samples would produce absurdly tight confidence intervals. Resampling *patients*
gives an honest picture.

For comparing two models, I resample the same patients for both and look at the
paired difference — which is far more sensitive, because it cancels out the huge
variation between patients.

## 8.7 Step seven: the tests

`tests/` — **85 automated tests** that run in 9 seconds without needing the real
data or a GPU. They build a small synthetic dataset with deliberately planted
problems (a short gap that must be filled, a long one that must not, a patient
stored backwards, a patient split across the train/test boundary) and check that
the pipeline handles each correctly.

The most important category tests **causality by perturbation**: change the
inputs *after* time t, and assert that the model's output *at* time t doesn't
move. If it moves, information is flowing backwards in time, and the test fails.
This is run against the convolution, both encoders, the TCN, the LSTM, Mamba-2,
and PatchTST's patching.

Also tested: the DTS error grid agrees with the organisers' own scorer on every
single point; the Mamba-2 SSD algorithm agrees with a slow reference
implementation; splits never share a patient; targets are never interpolated
values.

## 8.8 Step eight: the scripts

`scripts/` — one shell script per stage, so nothing has to be remembered:

```
00_setup_env.sh              check the environment
01_prepare_data.sh           build the sequence store (87 min, once)
02_smoke_test.sh             tiny end-to-end run of every architecture
03_run_tests.sh              the 85 tests
10_stageB_baselines.sh       simple baselines
20_stageC_backbones.sh       the five architectures
30_stageD_fusion_ablations.sh the fusion ladder
35_future_covariates.sh      the future-insulin study (added later)
40_stageE_modality_ablations.sh which signals matter
50_stageF_context_length.sh  how much history
60_stageG_generalization.sh  new patients, new studies
70_evaluate_all.sh           score everything
80_make_figures.sh           charts
90_make_submission.sh        competition files
99_run_everything.sh         all of it, in order
```

They share a GPU job queue that only starts a run on a card with enough free
memory — necessary because that machine is shared with about 48 other users.

---

# 9. The experiments and what they found

65 runs total. Here's each stage, what it asked, and what came back.

## Stage B — the baselines (what's the floor?)

Before judging fancy models you need to know what trivially simple methods
achieve.

| Method | What it does | RMSE 30 | RMSE 60 | RMSE 120 |
|---|---|---|---|---|
| Linear trend | extend the current slope | 32.01 | 60.34 | 98.95 |
| Persistence | predict "same as now" | 27.18 | 43.39 | 60.70 |
| Ridge | linear regression on lagged features | 22.74 | 37.52 | 50.01 |
| GBDT | gradient-boosted trees on the same features | 21.53 | 35.45 | 48.12 |
| **LSTM** | a basic 2-layer recurrent net | **20.41** | **33.62** | **45.80** |

Two things jump out.

**Linear extrapolation is worse than doing nothing.** Extending the current
slope two hours ahead is actively harmful — glucose curves bend.

**The plain LSTM reaches 33.62 at 60 minutes.** Keep that number. The best of the
57 sophisticated runs reaches 33.54. The entire architectural programme is worth
**0.08 mg/dL** over the simplest sequence model tried.

## Stage C — the five backbones

Everything held identical; only the backbone swapped.

| Backbone | RMSE 60 | DTS-A 60 |
|---|---|---|
| Mamba-2 | 33.59 | 69.75 |
| TCN | 33.67 | 69.53 |
| PatchTST | 33.74 | 69.46 |
| TFT | 34.20 | 69.18 |
| TiDE | 34.96 | 67.07 |

**The top three span 0.15 mg/dL.** For context, running the *same* configuration
with a different random seed moves it by 0.03. So the gap between Mamba-2 and
PatchTST is about five times the noise — technically detectable, practically
meaningless.

Only TiDE is clearly behind (+1.37). TiDE is the one with no temporal machinery
at all. So: *some* temporal structure helps; *which one* doesn't.

## Stage D — the fusion ladder (the project's central hypothesis)

This is the one the plan cared most about.

| Rung | RMSE 60 |
|---|---|
| 1. raw concatenation | 33.588 |
| 2. separate encoders + concat | 33.582 |
| 3. gated fusion | 33.667 |
| 4. **FiLM (the proposed model)** | **33.549** |
| FiLM without event features | 33.590 |
| FiLM without masks | 33.574 |
| FiLM with learned missing embedding | 33.625 |

FiLM beats raw concatenation by **0.039 mg/dL**.

The seed-to-seed standard deviation of the FiLM configuration alone, measured
over six seeds, is **0.029 mg/dL**.

In other words: **the effect is smaller than the noise.** Rerun with a different
seed and the ranking could flip. Gated fusion — a *more* structured method — is
the worst of the four.

I want to be direct about what this means, because it's the project's headline
hypothesis: **at this training budget, on this data, the structural argument is
not supported.** Modality-specific encoding doesn't measurably beat gluing the
numbers together, and CGM-centred conditioning doesn't measurably beat generic
fusion.

That is a legitimate scientific result — a well-designed experiment that returns
a null is informative — but it is not the result the plan was hoping for, and
the write-up needs to say so plainly with the noise floor stated.

## Stage E — which signals actually matter

| Inputs | RMSE 60 | Gain vs CGM only |
|---|---|---|
| CGM only | 34.99 | — |
| CGM + carbs | 34.56 | 0.43 |
| CGM + insulin | 33.63 | **1.36** |
| CGM + insulin + carbs | 33.55 | 1.44 |
| + heart rate & steps | 33.54 | 1.45 |

**Insulin is worth three times what carbohydrate is worth**, and adding carbs on
top of insulin buys almost nothing. Activity adds 0.01 — statistically nothing,
which is expected since it's >99% missing.

There's a hint of the later finding here: insulin is unusually informative. At
this point I read that as "insulin is physiologically important". Section 10
shows it's more complicated.

## Stage F — how much history

| Context | TCN | PatchTST | Mamba-2 |
|---|---|---|---|
| 2 hours | 34.52 | 34.57 | 34.54 |
| 4 hours | 34.32 | 34.42 | 34.29 |
| 8 hours | 34.20 | 34.29 | 34.17 |
| 12 hours | 34.10 | 34.17 | 34.06 |
| 24 hours | 33.67 | 33.74 | 33.57 |

More history helps, consistently, but modestly: **0.83–0.97 mg/dL** from 2 hours
to 24.

Crucially, it helps *all three architectures about equally*. The specific
hypothesis that Mamba-2's selective memory should pay off at long context is not
supported — at 24 hours its advantage over the TCN is 0.10 mg/dL.

This comparison is unusually clean, because the anchor set doesn't change with
context: every anchor is guaranteed 24 hours of clean history, so the 2-hour run
and the 24-hour run are scored on identical samples.

## Stage G — does it generalise?

**New patients vs. previously-seen patients:**

| | RMSE 60 |
|---|---|
| Novel patients (never in training) | 33.37 |
| Known patients (earlier data was in training) | 33.85 |

Patients the model has *never seen* are forecast slightly *better*. Whatever it
learned, it isn't memorising individuals. That's reassuring.

**Leave-one-study-out** (train without a study, test on it):

| Held out | RMSE 30 |
|---|---|
| OhioT1DM | 18.03 |
| AZT1D | 18.86 |
| CTR3 | 19.43 |
| BrisT1D | 23.21 |

⚠️ **These numbers are not comparable to any other number in this document.**
Each is scored on one study's patients only, not the 380,000-anchor official test
set. You raised exactly this point earlier and were right to.

What they *do* show: the same model and procedure varies by **5 mg/dL** depending
purely on which study you test it on. Cohort identity moves the score more than
every architectural choice in this study combined.

**The cohort shortcut check:** the plan warned that a model might learn "which
study is this?" instead of physiology. I tested this by explicitly giving a model
the study identity as an input. Score went from 33.588 to 33.560 — nothing. There
was no shortcut left to exploit.

---

# 10. The big discovery: future insulin

This part started with your question about a leaderboard entry scoring 92%+ in
DTS zone A at 30 minutes — a number that looked impossible next to our ~87%.

## The setup

Everything above is a **strict forecast**: the model sees only data from before
the anchor. That's what "causal" means in this repo.

But what if you let the model see insulin and carbohydrate values from *after*
the anchor?

There's a legitimate version of that question. If a person tells their pump "I'm
about to eat 60g and take 6 units", that's a genuinely *planned* action, known
at forecast time. Predicting glucose given a planned dose is a real and useful
task — it's what an artificial pancreas needs in order to choose a dose.

So I added a switch (`FUTURE=insulin|carbs|both`), off by default, that lets any
run read post-anchor insulin/carbs, and ran the same model four ways.

## The result

Same architecture, same seed, same split, same everything — one thing changed:

| What the model may see after "now" | RMSE 60 | DTS-A 30 | DTS-A 60 | DTS-A 120 |
|---|---|---|---|---|
| nothing (the benchmark) | 33.54 | 86.70 | 69.42 | 54.10 |
| carbohydrate only | 33.28 | 86.60 | 70.27 | 55.59 |
| **insulin only** | **24.76** | **92.18** | **82.88** | **72.09** |
| insulin + carbohydrate | 24.49 | 92.36 | 83.37 | 71.92 |

Read the third row again.

Adding future insulin moves 60-minute zone A from **69.4% to 82.9%**. RMSE drops
by 9 mg/dL. For comparison, *every architectural and representational choice in
the 57 causal runs combined* spans about 0.5 zone-A points.

And it reproduces the number you saw: **92.4% at 30 minutes**.

Future carbohydrate, meanwhile, buys 0.85 points — about what a modest genuine
improvement looks like.

## Why this is leakage, not skill

Now recall section 1: **85.8% of this data comes from people on AID systems**,
where an algorithm sets insulin automatically every five minutes, *as a function
of the CGM reading it just took*.

Written as a formula, in AID data:

```
    insulin(t+k)  ≈  g( CGM(t+k) )
```

So when you hand the model future insulin, you are handing it **a transformation
of the very thing it's supposed to predict**. It doesn't need to understand
physiology; it can partially invert the controller and read the answer.

That's a hypothesis. Here's the test: if this is the mechanism, the gain should
appear *only* in closed-loop cohorts and be absent where a human decides doses.

| Cohort | Insulin mode | RMSE 60 improvement from future insulin |
|---|---|---|
| IOBP2 | 100% AID | **+18.36** |
| DCLP5 | 100% AID | **+15.28** |
| DCLP3 | 100% AID | **+12.70** |
| Flair | 100% AID | **+12.66** |
| AZT1D | 100% AID | **+8.33** |
| Loop | 100% AID | **+7.17** |
| PEDAP | 100% AID | **+5.10** |
| CTR3 | 100% AID | +0.22 |
| ReplaceBG | 0% AID (pump) | +1.26 |
| HUPA-UCM | 0% AID (pump) | +0.35 |
| BrisT1D | 29% AID | −0.43 |
| T1D-UOM | 0% AID (injections) | −1.57 |

Seven of the eight closed-loop cohorts gain 5–18 mg/dL. **Every cohort without
closed-loop delivery gains essentially nothing, and two get worse.**

That's about as clean a mechanistic confirmation as observational data gives you.
(CTR3 is an honest exception — 100% AID but no gain. I've left it in the table
rather than hiding it.)

## Is using it cheating?

I checked what the competition organisers actually ship. In the secret holdout
file, at the exact target timestamps, across all 2,228 rows to be predicted:

| At anchor + | CGM available | CGM masked to −1 | Insulin available |
|---|---|---|---|
| +0 min | 100% | 0% | 100% |
| +30 min | 0% | **100%** | **100%** |
| +60 min | 0% | **100%** | **100%** |
| +90 min | 0% | **100%** | **100%** |
| +120 min | 0% | **100%** | **100%** |

They carefully masked the future **CGM** — the answers really are secret. They
left future **insulin** completely intact.

So: **using it is permitted by the competition's own data design.** It is not
against the rules. Anyone can do it, and evidently someone has.

But it means the leaderboard is scoring a different question from the one it
appears to score:

- *Forecasting*: "given what is known now, what will glucose be?"
- *Intervention-conditioned prediction*: "given that this dose is delivered, what
  will glucose be?"

Both are legitimate research questions. They are not the same question, and their
scores are not comparable. Only the first is a forecast.

## Why this matters even for the "legitimate" version

There's a subtler problem worth knowing about, even if you only want the
simulation task.

If you train on *recorded* doses, the model learns "insulin appeared → glucose
was high", because in the data insulin is a *response* to high glucose. Deploy
that as a controller and it will conclude that giving more insulin raises
glucose — the wrong sign. This is a classic confounding problem (the same
structure as "patients admitted to the ICU have worse outcomes, therefore ICUs
are harmful").

Doing the conditional task properly needs either data where doses were set
independently of the patient's state, or the causal-inference machinery built for
exactly this (marginal structural models, inverse-probability weighting) — which
is, incidentally, where your professor's survival-analysis comment points.

## The clinical finding that came out of the same analysis

While breaking results down by glucose level, something worse turned up. For the
best model, at 60 minutes:

| True glucose at t+60 | DTS zone A | n |
|---|---|---|
| Below 54 (severe hypo) | **2.5%** | 1,864 |
| 54–70 (hypo) | **9.4%** | 9,146 |
| 70–180 (in range) | 72.4% | 271,884 |
| 180–250 | 70.7% | 69,715 |
| Above 250 | 65.7% | 25,485 |

The model's aggregate score is 69.8% zone A. But in hypoglycaemia — the situation
that can kill someone this afternoon — it's **2.5%**.

The aggregate is carried entirely by the 72% of cases that are in range. No
overall metric would have revealed this; it only appeared because the plan
required clinical stratification.

In my view this is the most important practical finding in the whole project,
and the most promising direction for the next paper.

---

# 11. The competition submissions

Three files were produced, all under `artifacts/submission/`:

| File | Track | Rows | Model |
|---|---|---|---|
| `live_mamba2_film_futboth.parquet` | live leaderboard | 2,648,987 | future insulin+carbs |
| `annual_mamba2_film_futboth.parquet` | annual (secret) | 2,228 | future insulin+carbs |
| `annual_mamba2_rawconcat_causal.parquet` | annual (secret) | 2,228 | strictly causal |

All three pass the organisers' official format validator.

The live one can be scored locally because its answers ship with the toolkit:

| Horizon | RMSE | MARD | Zone A |
|---|---|---|---|
| 30 min | 14.63 | 7.28% | 93.2% |
| 60 min | 22.85 | 11.30% | 84.2% |
| 90 min | 27.79 | 13.98% | 78.0% |
| 120 min | 31.68 | 16.35% | 72.9% |
| **overall** | **25.06** | **12.23%** | **82.1%** |

Two bugs the submission code caught that are worth knowing about, because both
would have produced a valid-looking file with silently wrong numbers:

**1. Category codes don't transfer between files.** Each dataset build numbers
categories in the order it first sees them. The corpus has four gender values;
the holdout has two. So code `1` means different things in the two files, and an
embedding trained on one, indexed with the other's codes, is quietly wrong. Fixed
by re-mapping by *name* into the training vocabulary.

**2. Short history.** Competition rows aren't filtered for 24 hours of clean
history the way training anchors are. Those windows are now left-padded with an
explicit "not observed" marker rather than reaching back into the previous
patient's data.

---

# 12. What it all means

## Answering the plan's six questions

| Question | Answer |
|---|---|
| Does multimodal info beat CGM alone? | Yes, modestly — 1.44 mg/dL, almost all of it insulin |
| Does modality-specific encoding beat concatenation? | No measurable difference |
| Does CGM-centred conditioning (FiLM) beat generic fusion? | No — inside seed noise |
| Are gains consistent across backbones? | There are no gains to be consistent about |
| Does long history favour PatchTST or Mamba-2? | No — helps all three equally |
| Do conclusions survive new patients and dataset shift? | Yes |

## The honest summary

On 24 hours of CGM, insulin and carbohydrate, at this training budget:
**architecture doesn't matter and representation doesn't matter.** A two-layer
LSTM on concatenated inputs is within 0.1 mg/dL of everything else tried.

What *does* matter is the information set — which signals you use, over what
span, and above all **whether any of it comes from after the forecast time**.

## What I'd write the paper about

Not a new architecture — there's nothing to report there. The publishable
contributions are:

1. **A quantified leakage measurement** on a public benchmark, with the mechanism
   identified (closed-loop control), the magnitude measured (+13.5 zone-A
   points), and the confound isolated per cohort. This explains an entire public
   leaderboard.

2. **The hypoglycaemia failure** — 2.5% zone A below 54 mg/dL — with the
   stratified evaluation framework that surfaces it.

3. **A negative result, properly bounded**: four fusion strategies and five
   backbones, with the noise floor measured, showing that the representational
   sophistication the field assumes matters, doesn't — at least at this scale.

## What's not done

- **No published benchmark result was reproduced.** The plan's Milestone 1 asked
  for running an existing MetaboNet-Bench model end-to-end and matching its
  number. I verified our error grid against the organisers' scorer, which is
  weaker. This is the main outstanding gap.
- **One training budget, thin seeds.** A null result at 20 epochs isn't a null
  result at 200.
- **No age or device subgroup analysis** (per-study is implemented).

---

# 13. Where everything lives

```
glucose/
├── RUNBOOK.md              every command, costs, verification status
├── WALKTHROUGH.md          this document
├── data/                   the raw parquet files (not in git)
├── artifacts/
│   ├── store/main/         the 14 GB prepared dataset
│   └── submission/         the competition files
├── configs/                YAML settings for every experiment
│   ├── base.yaml           defaults everything inherits
│   ├── model/              one per architecture
│   ├── fusion/             one per fusion rung
│   └── experiment/         one per plan experiment (E1–E9)
├── metabonet_forecasting/  the code
│   ├── data/               prepare, store, windows, splits, transforms
│   ├── models/             encoders, fusion, the five backbones, head
│   ├── training/           the shared training loop
│   ├── evaluation/         metrics, clinical strata, bootstrap, plots
│   ├── baselines/          persistence, linear, ridge, GBDT
│   └── submission/         competition file generation
├── scripts/                the numbered shell scripts
├── tests/                  the 85 tests
├── runs/                   one directory per experiment (65 of them)
└── results/                aggregated tables across all runs
```

## To reproduce from scratch

```bash
scripts/00_setup_env.sh          # check the environment
scripts/01_prepare_data.sh       # 87 min, builds the 14 GB store
scripts/03_run_tests.sh          # 85 tests, 9 seconds
scripts/10_stageB_baselines.sh   # then each stage in turn
...
scripts/70_evaluate_all.sh --bootstrap 1000
```

## To look at a single result

Every run directory (`runs/e4_mamba2_film_s0/`) contains:

- `config.resolved.yaml` — the exact settings used
- `manifest.json` — git commit, versions, GPU, seed
- `train_log.csv` — loss and metrics per epoch
- `predictions_test.parquet` — the actual predictions, per anchor
- `metrics.json` / `metrics.csv` — every metric, stratum and slice

---

# 14. Glossary

| Term | Meaning |
|---|---|
| **AID** | Automated insulin delivery. A closed-loop system where an algorithm sets insulin every 5 min based on CGM. 85.8% of this corpus. |
| **Anchor** | A single forecast point — the "now" from which we predict. |
| **Backbone** | The part of the model that processes the time sequence. TCN / PatchTST / Mamba-2 / etc. |
| **Bolus** | A discrete insulin dose, usually for a meal or correction. |
| **Basal** | Continuous background insulin. |
| **CGM** | Continuous glucose monitor. Reports glucose every 5 minutes. |
| **Causal** (in this repo) | Uses only information from before the forecast time. The default and the benchmark. |
| **Context** | How much history the model sees. 288 steps = 24 hours. |
| **DTS error grid** | Clinical metric grading each prediction A–E by how much harm acting on it could cause. Zone A = harmless. |
| **Epoch** | One full pass over the training data. |
| **FiLM** | Feature-wise Linear Modulation. Insulin/carbs generate a scale and shift that modulate the CGM representation, rather than being concatenated to it. |
| **Fusion** | How the different input signals are combined into one vector per time step. |
| **Horizon** | How far ahead we predict: 30/60/90/120 minutes. |
| **Hypoglycaemia** | Low blood glucose (<70 mg/dL). Immediately dangerous. |
| **Hyperglycaemia** | High blood glucose (>180 mg/dL). Harmful over years. |
| **Leakage** | Future or answer information reaching the model's inputs. Makes a bad model look excellent. |
| **Loss** | A single number measuring how wrong the model's predictions are; training minimises it. |
| **Mamba-2** | A selective state-space model. Carries a memory through the sequence, deciding from the input itself what to keep and what to write. |
| **MARD** | Mean absolute relative difference — error as a % of the true value. Punishes errors at low glucose heavily. |
| **Mask** | A 0/1 flag saying whether a value was actually measured or is filled in. |
| **MDI** | Multiple daily injections — insulin by pen/syringe. |
| **PatchTST** | A Transformer that chops the sequence into patches and applies attention over those. |
| **Persistence** | The trivial forecast: "glucose will be what it is now". |
| **RMSE** | Root mean squared error, in mg/dL. Typical prediction error, with big errors weighted extra. |
| **SAP** | Sensor-augmented pump. Pump with pre-programmed rates; the human decides. |
| **Seed** | The random number controlling initialisation and shuffling; changing it shows how much of a result is luck. |
| **Segment** | A stretch of one patient's data with no gaps in the 5-minute grid. |
| **Stride** | Spacing between anchors. We use 6 steps = one sample per 30 min. |
| **TCN** | Temporal Convolutional Network. Stacked dilated convolutions with a fixed receptive field. |
| **TFT** | Temporal Fusion Transformer. Has built-in variable selection and gating. |
| **TiDE** | Time-series Dense Encoder. Flattens the window and uses plain dense layers. The "no temporal machinery" control. |
| **Target / label** | The true future CGM value we're trying to predict. |
| **Window** | The slice of history fed to the model, ending at the anchor. |
