"""Corpus facts and the channel contract.

Everything here is either a property of ``data/metabonet_public.parquet`` or a
decision this project makes once, in one place, so that every model sees the same
thing. The plan's stage A ("freeze the dataset version, preprocessing version,
patient IDs/splits, target horizons") is implemented as: change something here and
the store's ``schema_version`` changes, which invalidates old stores.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
ARTIFACT_DIR = REPO_ROOT / "artifacts"
CORPUS_PARQUET = DATA_DIR / "metabonet_public.parquet"

SCHEMA_VERSION = "seqstore-1"

# ------------------------------------------------------------------ time grid
GRID_MIN = 5                                  # the corpus is a dense 5-minute lattice
STEPS_PER_HOUR = 60 // GRID_MIN

# Primary horizons for the architecture paper (plan 3.2). 90 min is carried as a
# secondary horizon because the public benchmark scores it, and dropping it would
# make results non-comparable with MetaboNet-Bench.
PRIMARY_HORIZONS_MIN: Tuple[int, ...] = (30, 60, 120)
ALL_HORIZONS_MIN: Tuple[int, ...] = (30, 60, 90, 120)
HORIZON_STEPS: Tuple[int, ...] = tuple(h // GRID_MIN for h in ALL_HORIZONS_MIN)

# Context grid for the length study (plan 3.3): 2/4/8/12/24 h.
CONTEXT_GRID_STEPS: Tuple[int, ...] = (24, 48, 96, 144, 288)
MAX_CONTEXT_STEPS = max(CONTEXT_GRID_STEPS)

# ------------------------------------------------------------------- identity
# `id` is reused across studies, so the subject key is the pair.
SUBJECT_KEY = ("source_file", "id")

# ---------------------------------------------------------------- sensor range
# Outside this a CGM value is a sensor fault, not a reading, and becomes NaN.
CGM_VALID_LO, CGM_VALID_HI = 10.0, 600.0
# Devices report clamped at these; those are real reported readings and are kept.
CGM_REPORT_LO, CGM_REPORT_HI = 40.0, 400.0

# Clinical strata used throughout evaluation (plan 8.2).
HYPO_L2, HYPO_L1, RANGE_HI, HYPER_L2 = 54.0, 70.0, 180.0, 250.0

# ----------------------------------------------------------- columns we read
# Reading 17 of 37 columns keeps the scan at a few GB instead of ~60 GB.
LOAD_COLUMNS: List[str] = [
    "source_file", "id", "date", "is_test", "subject_split_across_traintest",
    "CGM", "basal", "bolus", "carbs",
    "heartrate", "steps",
    "age", "age_of_diagnosis", "height", "weight", "gender",
    "insulin_delivery_modality",
]

# Why each of the other 20 is absent. Dumped into the store manifest so a reader
# never has to guess whether a column was forgotten or excluded.
EXCLUDED_COLUMNS: Dict[str, str] = {
    "insulin": "exactly basal + bolus (max residual 3.6e-15 over 134M rows); "
               "keeping it triple-counts one signal",
    "ethnicity": "30 comma-joined free-text combinations, strongly cohort-correlated, "
                 "and a fairness hazard as a raw model input",
    "is_pregnant": "33% null, True for 10 subjects; a near-perfect single-cohort flag",
    "cgm_device": "sensor model; an era variable that is mostly a cohort proxy",
    "insulin_delivery_device": "pump model; same reasoning as cgm_device",
    "insulin_delivery_algorithm": "near-collinear with modality plus cohort",
    "insulin_type_basal": "20 unnormalised levels of equivalent-PK analogues; adds "
                          "study identity, not pharmacology",
    "insulin_type_bolus": "17 levels, same collapse as insulin_type_basal",
    "treatment_group": "trial arm; exists only for randomised studies, so it is the "
                       "study label",
    "randomization_date": "trial bookkeeping, 77.9% null",
    "extension_date": "trial bookkeeping, 94.2% null",
    "calories_burned": "76.4% null and physically impossible values (21,726 kcal/5 min)",
    "skin_temp": "99.91% null; 1,071 literal zeros are sensor-off stored as data",
    "galvanic_skin_response": "99.91% null, co-missing with skin_temp at phi=0.999",
    "air_temp": "99.94% null, one cohort, Fahrenheit",
    "workout_duration": "99.94% null; 11 rows exceed a day",
    "workout_intensity": "219 non-null values in 154.8M rows",
    "workout_label": "99.93% null free text",
    "meal_label": "99.94% null free text; carbs already quantifies the event",
    "misc_notes": "99.97% null free text",
}

# Cohorts the public leaderboard excludes: no carbohydrate data (DCLP3, DCLP5,
# Flair) or MDI-only (T1D-UOM). Used to report the benchmark-comparable slice.
LEADERBOARD_EXCLUDED_COHORTS: Tuple[str, ...] = ("DCLP3", "DCLP5", "Flair", "T1D-UOM")

# ------------------------------------------------------------- channel layout
# The sequence store is a dense [n_rows, n_channels] float32 memmap. This list is
# the contract between prepare.py and everything downstream; the store manifest
# records it so a stale store cannot be silently mixed with new code.
#
# Semantics, per the plan's modality roles (section 3.1):
#   cgm_*        the physiological state being forecast
#   ins_*/carb_* sparse interventions and their causal exposure summaries
#   act_*        contextual modifiers, present in a minority of cohorts
#   *_mask       per-variable observed/missing flag, always paired with its value
RAW_CHANNELS: List[str] = [
    "cgm",            # mg/dL, short-gap interpolated, NaN where still unknown
    "cgm_mask",       # 1 where the reading is real (not interpolated, not missing)
    "ins_basal",      # units delivered in the 5-min slot
    "ins_basal_mask",
    "ins_bolus",      # units
    "ins_bolus_mask",
    "carb_grams",     # grams
    "carb_mask",
    "act_heartrate",  # bpm
    "act_heartrate_mask",
    "act_steps",      # steps in the slot
    "act_steps_mask",
]

# Exponential exposure time constants (minutes). Insulin spans ~30 min to 4 h per
# the plan; carbohydrate uses shorter constants. These are inductive biases to be
# ablated, NOT a validated pharmacokinetic model.
BOLUS_TAUS_MIN: Tuple[int, ...] = (30, 60, 120, 240)
BASAL_TAUS_MIN: Tuple[int, ...] = (60, 240)
CARB_TAUS_MIN: Tuple[int, ...] = (20, 45, 90)

DERIVED_CHANNELS: List[str] = (
    [f"ins_bolus_exp{t}" for t in BOLUS_TAUS_MIN]
    + [f"ins_basal_exp{t}" for t in BASAL_TAUS_MIN]
    + [f"carb_exp{t}" for t in CARB_TAUS_MIN]
    + ["ins_bolus_since", "carb_since"]      # minutes since last event, capped
)

CHANNELS: List[str] = RAW_CHANNELS + DERIVED_CHANNELS
CHANNEL_INDEX: Dict[str, int] = {name: i for i, name in enumerate(CHANNELS)}

# Minutes-since-event is capped so "no event in living memory" is a finite number.
SINCE_CAP_MIN = 24 * 60

# --------------------------------------------------------------- modality map
# Which channels each modality contributes. Modality ablations (plan 7.5) drop
# whole groups by name, so this mapping is the only place the grouping is defined.
MODALITY_CHANNELS: Dict[str, List[str]] = {
    "cgm": ["cgm", "cgm_mask"],
    "insulin": (
        ["ins_basal", "ins_basal_mask", "ins_bolus", "ins_bolus_mask"]
        + [f"ins_bolus_exp{t}" for t in BOLUS_TAUS_MIN]
        + [f"ins_basal_exp{t}" for t in BASAL_TAUS_MIN]
        + ["ins_bolus_since"]
    ),
    "carbs": (
        ["carb_grams", "carb_mask"]
        + [f"carb_exp{t}" for t in CARB_TAUS_MIN]
        + ["carb_since"]
    ),
    "activity": ["act_heartrate", "act_heartrate_mask", "act_steps", "act_steps_mask"],
}

# Channels that exist only because of stage-2 event/exposure feature engineering.
# Turning them off is ablation D10 in the plan.
EVENT_FEATURE_CHANNELS: List[str] = DERIVED_CHANNELS
# Channels that exist only to describe missingness. Ablation D11.
MASK_CHANNELS: List[str] = [c for c in CHANNELS if c.endswith("_mask")]

# ------------------------------------------------------------------- statics
STATIC_NUMERIC: List[str] = ["age", "age_of_diagnosis", "diabetes_duration",
                             "height_m", "weight_kg", "bmi"]
# `source_file` is deliberately last: it is excluded from the primary model
# (plan 4.5) and only enabled for the cohort-shortcut sensitivity analysis.
STATIC_CATEGORICAL: List[str] = ["gender", "insulin_delivery_modality", "source_file"]
STATIC_CATEGORICAL_PRIMARY: List[str] = ["gender", "insulin_delivery_modality"]

FEET_TO_M = 0.3048
LB_TO_KG = 0.45359237
BMI_VALID = (10.0, 60.0)
