"""Column decisions and row-level cleaning.

This module is the single source of truth for *what happens to every one of the 37
columns* in the raw corpus. `COLUMN_DECISIONS` below is machine-readable and is
dumped verbatim into the run manifest, so a result can always be traced back to the
data treatment that produced it.

Every decision cites the EDA section in `data.ipynb` that justifies it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CGM_VALID_LO, CGM_VALID_HI

# ---------------------------------------------------------------------------
# Verdicts:
#   key       -- identity / bookkeeping, needed but never a feature
#   target    -- the thing being predicted
#   feature   -- fed to the model (possibly transformed)
#   derived   -- not fed raw; used only to build other features
#   drop      -- excluded, with a reason
# ---------------------------------------------------------------------------
COLUMN_DECISIONS: dict[str, dict] = {
    # ---- keys and bookkeeping ------------------------------------------------
    "id": dict(verdict="key", reason="Subject id. Not unique alone: 616 ids recur "
               "across cohorts (EDA 8.1). Always paired with source_file."),
    "source_file": dict(verdict="key+feature", reason="Origin study. Also used as a "
               "categorical feature when FeatureConfig.use_cohort_feature is on. "
               "Available at inference, but EDA 3.3 shows it is nearly determined "
               "by the null pattern, so it invites a cohort shortcut."),
    "date": dict(verdict="derived", reason="Timestamp. Shifted per study for "
               "de-identification and the max is in the future (EDA 7.2), so absolute "
               "calendar time is meaningless. Used only for time-of-day, day-of-week, "
               "lattice validation and elapsed-time-in-segment."),
    "is_test": dict(verdict="key", reason="Official split flag. Defines the split; "
               "never a feature."),
    "subject_split_across_traintest": dict(verdict="drop", reason="Metadata about the "
               "split itself. Feeding it leaks the evaluation design."),

    # ---- target --------------------------------------------------------------
    "CGM": dict(verdict="target+feature", reason="Interstitial glucose, mg/dL. The "
               "target at t+h and the dominant history channel. 15.48% null (EDA 3.1); "
               "interval-censored at 40 and 400 inconsistently by cohort (EDA 9.2)."),

    # ---- insulin / carbs -----------------------------------------------------
    "basal": dict(verdict="feature", reason="Units delivered in the 5-min slot (not a "
               "rate). 25% exact zeros. Integrated over trailing windows."),
    "bolus": dict(verdict="feature", reason="Units. 93% exact zeros -- a marked point "
               "process (EDA 5.4). Used via window sums + IOB decay, not raw."),
    "insulin": dict(verdict="drop", reason="EXACTLY equals basal + bolus: max residual "
               "3.6e-15 over 134,035,414 rows (EDA 10.1). Perfectly redundant; keeping "
               "it would triple-count the same signal."),
    "carbs": dict(verdict="feature", reason="Grams. 98% exact zeros. Used via window "
               "sums + COB decay. Absent entirely for Flair (EDA 3.3)."),

    # ---- static demographics -------------------------------------------------
    "age": dict(verdict="feature", reason="Years, subject-constant. Kept as a static "
               "covariate (one value per subject, not per row)."),
    "age_of_diagnosis": dict(verdict="feature", reason="T1D onset age. Also used to "
               "derive diabetes_duration = age - age_of_diagnosis."),
    "height": dict(verdict="feature", reason="In FEET (median 5.45, EDA 5.7). Converted "
               "to metres."),
    "weight": dict(verdict="feature", reason="In POUNDS (median 150, EDA 5.7). Converted "
               "to kilograms."),
    "gender": dict(verdict="feature", reason="4 stored levels collapse to 3: "
               "'Non-binary' and 'Non Binary' are the same level spelled twice "
               "(EDA 6.3)."),
    "ethnicity": dict(verdict="drop", reason="30 comma-joined free-text combinations, "
               "not a categorical (EDA 6.4). Strongly cohort-correlated. Excluded to "
               "avoid encoding study identity; also a fairness hazard as a raw feature."),
    "is_pregnant": dict(verdict="drop", reason="33% null and True for only 10 subjects "
               "(EDA verify). Too sparse to learn from; a near-perfect indicator of one "
               "cohort."),

    # ---- device / therapy context -------------------------------------------
    "insulin_delivery_modality": dict(verdict="feature", reason="AID / SAP / MDI, only "
               "3 levels, 0.87% null, clinically meaningful (closed-loop vs pump vs "
               "injections). Kept."),
    "cgm_device": dict(verdict="drop", reason="Sensor model. An era variable that "
               "changes mid-subject (EDA 8.5) and is ~90% co-missing with cohort "
               "identity (EDA 3.4). Mostly a cohort proxy."),
    "insulin_delivery_device": dict(verdict="drop", reason="Pump model. Same reasoning "
               "as cgm_device."),
    "insulin_delivery_algorithm": dict(verdict="drop", reason="Control-IQ / Loop / 670G "
               "etc. Almost collinear with modality + cohort."),
    "insulin_type_basal": dict(verdict="drop", reason="20 unnormalised levels that "
               "collapse to far fewer ('Humalog' / 'humalog' / 'Humalog (Lispro)'). "
               "Nearly all rapid-acting analogues with equivalent PK; adds cohort "
               "identity, not pharmacology."),
    "insulin_type_bolus": dict(verdict="drop", reason="Same as insulin_type_basal (17 "
               "levels, same collapse)."),

    # ---- trial bookkeeping ---------------------------------------------------
    "treatment_group": dict(verdict="drop", reason="Trial arm, 64.7% null -- present "
               "only for the randomised studies. Pure study identity (EDA 3.3)."),
    "randomization_date": dict(verdict="drop", reason="77.9% null, trial bookkeeping."),
    "extension_date": dict(verdict="drop", reason="94.2% null, trial bookkeeping."),

    # ---- wearable / environment channels ------------------------------------
    "heartrate": dict(verdict="drop", reason="99.33% null -- exists only in the small "
               "wearable cohorts (EDA 3.3). Corpus-wide it is unusable and its "
               "presence alone identifies the study."),
    "steps": dict(verdict="drop", reason="99.38% null. Same as heartrate."),
    "calories_burned": dict(verdict="drop", reason="76.4% null, and 7,127 slots exceed "
               "500 kcal per 5 minutes with a max of 21,726 -- physiologically "
               "impossible (EDA 12)."),
    "skin_temp": dict(verdict="drop", reason="99.91% null; 1,071 values are a literal 0, "
               "which is sensor-off encoded as data (EDA 12)."),
    "galvanic_skin_response": dict(verdict="drop", reason="99.91% null; co-missing with "
               "skin_temp at phi = 0.999 (EDA 3.4)."),
    "air_temp": dict(verdict="drop", reason="99.94% null, one cohort, and in Fahrenheit."),

    # ---- activity / free text ------------------------------------------------
    "workout_duration": dict(verdict="drop", reason="99.94% null; 11 rows exceed a day, "
               "max 8,218 min = 5.7 days (EDA 12)."),
    "workout_intensity": dict(verdict="drop", reason="219 non-null values in 154,842,077 "
               "rows (EDA 5.1). Statistically nonexistent."),
    "workout_label": dict(verdict="drop", reason="99.93% null, 80 free-text levels."),
    "meal_label": dict(verdict="drop", reason="99.94% null, 2,385 free-text levels. The "
               "quantitative `carbs` column covers the same events."),
    "misc_notes": dict(verdict="drop", reason="99.97% null free text."),
}

# Columns actually read from the parquet. Reading only these cuts the load from
# ~61 GB / 30 s to ~11 GB / 10 s.
LOAD_COLUMNS = [
    "source_file", "id", "date", "is_test", "CGM",
    "basal", "bolus", "carbs",
    "age", "age_of_diagnosis", "height", "weight", "gender",
    "insulin_delivery_modality",
]

STATIC_NUMERIC = ["age", "age_of_diagnosis", "height_m", "weight_kg", "bmi",
                  "diabetes_duration"]
STATIC_CATEGORICAL = ["gender", "insulin_delivery_modality"]   # + source_file, optional

# Unit conversions (EDA 5.7 -- the corpus mixes imperial and metric silently).
FEET_TO_M = 0.3048
LB_TO_KG = 0.45359237


def dropped_columns() -> list[str]:
    return [c for c, d in COLUMN_DECISIONS.items() if d["verdict"] == "drop"]


def kept_columns() -> list[str]:
    return [c for c, d in COLUMN_DECISIONS.items() if "feature" in d["verdict"]]


# ---------------------------------------------------------------------------
def normalise_gender(s: pd.Series) -> pd.Series:
    """'Non-binary' and 'Non Binary' are one level (EDA 6.3)."""
    out = s.astype("string").str.strip()
    out = out.str.replace(r"^Non[\s-]?[Bb]inary$", "Non-binary", regex=True)
    return out


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Apply every row-level cleaning rule. Mutates and returns `df`.

    Each step is logged in DATA_PIPELINE.md. Nothing here imputes: imputation is a
    modelling choice and lives in the model adapters, not in the data.
    """
    log: dict[str, int] = {}

    # 1. CGM sensor faults -> NaN. Keeps the 40/400 clamp values (they are real
    #    readings the device chose to report) but discards impossible ones.
    bad = (df["CGM"] < CGM_VALID_LO) | (df["CGM"] > CGM_VALID_HI)
    log["cgm_out_of_range_to_nan"] = int(bad.sum())
    df.loc[bad, "CGM"] = np.nan

    # 2. Negative bolus (4 rows, values -3 and -1; EDA 12). A negative dose is not
    #    physical -- these are correction/annotation artefacts.
    neg = df["bolus"] < 0
    log["negative_bolus_clipped"] = int(neg.sum())
    df.loc[neg, "bolus"] = 0.0

    # 3. Negative basal / carbs, defensively (none observed).
    for c in ("basal", "carbs"):
        n = int((df[c] < 0).sum())
        if n:
            df.loc[df[c] < 0, c] = 0.0
        log[f"negative_{c}_clipped"] = n

    # 4. Unit conversion (EDA 5.7).
    df["height_m"] = df["height"].astype("float32") * FEET_TO_M
    df["weight_kg"] = df["weight"].astype("float32") * LB_TO_KG
    df["bmi"] = (df["weight_kg"] / df["height_m"] ** 2).astype("float32")
    # Implausible BMI -> NaN rather than a fabricated value.
    bmi_bad = (df["bmi"] < 10) | (df["bmi"] > 60)
    log["bmi_implausible_to_nan"] = int(bmi_bad.sum())
    df.loc[bmi_bad, "bmi"] = np.nan
    df.drop(columns=["height", "weight"], inplace=True)

    # 5. Derived static covariate.
    df["diabetes_duration"] = (df["age"] - df["age_of_diagnosis"]).astype("float32")
    dd_bad = df["diabetes_duration"] < 0
    log["negative_duration_to_nan"] = int(dd_bad.sum())
    df.loc[dd_bad, "diabetes_duration"] = np.nan

    # 6. Categorical hygiene (EDA 6.3).
    df["gender"] = normalise_gender(df["gender"])
    df["insulin_delivery_modality"] = (df["insulin_delivery_modality"]
                                       .astype("string").str.strip())

    # 7. Dose/carb NaN semantics. A NaN in an event channel means "not recorded",
    #    and for a *sum over a window* the correct neutral element is 0. We keep the
    #    raw NaN for the presence features and fill only inside the accumulators
    #    (done in features.py), so this is recorded but not applied here.
    log["bolus_null"] = int(df["bolus"].isna().sum())
    log["basal_null"] = int(df["basal"].isna().sum())
    log["carbs_null"] = int(df["carbs"].isna().sum())

    df.attrs["clean_log"] = log
    return df
