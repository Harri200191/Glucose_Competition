"""Central configuration for the MetaboNet glucose-prediction experiments.

Every number here is either a competition constant or a decision justified by the
EDA in ``data.ipynb``. Where it is a decision, the docstring says which EDA finding
drove it. See DATA_PIPELINE.md for the full narrative.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
BUILD_DIR = ROOT / "build"        # engineered feature matrices land here
RUNS_DIR = ROOT / "runs"          # model artefacts + results land here

CORPUS_PARQUET = DATA_DIR / "metabonet_public.parquet"
HOLDOUT_PARQUET = DATA_DIR / "annual_competition_secret_holdout_set_public.parquet"

# ----------------------------------------------------------------- competition
GRID_MIN = 5                       # EDA 7.1: the corpus is a dense 5-minute lattice
HORIZONS = (30, 60, 90, 120)       # minutes ahead to forecast
HORIZON_STEPS = tuple(h // GRID_MIN for h in HORIZONS)   # (6, 12, 18, 24)

SUBJECT_KEY = ("source_file", "id")   # EDA 8.1: `id` alone collides across cohorts

# Cohorts the live leaderboard excludes (EDA 13). DCLP3/DCLP5/Flair have no carb
# data; T1D-UOM is MDI-only.
LEADERBOARD_EXCLUDED_COHORTS = ("DCLP3", "DCLP5", "Flair", "T1D-UOM")

# Sensor reporting limits. EDA 9.2: the target is interval-censored at both ends,
# inconsistently across cohorts.
CGM_CLAMP_LO = 40.0
CGM_CLAMP_HI = 400.0
# Values outside this are treated as sensor faults, not readings.
CGM_VALID_LO = 10.0
CGM_VALID_HI = 600.0


@dataclass
class SegmentConfig:
    """How a subject's record is cut into contiguous, modellable stretches.

    EDA 7.1 found two ShanghaiT1DM subjects stored as two concatenated wear
    periods, one of them *out of chronological order*. EDA 7.4 found 1.15 M
    CGM gaps whose lengths span 5 minutes to 636 days. Both mean a subject is
    not one time series.
    """
    # Cut wherever consecutive rows are not exactly GRID_MIN apart.
    enforce_lattice: bool = True
    # Cut wherever CGM has been absent for longer than this (minutes).
    max_gap_min: int = 30
    # Gaps at or below this are linearly interpolated for FEATURE purposes only.
    interp_gap_min: int = 30
    # Drop segments shorter than this (minutes) -- too short to carry 24h history.
    min_segment_min: int = 6 * 60


@dataclass
class FeatureConfig:
    """Windowed feature construction. EDA 9.6/10.2 justify the window choices."""
    # CGM lags in lattice steps. Dense over the last hour, then out to 24 h.
    cgm_lags: tuple = (1, 2, 3, 4, 5, 6, 8, 10, 12, 18, 24, 36, 48, 72, 96, 144, 288)
    # Differences over k steps -> also converted to per-minute rates.
    cgm_diffs: tuple = (1, 2, 3, 6, 12, 24)
    # Rolling window lengths in steps: 30 m, 1 h, 2 h, 4 h, 6 h, 24 h.
    windows: tuple = (6, 12, 24, 48, 72, 288)
    # Exponentially weighted CGM means, half-life in steps.
    ewm_halflives: tuple = (3, 6, 12)
    # Dose/carb channels integrated over `windows`.
    dose_channels: tuple = ("bolus", "basal", "carbs")
    # Exponential decay proxies. Half-lives in minutes (rapid-acting analogue
    # insulin ~55 min; mixed-meal carb absorption ~45 min).
    iob_halflife_min: float = 55.0
    cob_halflife_min: float = 45.0
    # History required before a row can be an anchor (minutes). Matches the
    # leaderboard filter (EDA 13).
    required_history_min: int = 24 * 60
    # Include `source_file` as a categorical feature. It IS available at
    # inference time, but EDA 3.3 showed cohort is nearly determined by the null
    # pattern, so it invites a shortcut. Report both settings.
    use_cohort_feature: bool = True
    # Include counts of missing CGM in the window. Legitimate sensor-health
    # signal, but partially encodes cohort. Kept, and flagged in the doc.
    use_missingness_features: bool = True


@dataclass
class SampleConfig:
    """Which lattice rows become training/eval samples."""
    # Take one anchor every `stride` lattice steps. 6 = one sample per 30 min.
    # EDA 9.6: r(CGM_t, CGM_t+5min) ~ 0.99, so consecutive anchors are
    # near-duplicates; striding cuts cost with almost no information loss.
    train_stride: int = 6
    eval_stride: int = 6
    # Hard cap on training rows (per horizon). None = no cap.
    max_train_samples: int | None = None
    # Require the anchor row itself to hold a real (non-interpolated) CGM value.
    require_anchor_cgm: bool = True
    # Require the target at t+h to be a real reading. Always True -- an
    # interpolated target is a fabricated label.
    require_real_target: bool = True


@dataclass
class SplitConfig:
    """EDA 11: split by subject, never by row; 166-169 subjects sit on both sides."""
    scheme: str = "official"        # official | subject | cohort
    val_frac: float = 0.12          # subject-level holdout carved from train
    n_folds: int = 5                # for scheme="subject"
    holdout_cohort: str | None = None   # for scheme="cohort"
    seed: int = 0


@dataclass
class ExperimentConfig:
    segment: SegmentConfig = field(default_factory=SegmentConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    sample: SampleConfig = field(default_factory=SampleConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    seed: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


DEFAULT = ExperimentConfig()
