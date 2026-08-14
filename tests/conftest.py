"""Shared fixtures: a synthetic corpus and a real store built from it.

The tests do not touch ``data/metabonet_public.parquet``. Instead they generate a
small corpus with the *same schema* and known, planted structure -- a short CGM
gap that should be interpolated, a long one that should not, a subject stored out
of chronological order, a subject split temporally across the train/test boundary
-- and run the real ``prepare.build_store`` over it. Every property the pipeline
claims is then checked against ground truth that the fixture knows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from metabonet_forecasting.data import constants as K
from metabonet_forecasting.data.prepare import PrepareConfig, build_store
from metabonet_forecasting.data.store import SequenceStore

N_STEPS = 720                      # 60 h per subject at 5 minutes
SHORT_GAP = (200, 204)             # 4 steps = 20 min  -> interpolated
LONG_GAP = (400, 420)              # 20 steps = 100 min -> left missing
COHORTS = ["StudyA", "StudyB", "StudyC"]


def _subject_frame(cohort: str, pid: str, start_min: int, is_test_flag,
                   known: bool, rng: np.random.Generator,
                   reverse_halves: bool = False) -> pd.DataFrame:
    t = np.arange(N_STEPS, dtype=np.int64) * K.GRID_MIN + start_min
    phase = rng.uniform(0, 2 * np.pi)
    cgm = 140 + 45 * np.sin(2 * np.pi * np.arange(N_STEPS) / 288.0 + phase) \
        + rng.normal(0, 4, N_STEPS)
    cgm[SHORT_GAP[0]:SHORT_GAP[1]] = np.nan
    cgm[LONG_GAP[0]:LONG_GAP[1]] = np.nan

    bolus = np.zeros(N_STEPS)
    bolus[rng.choice(N_STEPS, size=8, replace=False)] = rng.uniform(1, 9, 8)
    carbs = np.zeros(N_STEPS)
    carbs[rng.choice(N_STEPS, size=6, replace=False)] = rng.uniform(15, 80, 6)
    basal = np.full(N_STEPS, 0.05)

    df = pd.DataFrame({
        "source_file": cohort, "id": pid,
        "date": pd.to_datetime(t * 60, unit="s"),
        "CGM": cgm, "basal": basal, "bolus": bolus, "carbs": carbs,
        "heartrate": np.where(rng.random(N_STEPS) < 0.2, rng.uniform(55, 130, N_STEPS),
                              np.nan),
        "steps": np.where(rng.random(N_STEPS) < 0.2, rng.integers(0, 200, N_STEPS), np.nan),
        "is_test": is_test_flag,
        "subject_split_across_traintest": known,
        "age": 34.0, "age_of_diagnosis": 12.0, "height": 5.6, "weight": 165.0,
        "gender": "Female" if int(pid) % 2 else "Male",
        "insulin_delivery_modality": "AID" if cohort == "StudyA" else "SAP",
    })
    if reverse_halves:
        # The ShanghaiT1DM pathology: two wear periods concatenated, the later one
        # stored first. A naive shift() would compute a delta running backwards.
        half = N_STEPS // 2
        df = pd.concat([df.iloc[half:], df.iloc[:half]], ignore_index=True)
        df.loc[half:, "date"] = df.loc[half:, "date"] - pd.Timedelta(days=40)
    return df


@pytest.fixture(scope="session")
def synthetic_corpus(tmp_path_factory) -> str:
    rng = np.random.default_rng(0)
    frames = []
    pid = 0
    for cohort in COHORTS:
        for k in range(4):
            pid += 1
            if k == 0:                       # never seen in training: novel patient
                is_test = np.ones(N_STEPS, dtype=bool)
                known = False
            elif k == 1:                     # split temporally: known patient
                is_test = np.arange(N_STEPS) >= (N_STEPS * 2 // 3)
                known = True
            else:                            # training only
                is_test = np.zeros(N_STEPS, dtype=bool)
                known = False
            frames.append(_subject_frame(cohort, str(pid), 3000 * pid, is_test, known,
                                         rng, reverse_halves=(pid == 3)))
    corpus = pd.concat(frames, ignore_index=True)
    path = tmp_path_factory.mktemp("corpus") / "synthetic.parquet"
    pq.write_table(pa.Table.from_pandas(corpus, preserve_index=False), path)
    return str(path)


@pytest.fixture(scope="session")
def store_path(synthetic_corpus, tmp_path_factory) -> str:
    out = tmp_path_factory.mktemp("store") / "test_store"
    build_store(PrepareConfig(
        corpus=synthetic_corpus, out=str(out),
        min_segment_min=60 * 4, min_history_steps=48, anchor_stride=2,
    ))
    return str(out)


@pytest.fixture(scope="session")
def store(store_path) -> SequenceStore:
    return SequenceStore(store_path)


@pytest.fixture()
def base_config(store_path):
    """A tiny end-to-end config pointed at the synthetic store."""
    from metabonet_forecasting.common.config import load_config
    return load_config("base", [
        f"data.store={store_path}",
        "data.context=48",
        "data.horizons_min=[30, 60]",
        "data.sampling.max_train_windows=600",
        "data.sampling.max_val_windows=200",
        "data.sampling.max_test_windows=200",
        "data.sampling.standardizer_sample=600",
        "data.loader.batch_size=16",
        "data.loader.num_workers=0",
        "training.epochs=1",
        "training.max_steps_per_epoch=3",
        "training.device=cpu",
        "training.amp=false",
        "training.warmup_steps=1",
    ])
