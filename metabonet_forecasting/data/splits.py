"""Train / validation / test partitioning (plan 3.4, 7.7).

Three facts about this corpus force the design:

* consecutive samples are almost the same sample -- r(CGM_t, CGM_{t+5min}) ~ 0.99 --
  so a random split over anchors is near-total leakage. Split by **subject**;
* 166-169 subjects appear on *both* sides of the official ``is_test`` flag, split
  temporally (their earlier window is train, their later window is test). That is
  the benchmark's own design and gives the **known-patient** evaluation stratum;
  test subjects that never appear in train give the **novel-patient** stratum;
* missingness patterns nearly determine the source study, so subject-level
  splitting still rewards cohort shortcuts. Leave-one-source-out is the honest
  transportability check.

Every function returns boolean masks over the store's anchor table, never a copy
of the data.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import constants as K
from .store import SequenceStore


class SplitError(RuntimeError):
    """Raised when an invariant that would invalidate a result is violated."""


def _subject_holdout(subjects: np.ndarray, pool: np.ndarray, frac: float,
                     seed: int) -> np.ndarray:
    """Carve a subject-disjoint slice out of ``pool``."""
    available = np.unique(subjects[pool])
    if available.size == 0:
        raise SplitError("validation pool is empty")
    rng = np.random.default_rng(seed)
    n_val = max(1, int(round(available.size * frac)))
    chosen = rng.choice(available, size=min(n_val, available.size), replace=False)
    return pool & np.isin(subjects, chosen)


def official_split(store: SequenceStore, val_frac: float = 0.12,
                   seed: int = 0) -> Dict[str, np.ndarray]:
    """The benchmark's own patient-level split, plus a subject-disjoint val slice.

    Subjects that appear on both sides of ``is_test`` are excluded from the
    validation pool: their train rows are legitimate training data, but early
    stopping on them would tune the stopping point using people who also appear
    in test.
    """
    a = store.anchors
    subjects = a["subject"].to_numpy()
    is_test = a["is_test"].to_numpy().astype(bool)
    train_all = ~is_test

    both = np.unique(subjects[is_test])
    val_pool = train_all & ~np.isin(subjects, both)
    val = _subject_holdout(subjects, val_pool, val_frac, seed)
    return {"train": train_all & ~val, "val": val, "test": is_test}


def subject_kfold(store: SequenceStore, n_folds: int = 5, fold: int = 0,
                  val_frac: float = 0.12, seed: int = 0) -> Dict[str, np.ndarray]:
    """Group K-fold over subjects, drawn only from the official train portion.

    Used for architecture search, so the official test split stays untouched
    until the primary comparison is frozen (plan 9.3).
    """
    a = store.anchors
    subjects = a["subject"].to_numpy()
    pool = ~a["is_test"].to_numpy().astype(bool)
    uniq = np.unique(subjects[pool])
    rng = np.random.default_rng(seed)
    assign = dict(zip(uniq, rng.permutation(uniq.size) % n_folds))
    fold_of = np.array([assign.get(s, -1) for s in subjects])

    test = pool & (fold_of == fold)
    rest = pool & ~test
    val = _subject_holdout(subjects, rest, val_frac, seed + 1000 + fold)
    return {"train": rest & ~val, "val": val, "test": test}


def leave_one_source_out(store: SequenceStore, cohort: str, val_frac: float = 0.12,
                         seed: int = 0) -> Dict[str, np.ndarray]:
    """Hold out an entire source dataset -- transportability, not just new patients."""
    a = store.anchors
    subjects = a["subject"].to_numpy()
    held = a["cohort"].to_numpy() == store.cohort_code(cohort)
    rest = ~held
    val = _subject_holdout(subjects, rest, val_frac, seed)
    return {"train": rest & ~val, "val": val, "test": held}


def make_split(store: SequenceStore, scheme: str, **kw) -> Dict[str, np.ndarray]:
    if scheme == "official":
        split = official_split(store, kw.get("val_frac", 0.12), kw.get("seed", 0))
    elif scheme == "subject_kfold":
        split = subject_kfold(store, kw.get("n_folds", 5), kw.get("fold", 0),
                              kw.get("val_frac", 0.12), kw.get("seed", 0))
    elif scheme == "loso":
        cohort = kw.get("holdout_source")
        if not cohort:
            raise ValueError("scheme='loso' requires holdout_source=<source_file>")
        split = leave_one_source_out(store, cohort, kw.get("val_frac", 0.12),
                                     kw.get("seed", 0))
    else:
        raise ValueError(f"unknown split scheme {scheme!r}")
    assert_no_subject_overlap(store, split, allow_train_test=(scheme == "official"))
    return split


# --------------------------------------------------------------- eval strata
def test_strata(store: SequenceStore, split: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Named subsets of the test mask that every run is scored on.

    * ``all``            every test anchor
    * ``novel_patient``  test subjects with no training data at all
    * ``known_patient``  subjects split temporally across the official boundary
    * ``leaderboard``    excludes the four cohorts the public leaderboard drops
                         (no carbohydrate data, or MDI-only), for comparability
    """
    a = store.anchors
    test = split["test"]
    subjects = a["subject"].to_numpy()
    train_subjects = np.unique(subjects[split["train"]])
    seen = np.isin(subjects, train_subjects)

    excluded = [store.cohort_code(c) for c in K.LEADERBOARD_EXCLUDED_COHORTS
                if c in store.cohort_names()]
    cohort = a["cohort"].to_numpy()

    return {
        "all": test,
        "novel_patient": test & ~seen,
        "known_patient": test & seen,
        "leaderboard": test & ~np.isin(cohort, excluded),
    }


def seen_in_train(store, split: Dict[str, np.ndarray],
                  anchor_index: np.ndarray) -> np.ndarray:
    """Did each anchor's patient contribute any training window to *this* run?

    This, not the corpus's ``subject_split_across_traintest`` flag, is what
    separates the novel-patient stratum from the known-patient one. The two agree
    on the official split by construction, but they diverge exactly where it
    matters: under leave-one-source-out every held-out patient is novel to the
    model regardless of how the benchmark labelled them.
    """
    subjects = store.anchors["subject"].to_numpy()
    train_subjects = np.unique(subjects[split["train"]])
    return np.isin(subjects[anchor_index], train_subjects)


# ------------------------------------------------------------------- guards
def assert_no_subject_overlap(store: SequenceStore, split: Dict[str, np.ndarray],
                              allow_train_test: bool = False) -> None:
    """Fail loudly, never warn.

    train/val overlap is always an error. train/test overlap is an error for every
    scheme except ``official``, where the benchmark deliberately splits some
    subjects in time -- and even there it is only allowed because targets and
    windows were bounded to one side of the boundary at prepare time.
    """
    subjects = store.anchors["subject"].to_numpy()
    tr = set(np.unique(subjects[split["train"]]).tolist())
    va = set(np.unique(subjects[split["val"]]).tolist())
    te = set(np.unique(subjects[split["test"]]).tolist())
    if tr & va:
        raise SplitError(f"{len(tr & va)} subjects appear in both train and val")
    if not allow_train_test and (tr & te):
        raise SplitError(f"{len(tr & te)} subjects appear in both train and test")
    for name, mask in split.items():
        if not mask.any():
            raise SplitError(f"split part {name!r} is empty")


def describe_split(store: SequenceStore, split: Dict[str, np.ndarray]) -> pd.DataFrame:
    a = store.anchors
    rows = []
    for name, mask in split.items():
        sub = a[mask]
        rows.append({
            "part": name,
            "anchors": int(mask.sum()),
            "subjects": int(sub["subject"].nunique()),
            "sources": int(sub["cohort"].nunique()),
        })
    return pd.DataFrame(rows).set_index("part")


def subsample_mask(mask: np.ndarray, max_items: Optional[int], seed: int = 0,
                   subjects: Optional[np.ndarray] = None) -> np.ndarray:
    """Cap the number of anchors in a mask, keeping subject coverage even.

    Used for training-set caps and for fast evaluation passes. Sampling is
    stratified by subject when subject ids are supplied, so a cap does not
    silently turn the run into a study of whoever has the longest recording.
    """
    if max_items is None or mask.sum() <= max_items:
        return mask
    idx = np.flatnonzero(mask)
    rng = np.random.default_rng(seed)
    if subjects is None:
        keep = rng.choice(idx, size=max_items, replace=False)
    else:
        subj = subjects[idx]
        uniq, inv = np.unique(subj, return_inverse=True)
        per = max(1, int(np.ceil(max_items / uniq.size)))
        keep_parts: List[np.ndarray] = []
        for s in range(uniq.size):
            here = idx[inv == s]
            if here.size > per:
                here = rng.choice(here, size=per, replace=False)
            keep_parts.append(here)
        keep = np.concatenate(keep_parts)
        if keep.size > max_items:
            keep = rng.choice(keep, size=max_items, replace=False)
    out = np.zeros_like(mask)
    out[np.sort(keep)] = True
    return out
