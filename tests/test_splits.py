"""Split invariants.

Two subjects on both sides of a boundary is not a rounding error; it is the
difference between "generalises to new patients" and "memorised these patients".
The code raises rather than warns, and these tests check that it does.
"""
from __future__ import annotations

import numpy as np
import pytest

from metabonet_forecasting.data import splits as S


def test_official_split_keeps_train_and_val_subject_disjoint(store):
    split = S.official_split(store, val_frac=0.25, seed=0)
    subjects = store.anchors["subject"].to_numpy()
    tr = set(subjects[split["train"]].tolist())
    va = set(subjects[split["val"]].tolist())
    assert tr and va and not (tr & va)


def test_validation_never_contains_a_subject_that_appears_in_test(store):
    split = S.official_split(store, val_frac=0.3, seed=1)
    subjects = store.anchors["subject"].to_numpy()
    va = set(subjects[split["val"]].tolist())
    te = set(subjects[split["test"]].tolist())
    assert not (va & te)


def test_novel_and_known_patient_strata_partition_the_test_split(store):
    split = S.official_split(store, val_frac=0.2, seed=0)
    strata = S.test_strata(store, split)
    assert (strata["novel_patient"] | strata["known_patient"] == strata["all"]).all()
    assert not (strata["novel_patient"] & strata["known_patient"]).any()
    assert strata["novel_patient"].sum() > 0
    assert strata["known_patient"].sum() > 0


def test_leave_one_source_out_holds_out_the_whole_study(store):
    cohort = store.cohort_names()[0]
    split = S.leave_one_source_out(store, cohort, val_frac=0.2, seed=0)
    code = store.cohort_code(cohort)
    cohorts = store.anchors["cohort"].to_numpy()
    assert (cohorts[split["test"]] == code).all()
    assert (cohorts[split["train"]] != code).all()
    subjects = store.anchors["subject"].to_numpy()
    assert not (set(subjects[split["train"]].tolist())
                & set(subjects[split["test"]].tolist()))


def test_subject_kfold_leaves_the_official_test_split_untouched(store):
    split = S.subject_kfold(store, n_folds=3, fold=0, val_frac=0.2, seed=0)
    is_test = store.anchors["is_test"].to_numpy()
    for part in ("train", "val", "test"):
        assert not (split[part] & is_test).any()


def test_overlap_guard_raises(store):
    split = S.official_split(store, val_frac=0.2, seed=0)
    bad = {"train": split["train"] | split["val"], "val": split["val"],
           "test": split["test"]}
    with pytest.raises(S.SplitError, match="train and val"):
        S.assert_no_subject_overlap(store, bad)


def test_subsample_keeps_subject_coverage(store):
    subjects = store.anchors["subject"].to_numpy()
    mask = np.ones(len(subjects), dtype=bool)
    small = S.subsample_mask(mask, 60, seed=0, subjects=subjects)
    assert small.sum() <= 60
    # a naive head-of-array cap would have kept one or two subjects
    assert np.unique(subjects[small]).size >= min(6, np.unique(subjects).size)
