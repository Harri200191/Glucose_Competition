"""Train / validation / test partitioning.

EDA 11 is unambiguous about what is and is not allowed:

* A random **row** split leaks almost completely -- r(CGM_t, CGM_t+5min) ~ 0.99.
* 166-169 subjects appear on **both** sides of `is_test`, split *temporally*: their
  earlier window is train, their later window is test. Grouping by subject alone is
  therefore not enough for those subjects; the time boundary matters too.
* Null patterns nearly determine the source cohort (EDA 3.3), so subject-level CV can
  still reward cohort-specific shortcuts. Leave-one-cohort-out is the honest check.

Three schemes are provided and all of them return index arrays into the built
dataset, never a copy of it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import LEADERBOARD_EXCLUDED_COHORTS


def _subject_holdout(subjects: pd.Series, mask: np.ndarray, frac: float,
                     seed: int) -> np.ndarray:
    """Carve a subject-disjoint validation slice out of `mask`."""
    pool = pd.unique(subjects[mask])
    rng = np.random.default_rng(seed)
    n_val = max(1, int(round(len(pool) * frac)))
    val_subjects = set(rng.choice(pool, size=min(n_val, len(pool)), replace=False))
    return mask & subjects.isin(val_subjects).to_numpy()


def official_split(meta: pd.DataFrame, val_frac: float = 0.12, seed: int = 0
                   ) -> dict[str, np.ndarray]:
    """The competition's own split, with a subject-disjoint val slice from train.

    Returns boolean masks: train, val, test.

    The 166-169 subjects that sit on both sides of `is_test` (EDA 11.1) are
    **excluded from the validation pool**. Their train rows are legitimate training
    data, but using them for early stopping would tune the stopping point on people
    who also appear in test.
    """
    is_test = meta["is_test"].to_numpy().astype(bool)
    train_all = ~is_test

    both = set(pd.unique(meta["subject"][is_test]))
    val_pool = train_all & ~meta["subject"].isin(both).to_numpy()
    val = _subject_holdout(meta["subject"], val_pool, val_frac, seed)

    train = train_all & ~val
    return {"train": train, "val": val, "test": is_test}


def subject_kfold(meta: pd.DataFrame, n_folds: int = 5, fold: int = 0,
                  val_frac: float = 0.12, seed: int = 0,
                  within_train_only: bool = True) -> dict[str, np.ndarray]:
    """Group K-fold over `(source_file, id)`.

    With `within_train_only`, folds are drawn only from rows the competition calls
    train, so the official test split stays untouched for a final read.
    """
    pool_mask = (~meta["is_test"].to_numpy().astype(bool)) if within_train_only \
        else np.ones(len(meta), dtype=bool)
    subjects = pd.unique(meta["subject"][pool_mask])
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(subjects))
    assign = {s: int(order[i] % n_folds) for i, s in enumerate(subjects)}
    fold_of = meta["subject"].map(assign)

    test = pool_mask & (fold_of == fold).to_numpy()
    rest = pool_mask & ~test
    val = _subject_holdout(meta["subject"], rest, val_frac, seed + 1000 + fold)
    return {"train": rest & ~val, "val": val, "test": test}


def leave_one_cohort_out(meta: pd.DataFrame, cohort: str, val_frac: float = 0.12,
                         seed: int = 0) -> dict[str, np.ndarray]:
    """Hold out an entire source study.

    This is the only split that measures whether a model has learned physiology
    rather than 'which study is this' (EDA 3.3, finding 2).
    """
    held = (meta["source_file"] == cohort).to_numpy()
    rest = ~held
    val = _subject_holdout(meta["subject"], rest, val_frac, seed)
    return {"train": rest & ~val, "val": val, "test": held}


def make_split(meta: pd.DataFrame, scheme: str, **kw) -> dict[str, np.ndarray]:
    if scheme == "official":
        return official_split(meta, kw.get("val_frac", 0.12), kw.get("seed", 0))
    if scheme == "subject":
        return subject_kfold(meta, kw.get("n_folds", 5), kw.get("fold", 0),
                             kw.get("val_frac", 0.12), kw.get("seed", 0))
    if scheme == "cohort":
        c = kw.get("holdout_cohort")
        if not c:
            raise ValueError("scheme='cohort' needs holdout_cohort=<name>")
        return leave_one_cohort_out(meta, c, kw.get("val_frac", 0.12), kw.get("seed", 0))
    raise ValueError(f"unknown split scheme {scheme!r}")


def leaderboard_mask(meta: pd.DataFrame) -> np.ndarray:
    """Approximate the actually-scored slice (EDA 13).

    The dataset builder already enforced the 24 h clean-history rule, so what is
    left is the cohort exclusion. This is an upper bound on the official subset,
    not a reproduction -- the organisers also constrain insulin outages and use
    their own sampling stride.
    """
    return (meta["is_test"].to_numpy().astype(bool)
            & ~meta["source_file"].isin(LEADERBOARD_EXCLUDED_COHORTS).to_numpy())


def describe_split(meta: pd.DataFrame, split: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name, m in split.items():
        sub = meta[m]
        rows.append(dict(part=name, rows=int(m.sum()),
                         subjects=int(sub["subject"].nunique()),
                         cohorts=int(sub["source_file"].nunique()),
                         first=sub["date"].min(), last=sub["date"].max()))
    out = pd.DataFrame(rows).set_index("part")
    # the invariant that matters: no subject may straddle train and val
    tr = set(meta["subject"][split["train"]])
    va = set(meta["subject"][split["val"]])
    te = set(meta["subject"][split["test"]])
    out.attrs["train_val_subject_overlap"] = len(tr & va)
    out.attrs["train_test_subject_overlap"] = len(tr & te)
    return out
