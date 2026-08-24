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

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..common.config import REPO_ROOT
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


# ------------------------------------------------ competition-subject policies
def _subject_keys(store: SequenceStore) -> pd.DataFrame:
    """``(cohort, pid)`` per store subject -- the only identity that survives
    across stores.

    ``subject`` is a per-store row index and means nothing outside the store that
    minted it. ``cohort`` and ``pid`` are codes against the vocabularies built
    while reading the corpus, so two stores prepared from the *same* corpus file
    agree on them. Two stores built from different corpora do not, which is why
    :func:`competition_subject_mask` refuses to compare them.
    """
    subs = pd.read_parquet(store.root / "subjects.parquet")
    return subs[["subject", "cohort", "pid"]]


def competition_subject_mask(store: SequenceStore,
                             other_store_root: os.PathLike) -> np.ndarray:
    """Anchors of ``store`` whose patient also appears in ``other_store``."""
    root = Path(other_store_root)
    if not (root / "subjects.parquet").exists():
        raise SplitError(
            f"no subject table at {root}. Build the store first, e.g. "
            f"scripts/90_make_submission.sh, which runs prepare.py with "
            f"--anchors-from the competition template.")

    other = SequenceStore(root)
    mine, theirs = _subject_keys(store), _subject_keys(other)
    if (store.manifest.get("levels", {}).get("source_file", [])
            != other.manifest.get("levels", {}).get("source_file", [])):
        raise SplitError(
            f"{root.name} was prepared from a different corpus than {store.root.name} "
            f"(source_file vocabularies differ), so its cohort/pid codes are not "
            f"comparable and this would silently match the wrong patients.")

    keys = set(zip(theirs["cohort"], theirs["pid"]))
    hit_subjects = mine.loc[[k in keys for k in zip(mine["cohort"], mine["pid"])],
                            "subject"].to_numpy()
    return np.isin(store.anchors["subject"].to_numpy(), hit_subjects)


def apply_live_policy(store: SequenceStore, split: Dict[str, np.ndarray],
                      policy: str, other_store_root: os.PathLike
                      ) -> Tuple[Dict[str, np.ndarray], str]:
    """Rewrite the split according to how the live-leaderboard patients are treated.

    The live set is drawn entirely from the official *test* portion, but 125 of its
    279 patients also contribute earlier recordings to training under the
    benchmark's temporal known-patient split. The three policies differ only in
    what happens to those patients' anchors:

    ``none``     the benchmark split, untouched. The default, and the only
                 setting whose test score is comparable with every other run.
    ``exclude``  drop the patients from train and val entirely, so the model has
                 never met anyone it will be scored on. Test is left alone.
    ``include``  move the patients' *test-side* anchors into train as well, so the
                 model trains on the leaderboard data itself. This deliberately
                 contaminates the internal test split for those patients -- the
                 resulting ``predictions_test`` numbers are no longer an honest
                 estimate of anything, and only the submission is meaningful.

    Returns the new split and a one-line description for the run log.
    """
    if policy in (None, "", "none"):
        return split, "live_policy=none (benchmark split untouched)"
    if policy not in ("exclude", "include"):
        raise SplitError(f"unknown live_policy {policy!r}; expected none, exclude or include")

    hit = competition_subject_mask(store, other_store_root)
    out = dict(split)
    before = int(out["train"].sum())

    if policy == "exclude":
        for part in ("train", "val"):
            if part in out:
                out[part] = out[part] & ~hit
        note = (f"live_policy=exclude: dropped {before - int(out['train'].sum()):,} train "
                f"anchors of live-leaderboard patients "
                f"({before:,} -> {int(out['train'].sum()):,}); val and test untouched")
    else:
        # Everything of theirs that is not already training data -- in practice the
        # test-side anchors, since the train-side ones are in the mask already.
        added = hit & ~out["train"] & ~out["val"]
        out["train"] = out["train"] | added
        note = (f"live_policy=include: added {int(added.sum()):,} anchors of "
                f"live-leaderboard patients to train ({before:,} -> "
                f"{int(out['train'].sum()):,}). The internal test split now overlaps "
                f"training for these patients -- test metrics are NOT an honest estimate")
    return out, note


# ------------------------------------------------------ annual-holdout mimic
# The annual competition holdout is one homogeneous slice of the corpus, not a
# random sample of it. Profiled from the public release of its feature columns
# (``data/annual_competition_secret_holdout_set_public.parquet``, 90 patients,
# 2.62 M rows):
#
#     cgm_device                  Dexcom G6               100% of patients
#     insulin_delivery_modality   AID                     100%
#     insulin_delivery_algorithm  iLet Bionic Pancreas    100%
#     insulin_delivery_device     Beta Bionics Gen 4 iLet 100%
#     carbohydrate log            absent                  0% of rows
#     activity (heart rate, steps) absent                 0% of rows
#     age                         median 17, 52% under 18, range 6-71
#     age_of_diagnosis            withheld (all null)
#
# Exactly one source dataset in the public corpus carries that
# device/algorithm pair: IOBP2, the iLet Bionic Pancreas trial -- 332 patients,
# Dexcom G6, AID, and likewise no carbohydrate and no activity logging. 332 + 90
# is close to that trial's full enrolment, so the annual set is best read as the
# IOBP2 patients withheld from the public release.
#
# The mimic therefore holds out whole IOBP2 patients, sampled to match the annual
# age distribution band for band. Two mismatches are structural and are not
# papered over:
#
#   * sex cannot be matched. IOBP2 ships no gender at all, while the annual set
#     is 57 male / 32 female. Nothing in the corpus can fix that.
#   * the annual template asks for ~25 anchors per patient; the mimic scores
#     every anchor those patients have, which is ~2,500 each. That is a
#     deliberate difference -- more anchors is a tighter estimate of the same
#     quantity, not a different quantity -- and ``max_anchors_per_subject``
#     narrows it for anyone who wants the shapes to match exactly.
ANNUAL_MIMIC_SOURCE = "IOBP2"
ANNUAL_MIMIC_N_SUBJECTS = 89
# 200 rather than inf as the top edge: these end up in a JSON sidecar, and
# Infinity is not valid JSON.
ANNUAL_AGE_EDGES: Tuple[float, ...] = (0.0, 10.0, 14.0, 18.0, 26.0, 40.0, 60.0, 200.0)
# Patients per band in the annual holdout's 89 windowed subjects. Used only when
# the prepared annual store is absent; when it is present the shares are read
# from it so this never drifts out of date.
ANNUAL_AGE_COUNTS: Tuple[int, ...] = (10, 17, 19, 6, 15, 12, 10)
DEFAULT_ANNUAL_STORE = "artifacts/store/submission_annual"


def _age_bands(ages: np.ndarray, edges: Sequence[float]) -> np.ndarray:
    """Band index per age; ``-1`` for a missing age."""
    band = np.searchsorted(np.asarray(edges[1:], dtype=float), ages, side="right")
    return np.where(np.isfinite(ages), band, -1)


def _reference_age_counts(reference_store: Optional[os.PathLike],
                          edges: Sequence[float]) -> Tuple[np.ndarray, str]:
    """Patients per age band in the real annual holdout.

    Read from the prepared annual store when it exists, so the target
    distribution tracks the actual holdout; otherwise the frozen profile above.
    """
    n_bands = len(edges) - 1
    if reference_store is not None:
        root = Path(reference_store)
        table = root / "subjects.parquet"
        if table.exists():
            ages = pd.read_parquet(table)["age"].to_numpy(dtype=float)
            bands = _age_bands(ages, edges)
            counts = np.array([(bands == b).sum() for b in range(n_bands)], dtype=float)
            if counts.sum() > 0:
                return counts, f"age profile from {root}"
    counts = np.asarray(ANNUAL_AGE_COUNTS, dtype=float)
    if counts.size != n_bands:
        raise SplitError(
            f"frozen annual age profile has {counts.size} bands but {n_bands} edges "
            f"were given; pass a reference store or matching age_edges")
    return counts, "frozen annual age profile (annual store not built)"


def annual_mimic_subjects(store: SequenceStore,
                          n_subjects: int = ANNUAL_MIMIC_N_SUBJECTS,
                          source: str = ANNUAL_MIMIC_SOURCE,
                          age_edges: Sequence[float] = ANNUAL_AGE_EDGES,
                          reference_store: Optional[os.PathLike] = None,
                          eligible: Optional[np.ndarray] = None,
                          seed: int = 0) -> Tuple[np.ndarray, str]:
    """Pick the internal-holdout patients: ``source`` patients whose age
    distribution matches the annual holdout's, band by band.

    ``eligible`` is a boolean mask over the store's subject table restricting who
    may be chosen -- used to keep the official test subjects out, so the official
    test split is left byte-identical and every historical number stays
    comparable.

    Returns the chosen subject ids and a one-line description.
    """
    subjects = store.subjects
    try:
        code = store.cohort_code(source)
    except KeyError as exc:
        raise SplitError(
            f"internal holdout wants source {source!r}, which this store does not "
            f"have: {store.cohort_names()}") from exc

    pool = (subjects["cohort"].to_numpy() == code)
    if eligible is not None:
        pool = pool & eligible
    if not pool.any():
        raise SplitError(f"no eligible {source} patients to draw an internal holdout from")

    target, ref_note = _reference_age_counts(reference_store, age_edges)
    target = target / target.sum() * float(n_subjects)

    ages = subjects["age"].to_numpy(dtype=float)
    bands = _age_bands(ages, age_edges)
    sid = subjects["subject"].to_numpy()
    rng = np.random.default_rng(seed)

    # Largest-remainder allocation, then draw within each band. A band the pool
    # cannot fill gives back its shortfall, which is redistributed over the bands
    # that still have patients left -- so the count is always exactly n_subjects
    # when the pool is big enough, and the shortfall is reported when it is not.
    want = np.floor(target).astype(int)
    for b in np.argsort(-(target - want))[: int(round(target.sum())) - int(want.sum())]:
        want[b] += 1

    chosen: List[np.ndarray] = []
    available = {b: sid[pool & (bands == b)] for b in range(len(age_edges) - 1)}
    short = 0
    for b, k in enumerate(want):
        have = available[b]
        take = min(int(k), have.size)
        short += int(k) - take
        if take:
            chosen.append(rng.choice(have, size=take, replace=False))
            available[b] = np.setdiff1d(have, chosen[-1])
    if short:
        rest = np.concatenate([v for v in available.values() if v.size]) \
            if any(v.size for v in available.values()) else np.zeros(0, dtype=sid.dtype)
        if rest.size:
            extra = rng.choice(rest, size=min(short, rest.size), replace=False)
            chosen.append(extra)
            short -= extra.size

    picked = np.sort(np.concatenate(chosen)) if chosen else np.zeros(0, dtype=sid.dtype)
    note = (f"internal holdout: {picked.size} {source} patients "
            f"(pool {int(pool.sum())}, {ref_note})")
    if short:
        note += f"; {short} short of the requested {n_subjects}"
    return picked, note


DEFAULT_HOLDOUT_PATH = "data/internal_holdout.parquet"


def _holdout_table(store: SequenceStore, mask: np.ndarray) -> pd.DataFrame:
    """Anchor-level dump of the holdout: what it is, in one file.

    Keyed on ``(source_file, pid)`` rather than ``subject``: ``subject`` is a row
    index into whichever store minted it and means nothing anywhere else, while
    the source name and patient id survive a rebuild.
    """
    a = store.anchors
    idx = np.flatnonzero(mask)
    names = store.cohort_names()
    subs = store.subjects.set_index("subject")
    sid = a["subject"].to_numpy()[idx]
    return pd.DataFrame({
        "anchor": idx.astype(np.int64),
        "subject": sid.astype(np.int32),
        "pid": subs["pid"].reindex(sid).to_numpy(),
        "source_file": [names[int(c)] for c in subs["cohort"].reindex(sid).to_numpy()],
        "age": subs["age"].reindex(sid).to_numpy(),
        "is_test": a["is_test"].to_numpy()[idx].astype(bool),
    })


def _holdout_mask_from_table(store: SequenceStore, table: pd.DataFrame,
                             n_anchors: Optional[int]) -> Tuple[np.ndarray, str]:
    """Rebuild the holdout mask for ``store`` from a saved table."""
    a = store.anchors
    if n_anchors == len(a):
        mask = np.zeros(len(a), dtype=bool)
        mask[table["anchor"].to_numpy()] = True
        return mask, "by anchor id"
    # Different store: fall back to patient identity and take all their anchors.
    names = store.cohort_names()
    subs = store.subjects
    keys = set(zip(table["source_file"].astype(str), table["pid"].astype(int)))
    mine = [(names[int(c)], int(p)) for c, p in zip(subs["cohort"], subs["pid"])]
    picked = subs["subject"].to_numpy()[np.array([k in keys for k in mine], dtype=bool)]
    if picked.size == 0:
        raise SplitError(
            f"saved internal holdout names no patient present in {store.root}; "
            f"the stores were built from different corpora")
    return np.isin(a["subject"].to_numpy(), picked), "by patient identity (store differs)"


def apply_internal_holdout(store: SequenceStore, split: Dict[str, np.ndarray],
                           n_subjects: int = ANNUAL_MIMIC_N_SUBJECTS,
                           source: str = ANNUAL_MIMIC_SOURCE,
                           age_edges: Sequence[float] = ANNUAL_AGE_EDGES,
                           reference_store: Optional[os.PathLike] = None,
                           allow_test_subjects: bool = False,
                           max_anchors_per_subject: Optional[int] = None,
                           path: Optional[os.PathLike] = None,
                           refresh: bool = False,
                           seed: int = 0,
                           ) -> Tuple[Dict[str, np.ndarray], str]:
    """Add an ``internal_holdout`` part that mimics the annual competition set.

    The chosen patients are removed from **train and val entirely** -- every
    anchor of theirs, on both sides of the official temporal boundary. The model
    therefore never sees a single window from a holdout patient, which is the
    whole point: the internal holdout has to answer "how would this do on 89
    iLet patients it has never met", and it can only answer that if it has never
    met them. :func:`assert_holdout_unseen` enforces it before this returns, and
    again inside :func:`assert_no_subject_overlap`.

    The selection is written once to ``path`` (default
    ``data/internal_holdout.parquet``) and every later run reuses that file, so
    all runs are compared on an identical patient set even if the code, the seed
    or the config drifts. ``refresh=True`` re-draws and overwrites it.

    By default the holdout is drawn only from patients with no anchors in the
    official test split, so ``predictions_test`` is unchanged and every number
    already in ``results/all_runs.csv`` stays comparable. Set
    ``allow_test_subjects`` to relax that.
    """
    a = store.anchors
    subjects = a["subject"].to_numpy()
    dest = Path(path) if path is not None else Path(DEFAULT_HOLDOUT_PATH)
    if not dest.is_absolute():
        dest = REPO_ROOT / dest
    meta_path = dest.with_suffix(".json")

    if dest.exists() and not refresh:
        table = pd.read_parquet(dest)
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        hold, how = _holdout_mask_from_table(store, table, meta.get("store_n_anchors"))
        picked = np.unique(subjects[hold])
        note = (f"internal holdout: reused {dest} -- {picked.size} patients, "
                f"{int(hold.sum()):,} anchors, resolved {how}")
    else:
        eligible = np.ones(len(store.subjects), dtype=bool)
        if not allow_test_subjects:
            test_subjects = np.unique(subjects[split["test"]])
            eligible &= ~np.isin(store.subjects["subject"].to_numpy(), test_subjects)

        picked, note = annual_mimic_subjects(
            store, n_subjects=n_subjects, source=source, age_edges=age_edges,
            reference_store=reference_store, eligible=eligible, seed=seed)
        if picked.size == 0:
            raise SplitError("internal holdout selected no patients")

        hold = np.isin(subjects, picked)
        if max_anchors_per_subject:
            hold = subsample_mask(hold, int(max_anchors_per_subject) * picked.size,
                                  seed + 7, subjects)

        dest.parent.mkdir(parents=True, exist_ok=True)
        _holdout_table(store, hold).to_parquet(dest, index=False)
        ages = store.subjects.set_index("subject")["age"].reindex(picked).to_numpy()
        meta_path.write_text(json.dumps({
            "store": str(store.root), "store_n_anchors": int(len(a)),
            "source": source, "n_subjects": int(picked.size),
            "n_anchors": int(hold.sum()), "seed": int(seed),
            "age_edges": [float(e) for e in age_edges],
            "age_median": float(np.nanmedian(ages)) if ages.size else None,
            "frac_under_18": float(np.nanmean(ages < 18)) if ages.size else None,
            "allow_test_subjects": bool(allow_test_subjects),
            "max_anchors_per_subject": max_anchors_per_subject,
        }, indent=1))
        note += f"; wrote {dest}"

    out = dict(split)
    before = int(out["train"].sum())
    for part in ("train", "val"):
        if part in out:
            out[part] = out[part] & ~np.isin(subjects, picked)
    out["internal_holdout"] = hold

    note += (f"; {int(hold.sum()):,} holdout anchors over {picked.size} patients; "
             f"train {before:,} -> {int(out['train'].sum()):,}")
    assert_holdout_unseen(store, out)
    return out, note


def assert_holdout_unseen(store: SequenceStore, split: Dict[str, np.ndarray]) -> None:
    """The one invariant that makes the internal holdout worth reading.

    Called from :func:`apply_internal_holdout` and again from
    :func:`make_split`'s guard, so a hand-edited split cannot smuggle a holdout
    patient into training without the run dying first.
    """
    if "internal_holdout" not in split:
        return
    subjects = store.anchors["subject"].to_numpy()
    held = set(np.unique(subjects[split["internal_holdout"]]).tolist())
    for part in ("train", "val"):
        if part not in split:
            continue
        seen = set(np.unique(subjects[split[part]]).tolist())
        overlap = held & seen
        if overlap:
            raise SplitError(
                f"{len(overlap)} internal-holdout patients also appear in {part}: "
                f"the holdout would be scoring patients the model trained on. "
                f"Subjects: {sorted(overlap)[:10]}")


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
    assert_holdout_unseen(store, split)


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
