"""Cutting the corpus into contiguous, modellable stretches.

Two distinct notions, deliberately kept apart:

**Segment** -- a maximal run of rows that are (a) the same subject and (b) exactly
`GRID_MIN` apart. This is a property of the *lattice*, not of the CGM signal. EDA 7.1
found only 1,909 such runs across 1,907 subjects: the lattice is dense, and the only
two breaks are the ShanghaiT1DM subjects stored as two concatenated wear periods
(one of them backwards in time by 39.7 days). Lags and rolling windows must never
cross a segment boundary.

**CGM gap** -- a run of NaN in the target channel *inside* a segment. EDA 7.4 found
1,147,210 of these, holding 15.48% of all slots. These do NOT break the segment,
because the insulin and carb channels keep recording through them; they instead
disqualify anchors (via `max_cgm_gap_in_window`) and bound interpolation.

Conflating the two is the mistake this module exists to prevent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import GRID_MIN


def compute_segments(source_file: pd.Series, subject_id: pd.Series,
                     date: pd.Series, grid_min: int = GRID_MIN) -> np.ndarray:
    """Return an int64 segment id per row.

    Assumes rows are grouped by subject and ascending in time within a subject --
    true for this corpus (EDA 7.1). Any violation simply starts a new segment, so
    the result is correct even if that assumption breaks; it just yields more
    segments. `audit_segments` reports if that happens.
    """
    src = pd.factorize(source_file)[0]
    sid = pd.factorize(subject_id)[0]
    dts = date.to_numpy()

    same_subject = np.empty(len(src), dtype=bool)
    same_subject[0] = False
    same_subject[1:] = (src[1:] == src[:-1]) & (sid[1:] == sid[:-1])

    delta_s = np.empty(len(src), dtype=np.int64)
    delta_s[0] = -1
    delta_s[1:] = (dts[1:] - dts[:-1]).astype("timedelta64[s]").astype(np.int64)

    continues = same_subject & (delta_s == grid_min * 60)
    return np.cumsum(~continues, dtype=np.int64)


def segment_bounds(seg_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First and last row index of each segment, and the per-row start index."""
    starts = np.flatnonzero(np.concatenate(([True], seg_id[1:] != seg_id[:-1])))
    ends = np.concatenate((starts[1:] - 1, [len(seg_id) - 1]))
    return starts, ends


def row_segment_start(seg_id: np.ndarray) -> np.ndarray:
    """For each row, the index at which its segment begins."""
    starts, _ = segment_bounds(seg_id)
    # seg_id is 1-based and dense because it comes from a cumsum of a True-first mask
    return starts[seg_id - seg_id.min()]


def audit_segments(seg_id: np.ndarray, source_file: pd.Series, subject_id: pd.Series,
                   date: pd.Series) -> dict:
    """Facts about the segmentation, for the run manifest."""
    n_seg = int(seg_id[-1] - seg_id[0] + 1)
    n_subj = int(pd.MultiIndex.from_arrays([source_file, subject_id]).nunique())
    starts, ends = segment_bounds(seg_id)
    lengths = ends - starts + 1
    dts = date.to_numpy()
    # every segment must be strictly increasing in time
    monotone = bool(np.all((dts[1:] > dts[:-1]) | (seg_id[1:] != seg_id[:-1])))
    return {
        "n_segments": n_seg,
        "n_subjects": n_subj,
        "extra_segments_beyond_subjects": n_seg - n_subj,
        "segment_len_min": int(lengths.min()),
        "segment_len_median": float(np.median(lengths)),
        "segment_len_max": int(lengths.max()),
        "all_segments_time_monotone": monotone,
        "segments_under_6h": int((lengths < 6 * 60 // GRID_MIN).sum()),
    }


def nan_run_lengths(missing: np.ndarray, seg_id: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For a boolean `missing` mask, return per-row run length (0 where present),
    plus the first and last row index of every run.

    Runs are broken at segment boundaries so a gap never spans two wear periods.
    """
    n = len(missing)
    seg_start = np.empty(n, dtype=bool)
    seg_start[0] = True
    seg_start[1:] = seg_id[1:] != seg_id[:-1]

    prev_missing = np.empty(n, dtype=bool)
    prev_missing[0] = False
    prev_missing[1:] = missing[:-1]

    new_run = missing & (~prev_missing | seg_start)
    run_id = np.cumsum(new_run)                      # 1-based; 0 means "before any run"
    counts = np.bincount(run_id[missing])            # counts[0] is meaningless
    lengths = counts[1:] if len(counts) > 1 else np.array([], dtype=np.int64)

    per_row = np.where(missing, counts[run_id] if len(counts) else 0, 0).astype(np.int32)
    first = np.flatnonzero(new_run)
    last = first + lengths - 1
    return per_row, first, last


def interpolate_short_gaps(cgm: np.ndarray, seg_id: np.ndarray,
                           max_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Linearly fill CGM gaps of at most `max_steps`, inside a segment only.

    Returns ``(filled, was_interpolated)``.

    Longer gaps are left NaN on purpose: EDA 7.4 showed 83% of gaps are short but
    they hold only 6.6% of the lost time, so interpolating everything would
    fabricate the overwhelming majority of the missing signal.

    This value is used **only to build features**. The prediction target is always
    a real reading -- see `build_dataset.py`.
    """
    n = len(cgm)
    missing = ~np.isfinite(cgm)
    per_row, first, last = nan_run_lengths(missing, seg_id)

    filled = pd.Series(cgm).interpolate(method="linear", limit_area="inside").to_numpy()

    # A run may only be filled if it is short AND has a real neighbour on both
    # sides *within the same segment*.
    if len(first):
        lengths = last - first + 1
        ok = lengths <= max_steps
        ok &= first > 0
        ok &= last < n - 1
        prev_idx = np.clip(first - 1, 0, n - 1)
        next_idx = np.clip(last + 1, 0, n - 1)
        ok &= seg_id[prev_idx] == seg_id[first]
        ok &= seg_id[next_idx] == seg_id[last]

        allowed = np.zeros(n, dtype=bool)
        good_first, good_last = first[ok], last[ok]
        if len(good_first):
            marks = np.zeros(n + 1, dtype=np.int32)
            np.add.at(marks, good_first, 1)
            np.add.at(marks, good_last + 1, -1)
            allowed = np.cumsum(marks[:-1]) > 0
    else:
        allowed = np.zeros(n, dtype=bool)

    was_interp = missing & allowed
    out = np.where(missing, np.where(allowed, filled, np.nan), cgm)
    return out.astype(np.float32), was_interp
