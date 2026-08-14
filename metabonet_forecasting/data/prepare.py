"""Corpus -> sequence store.

    python -m metabonet_forecasting.data.prepare --out artifacts/store/main

What this produces (one directory, described by ``manifest.json``):

    channels.npy    float32 [n_rows, n_channels]  the 5-minute lattice, all subjects
                    concatenated, cut into segments; column order = constants.CHANNELS
    segments.parquet  one row per contiguous segment: subject, cohort, offsets
    anchors.parquet   one row per candidate "now": global row index + context flags
    targets.npy     float32 [n_anchors, n_horizons] real CGM at t+h, NaN if unusable
    statics.npy     float32 [n_subjects, n_static]  numeric statics + categorical codes
    subjects.parquet  subject table (key, cohort, official split membership)
    manifest.json     config, counts, level tables, provenance

Why a store instead of materialised windows: a 24 h window per anchor would be
288 x 23 x 4 B x ~16 M anchors = 4 TB. Storing the lattice once (about 14 GB) and
gathering windows lazily by row index costs nothing at train time and lets the
context-length study reuse one build.

Ordering of operations, and why each is where it is:

1. read only the 17 columns that survive review (constants.LOAD_COLUMNS);
2. sort by (subject, time) -- two ShanghaiT1DM subjects are stored as two
   concatenated wear periods, one of them *backwards in time*, so an unsorted
   diff would compute a "5-minute delta" across 39.7 days running in reverse;
3. cut segments wherever consecutive rows are not exactly 5 minutes apart;
4. clean per row (physically impossible values -> NaN, unit conversions);
5. interpolate CGM gaps of at most 30 min *for features only* -- targets are always
   read from the raw column, so an interpolated slot never becomes a label;
6. build causal event/exposure channels per segment (IIR filters, reset at each
   segment start, so nothing crosses a segment edge);
7. select anchors and precompute targets.

Every step is causal or purely per-segment. No statistic is fitted here at all --
normalisation happens later, fitted on training subjects only (data/transforms.py).
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.signal import lfilter

from ..common.logging import get_console_logger, provenance
from . import constants as K

LOG = get_console_logger("prepare")


@contextlib.contextmanager
def no_cyclic_gc():
    """Turn off Python's cyclic garbage collector for the duration of the build.

    This is worth 50-100x here, and the reason is worth writing down. The build
    holds a dozen arrays of tens of millions of elements while allocating a new
    pyarrow Table per row group. Every allocation burst trips a generation-2
    collection, each of which walks the whole live object graph, so the cost per
    row group grows with the data already read: 0.04 s for the first row group,
    9 s for the second, 64 s for the fourth. Measured on this corpus, five row
    groups took 100 s with the collector on and 1.1 s with it off.

    Nothing here creates reference cycles -- the large objects are numpy arrays
    and Arrow tables held by name and dropped by refcount -- so the collector has
    nothing to find. It is re-enabled on the way out, including on exceptions.
    """
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


class Phase:
    """Log the start and duration of a build phase.

    The full corpus takes tens of minutes; a build that prints nothing for twenty
    of them is indistinguishable from a hung one, and this step runs rarely enough
    that nobody remembers what it does next.
    """

    def __init__(self, name: str, stats: Optional[Dict[str, object]] = None) -> None:
        self.name = name
        self.stats = stats

    def __enter__(self) -> "Phase":
        self.t0 = time.time()
        LOG.info(f"[{self.name}] start")
        return self

    def __exit__(self, *exc) -> None:
        dt = time.time() - self.t0
        LOG.info(f"[{self.name}] done in {dt:.1f}s")
        if self.stats is not None:
            self.stats.setdefault("phase_seconds", {})[self.name] = round(dt, 1)


# ----------------------------------------------------------------- build config
@dataclass
class PrepareConfig:
    """Build-time knobs. Anything that changes the store's contents lives here."""
    corpus: str = str(K.CORPUS_PARQUET)
    out: str = str(K.ARTIFACT_DIR / "store" / "main")

    # Segmentation
    max_cgm_gap_min: int = 30          # gap above which history is considered broken
    interp_gap_min: int = 30           # CGM gaps <= this are interpolated for features
    min_segment_min: int = 6 * 60      # shorter segments can never carry 24 h history

    # Anchors
    min_history_steps: int = K.MAX_CONTEXT_STEPS   # 288 = 24 h; fixes the anchor set
                                                   # across the context-length study
    anchor_stride: int = 6             # one anchor per 30 min; r(CGM_t, CGM_t+5) ~ 0.99
    require_clean_history: bool = True  # no CGM gap > max_cgm_gap_min inside the window

    # Scale control for smoke builds / debugging
    row_groups: Optional[int] = None   # read only the first N parquet row groups
    max_subjects: Optional[int] = None
    cohorts: Optional[List[str]] = None
    subject_seed: int = 0


# ------------------------------------------------------------------- utilities
def _dict_encode(chunked: pa.ChunkedArray, vocab: Dict[str, int]) -> np.ndarray:
    """String column -> int32 codes against a growing global vocabulary.

    -1 means "not recorded". A chunk can legitimately be entirely null -- a whole
    row group of a cohort that never recorded ``gender``, for instance -- which
    dictionary-encodes to an empty dictionary, so that case is handled explicitly
    rather than indexing into a zero-length lookup.
    """
    out = np.empty(len(chunked), dtype=np.int32)
    pos = 0
    for chunk in chunked.chunks:
        n = len(chunk)
        if n == 0:
            continue
        dic = chunk.dictionary_encode()
        levels = [str(v) for v in dic.dictionary.to_pylist()]
        if not levels:
            out[pos:pos + n] = -1
            pos += n
            continue
        remap = np.array([vocab.setdefault(v, len(vocab)) for v in levels], dtype=np.int32)
        idx = np.asarray(dic.indices.to_numpy(zero_copy_only=False))
        codes = np.where(np.isnan(idx.astype(np.float64)), -1, idx).astype(np.int64)
        vals = np.where(codes < 0, -1, remap[np.clip(codes, 0, len(remap) - 1)])
        out[pos:pos + n] = vals
        pos += n
    return out


def _f32(chunked: pa.ChunkedArray) -> np.ndarray:
    return np.asarray(chunked.to_numpy(zero_copy_only=False), dtype=np.float32)


def last_true_index(flags: np.ndarray) -> np.ndarray:
    """For each i, the largest j <= i with flags[j], else -1. O(n), no windows."""
    idx = np.where(flags, np.arange(flags.size, dtype=np.int64), -1)
    return np.maximum.accumulate(idx)


def steps_since(flags: np.ndarray, cap: int) -> np.ndarray:
    """Steps since the most recent True, capped. ``cap`` where none has occurred."""
    last = last_true_index(flags)
    pos = np.arange(flags.size, dtype=np.int64)
    out = np.where(last < 0, cap, pos - last)
    return np.minimum(out, cap).astype(np.float32)


def exp_exposure(x: np.ndarray, tau_steps: float) -> np.ndarray:
    """Causal exponential exposure  y[i] = a*y[i-1] + x[i],  a = exp(-1/tau).

    This is the plan's ``I_tau(t) = sum_k I(t-k) exp(-k/tau)`` written as a
    first-order IIR so it costs one pass. Run per segment, so the filter state
    never carries across a segment boundary.

    Implemented with ``lfilter`` rather than the closed form
    ``a**i * cumsum(x / a**i)``: the latter overflows once i reaches the tens of
    thousands, and segments here reach 287,102 rows.
    """
    a = float(np.exp(-1.0 / max(tau_steps, 1e-6)))
    return lfilter([1.0], [1.0, -a], x.astype(np.float64)).astype(np.float32)


def interpolate_short_gaps(values: np.ndarray, observed: np.ndarray,
                           max_gap: int) -> Tuple[np.ndarray, np.ndarray]:
    """Linearly fill runs of missing values of length <= ``max_gap``.

    A gap is only filled when it has a real reading on both sides *inside this
    segment*: a gap touching a segment edge is never filled. Returns the filled
    series and the count of filled slots.

    Forward-fill is deliberately not used: it creates flat plateaus that any
    difference feature reads as "glucose is perfectly stable", which is the
    opposite of what a sensor dropout means.
    """
    n = values.size
    filled = values.copy()
    if not observed.any():
        return filled, 0
    obs_idx = np.flatnonzero(observed)
    # Interpolate everywhere between the first and last observation, then undo it
    # for gaps that are too long.
    lo, hi = obs_idx[0], obs_idx[-1]
    if hi > lo:
        span = np.arange(lo, hi + 1)
        filled[lo:hi + 1] = np.interp(span, obs_idx, values[obs_idx])
    filled[:lo] = np.nan
    filled[hi + 1:] = np.nan

    # Undo fills inside runs longer than max_gap.
    missing = ~observed
    if missing.any():
        # run id for each missing sample: number of observed samples before it
        run_id = np.cumsum(observed)
        lengths = np.bincount(run_id[missing], minlength=int(run_id.max()) + 1)
        too_long = lengths > max_gap
        undo = missing & too_long[run_id]
        filled[undo] = np.nan
    n_filled = int(np.isfinite(filled).sum() - observed.sum())
    return filled, max(n_filled, 0)


# --------------------------------------------------------------------- reading
# (source column, destination name) for the plain numeric channels.
_NUMERIC_COLUMNS = (("CGM", "cgm"), ("basal", "basal"), ("bolus", "bolus"),
                    ("carbs", "carbs"), ("heartrate", "heartrate"),
                    ("steps", "steps"), ("age", "age"),
                    ("age_of_diagnosis", "age_of_diagnosis"),
                    ("height", "height"), ("weight", "weight"))


def _read_corpus(cfg: PrepareConfig) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, int]]]:
    """Stream the parquet into flat arrays, dictionary-encoding the string columns.

    The destination arrays are allocated once, from the row counts in the parquet
    footer, and each row group is written straight into its slice. The obvious
    alternative -- collect per-group blocks in a list and ``np.concatenate`` at the
    end -- costs twice the peak memory and, on a machine whose memory is
    fragmented by other tenants, spends most of its time in the kernel growing and
    releasing the heap. On this corpus that difference was the majority of the
    build's wall clock.
    """
    pf = pq.ParquetFile(cfg.corpus)
    n_groups = pf.num_row_groups if cfg.row_groups is None else min(cfg.row_groups,
                                                                    pf.num_row_groups)
    total = sum(pf.metadata.row_group(g).num_rows for g in range(n_groups))
    LOG.info(f"reading {n_groups}/{pf.num_row_groups} row groups "
             f"({total:,} rows) from {cfg.corpus}")

    vocab: Dict[str, Dict[str, int]] = {"source_file": {}, "id": {}, "gender": {},
                                        "insulin_delivery_modality": {}}
    data: Dict[str, np.ndarray] = {
        "cohort": np.empty(total, dtype=np.int32),
        "pid": np.empty(total, dtype=np.int32),
        "gender": np.empty(total, dtype=np.int32),
        "modality": np.empty(total, dtype=np.int32),
        "time_min": np.empty(total, dtype=np.int64),
        "is_test": np.empty(total, dtype=bool),
        "known": np.empty(total, dtype=bool),
    }
    for _, name in _NUMERIC_COLUMNS:
        data[name] = np.empty(total, dtype=np.float32)

    pos = 0
    t0 = time.time()
    for g in range(n_groups):
        tbl = pf.read_row_group(g, columns=K.LOAD_COLUMNS)
        m = tbl.num_rows
        sl = slice(pos, pos + m)
        data["cohort"][sl] = _dict_encode(tbl["source_file"], vocab["source_file"])
        data["pid"][sl] = _dict_encode(tbl["id"], vocab["id"])
        data["gender"][sl] = _dict_encode(tbl["gender"], vocab["gender"])
        data["modality"][sl] = _dict_encode(tbl["insulin_delivery_modality"],
                                            vocab["insulin_delivery_modality"])
        data["time_min"][sl] = (tbl["date"].to_numpy(zero_copy_only=False)
                                .astype("datetime64[m]").astype(np.int64))
        data["is_test"][sl] = np.asarray(
            tbl["is_test"].to_numpy(zero_copy_only=False)).astype(bool)
        data["known"][sl] = np.asarray(
            tbl["subject_split_across_traintest"].to_numpy(zero_copy_only=False)).astype(bool)
        for col, name in _NUMERIC_COLUMNS:
            data[name][sl] = _f32(tbl[col])
        del tbl
        pos += m
        if (g + 1) % 10 == 0 or g == n_groups - 1:
            elapsed = time.time() - t0
            LOG.info(f"  row group {g + 1}/{n_groups}  {pos:,}/{total:,} rows  "
                     f"{elapsed:.0f}s  ({pos / max(elapsed, 1e-6) / 1e6:.2f}M rows/s)")

    if pos != total:
        raise RuntimeError(f"read {pos} rows but the footer promised {total}")
    LOG.info(f"read {total:,} rows in {time.time() - t0:.0f}s")
    return data, vocab


def _select_subjects(data: Dict[str, np.ndarray], vocab: Dict[str, Dict[str, int]],
                     cfg: PrepareConfig) -> np.ndarray:
    """Row mask implementing --cohorts / --max-subjects."""
    keep = np.ones(len(data["cgm"]), dtype=bool)
    if cfg.cohorts:
        wanted = {vocab["source_file"][c] for c in cfg.cohorts if c in vocab["source_file"]}
        missing = [c for c in cfg.cohorts if c not in vocab["source_file"]]
        if missing:
            raise ValueError(f"unknown cohorts {missing}; have {sorted(vocab['source_file'])}")
        keep &= np.isin(data["cohort"], list(wanted))
    if cfg.max_subjects:
        subject = data["cohort"].astype(np.int64) * (10 ** 7) + data["pid"]
        uniq = np.unique(subject[keep])
        rng = np.random.default_rng(cfg.subject_seed)
        chosen = rng.choice(uniq, size=min(cfg.max_subjects, uniq.size), replace=False)
        keep &= np.isin(subject, chosen)
    return keep


# ---------------------------------------------------------------------- build
def build_store(cfg: PrepareConfig) -> Path:
    with no_cyclic_gc():
        return _build_store(cfg)


def _build_store(cfg: PrepareConfig) -> Path:
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    stats: Dict[str, object] = {}

    with Phase("read", stats):
        data, vocab = _read_corpus(cfg)
        keep = _select_subjects(data, vocab, cfg)
        if not keep.all():
            data = {k: v[keep] for k, v in data.items()}
            LOG.info(f"subject/cohort filter kept {len(data['cgm']):,} rows")
        stats["rows_read"] = int(len(data["cgm"]))

    # --- 2. order by (subject, time) -------------------------------------------
    with Phase("sort", stats):
        subject = data["cohort"].astype(np.int64) * (10 ** 7) + data["pid"].astype(np.int64)
        order = np.lexsort((data["time_min"], subject))
        was_sorted = bool(np.array_equal(order, np.arange(order.size)))
        if was_sorted:
            # The corpus arrives grouped by subject and ordered in time. Skipping
            # the reorder avoids 13 random-access gathers over the whole corpus,
            # which is the single most expensive avoidable step in this build.
            del order
        else:
            subject = subject[order]
            data = {k: v[order] for k, v in data.items()}
            del order
        stats["corpus_already_sorted"] = was_sorted
        LOG.info(f"sorted by (subject, time); input was already sorted: {was_sorted}")

    # --- 3. segments -----------------------------------------------------------
    with Phase("segment", stats):
        n = subject.size
        new_subject = np.empty(n, dtype=bool)
        new_subject[0] = True
        new_subject[1:] = subject[1:] != subject[:-1]
        dt = np.empty(n, dtype=np.int64)
        dt[0] = -1
        dt[1:] = data["time_min"][1:] - data["time_min"][:-1]
        seg_start = new_subject | (dt != K.GRID_MIN)
        del new_subject, dt
        starts = np.flatnonzero(seg_start)
        del seg_start
        ends = np.append(starts[1:], n)                  # exclusive
        lengths = ends - starts
        min_len = cfg.min_segment_min // K.GRID_MIN
        long_enough = lengths >= min_len
        stats["segments_found"] = int(len(starts))
        stats["segments_dropped_short"] = int((~long_enough).sum())
        stats["rows_dropped_short_segments"] = int(lengths[~long_enough].sum())
        LOG.info(f"{len(starts):,} segments; dropping {int((~long_enough).sum()):,} shorter "
                 f"than {cfg.min_segment_min} min")

        # Compact to the kept segments, renumbering rows so the store is contiguous.
        if not long_enough.all():
            keep_rows = np.repeat(long_enough, lengths)
            data = {k: v[keep_rows] for k, v in data.items()}
            subject = subject[keep_rows]
            del keep_rows
        lengths = lengths[long_enough]
        ends = np.cumsum(lengths)
        starts = ends - lengths
        n = int(ends[-1]) if len(ends) else 0
        if n == 0:
            raise RuntimeError("no segments survived filtering; loosen --min-segment-min")
        stats["rows_kept"] = n
        stats["n_segments"] = int(len(starts))

    # --- 4. row-level cleaning --------------------------------------------------
    with Phase("clean", stats):
        clean_log: Dict[str, int] = {}
        cgm = data["cgm"]
        bad = np.isfinite(cgm) & ((cgm < K.CGM_VALID_LO) | (cgm > K.CGM_VALID_HI))
        clean_log["cgm_out_of_range_to_nan"] = int(bad.sum())
        cgm[bad] = np.nan
        for name in ("basal", "bolus", "carbs", "steps"):
            neg = np.isfinite(data[name]) & (data[name] < 0)
            clean_log[f"negative_{name}_clipped"] = int(neg.sum())
            data[name][neg] = 0.0
        hr = data["heartrate"]
        bad_hr = np.isfinite(hr) & ((hr < 20) | (hr > 240))
        clean_log["heartrate_out_of_range_to_nan"] = int(bad_hr.sum())
        hr[bad_hr] = np.nan
        stats["clean_log"] = clean_log

    # --- 5. channel matrix ------------------------------------------------------
    # Built one segment at a time, in RAM, and written to the memmap with a single
    # contiguous assignment per segment.
    #
    # This shape is a performance requirement, not a style choice. The store is
    # row-major [n_rows, 23], so assigning a *column* (``channels[:, j] = ...``)
    # touches every 4 KB page of a multi-gigabyte file, and doing that once per
    # channel means ~13 full passes over the file in page faults and writeback.
    # On the full corpus that turns a few minutes of work into hours of kernel
    # time. Filling a contiguous block and writing it once is one sequential pass.
    phase5 = Phase("channels", stats).__enter__()
    n_ch = len(K.CHANNELS)
    ch_path = out / "channels.npy"
    channels = np.lib.format.open_memmap(ch_path, mode="w+", dtype=np.float32,
                                         shape=(n, n_ch))
    ci = K.CHANNEL_INDEX
    value_columns = (("basal", "ins_basal", "ins_basal_mask"),
                     ("bolus", "ins_bolus", "ins_bolus_mask"),
                     ("carbs", "carb_grams", "carb_mask"),
                     ("heartrate", "act_heartrate", "act_heartrate_mask"),
                     ("steps", "act_steps", "act_steps_mask"))

    cgm_observed = np.isfinite(cgm)
    cgm_feat = np.empty(n, dtype=np.float32)
    # Kept in RAM because the anchor table needs them: reading them back out of
    # the memmap would be another strided pass over the whole file.
    since_bolus = np.empty(n, dtype=np.float32)
    since_carb = np.empty(n, dtype=np.float32)

    interp_slots = 0
    max_interp = cfg.interp_gap_min // K.GRID_MIN
    cap_steps = K.SINCE_CAP_MIN // K.GRID_MIN
    t_seg = time.time()
    for si, (s, e) in enumerate(zip(starts, ends)):
        s, e = int(s), int(e)
        sl = slice(s, e)
        block = np.zeros((e - s, n_ch), dtype=np.float32)

        filled, nfill = interpolate_short_gaps(cgm[sl], cgm_observed[sl], max_interp)
        cgm_feat[sl] = filled
        interp_slots += nfill
        block[:, ci["cgm"]] = filled
        block[:, ci["cgm_mask"]] = cgm_observed[sl]

        for src, val_ch, mask_ch in value_columns:
            vals = data[src][sl]
            obs = np.isfinite(vals)
            block[:, ci[mask_ch]] = obs
            # Value channels carry 0 where unobserved; the mask says which zeros
            # are real. For dose/carb spike trains 0 is the neutral element, so
            # this is information-preserving as long as the mask travels with it.
            block[:, ci[val_ch]] = np.where(obs, vals, 0.0)

        bolus = block[:, ci["ins_bolus"]]
        basal = block[:, ci["ins_basal"]]
        carb = block[:, ci["carb_grams"]]
        for tau in K.BOLUS_TAUS_MIN:
            block[:, ci[f"ins_bolus_exp{tau}"]] = exp_exposure(bolus, tau / K.GRID_MIN)
        for tau in K.BASAL_TAUS_MIN:
            block[:, ci[f"ins_basal_exp{tau}"]] = exp_exposure(basal, tau / K.GRID_MIN)
        for tau in K.CARB_TAUS_MIN:
            block[:, ci[f"carb_exp{tau}"]] = exp_exposure(carb, tau / K.GRID_MIN)

        since_bolus[sl] = steps_since(bolus > 0, cap_steps) * K.GRID_MIN
        since_carb[sl] = steps_since(carb > 0, cap_steps) * K.GRID_MIN
        block[:, ci["ins_bolus_since"]] = since_bolus[sl]
        block[:, ci["carb_since"]] = since_carb[sl]

        channels[sl] = block
        if (si + 1) % 200 == 0:
            LOG.info(f"  segment {si + 1}/{len(starts)}  {e:,}/{n:,} rows  "
                     f"({time.time() - t_seg:.0f}s)")

    stats["cgm_real_frac"] = float(cgm_observed.mean())
    stats["cgm_interpolated_slots"] = int(interp_slots)
    stats["cgm_missing_after_interp"] = int((~np.isfinite(cgm_feat)).sum())
    channels.flush()
    phase5.__exit__()

    # Dense subject codes, computed once over all rows so that the anchor table,
    # the segment table and the subject table all agree on what subject 7 is.
    subject_code = _subject_code(subject)

    # --- 6. anchors -------------------------------------------------------------
    phase6 = Phase("anchors", stats).__enter__()
    anchors = _select_anchors(cfg, starts, ends, cgm_observed, data["is_test"], stats)
    seg_end_row = np.repeat(ends.astype(np.int64), lengths)
    targets = _build_targets(anchors["row"].to_numpy(), cgm, cgm_observed,
                             data["is_test"], seg_end_row)
    np.save(out / "targets.npy", targets)
    stats["targets_valid_per_horizon"] = {
        f"{h}min": int(np.isfinite(targets[:, i]).sum())
        for i, h in enumerate(K.ALL_HORIZONS_MIN)}

    # anchor-level context used by clinical strata and by plots
    rows = anchors["row"].to_numpy()
    anchors["cgm_now"] = cgm[rows]
    prev = np.maximum(rows - 6, 0)
    anchors["cgm_delta_30min"] = cgm_feat[rows] - cgm_feat[prev]
    anchors["min_since_carb"] = since_carb[rows]
    anchors["min_since_bolus"] = since_bolus[rows]
    tmin = data["time_min"][rows]
    anchors["tod_min"] = (tmin % 1440).astype(np.int16)
    anchors["dow"] = ((tmin // 1440) % 7).astype(np.int8)
    anchors["is_test"] = data["is_test"][rows]
    anchors["known_patient"] = data["known"][rows]
    anchors["subject"] = subject_code[rows]
    anchors["cohort"] = data["cohort"][rows].astype(np.int16)
    anchors.to_parquet(out / "anchors.parquet", index=False)
    phase6.__exit__()

    # --- 7. segment, subject and static tables ---------------------------------
    phase7 = Phase("tables", stats).__enter__()
    seg_df = pd.DataFrame({
        "seg": np.arange(len(starts), dtype=np.int32),
        "start": starts.astype(np.int64),
        "end": ends.astype(np.int64),
        "n_rows": lengths.astype(np.int64),
        "subject": subject_code[starts],
        "cohort": data["cohort"][starts].astype(np.int16),
        "t0_min": data["time_min"][starts],
    })
    seg_df.to_parquet(out / "segments.parquet", index=False)

    subj_df, statics = _subject_table(data, subject_code, vocab)
    subj_df.to_parquet(out / "subjects.parquet", index=False)
    np.save(out / "statics.npy", statics)
    phase7.__exit__()

    # --- 8. manifest ------------------------------------------------------------
    manifest = {
        "schema_version": K.SCHEMA_VERSION,
        "config": asdict(cfg),
        "channels": K.CHANNELS,
        "modality_channels": K.MODALITY_CHANNELS,
        "horizons_min": list(K.ALL_HORIZONS_MIN),
        "primary_horizons_min": list(K.PRIMARY_HORIZONS_MIN),
        "grid_min": K.GRID_MIN,
        "static_numeric": K.STATIC_NUMERIC,
        "static_categorical": K.STATIC_CATEGORICAL,
        "levels": {k: [name for name, _ in sorted(v.items(), key=lambda kv: kv[1])]
                   for k, v in vocab.items() if k != "id"},
        "n_rows": int(n),
        "n_anchors": int(len(anchors)),
        "n_subjects": int(len(subj_df)),
        "excluded_columns": K.EXCLUDED_COLUMNS,
        "stats": stats,
        "build_seconds": round(time.time() - t_start, 1),
        "provenance": provenance(),
    }
    with open(out / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)

    LOG.info(f"store written to {out}: {n:,} rows x {n_ch} channels, "
             f"{len(anchors):,} anchors, {len(subj_df):,} subjects "
             f"({time.time() - t_start:.0f}s)")
    return out


def _subject_code(subject_keys: np.ndarray) -> np.ndarray:
    """Dense int32 codes for the packed (cohort, id) subject keys."""
    _, codes = np.unique(subject_keys, return_inverse=True)
    return codes.astype(np.int32)


def _select_anchors(cfg: PrepareConfig, starts: np.ndarray, ends: np.ndarray,
                    cgm_observed: np.ndarray, is_test: np.ndarray,
                    stats: Dict[str, object]) -> pd.DataFrame:
    """Which lattice rows may serve as "now".

    Filters, in order (each count is recorded):
      1. at least ``min_history_steps`` rows of history inside the same segment;
      2. the anchor row itself holds a real CGM reading -- you cannot forecast
         from an imputed present;
      3. no CGM gap longer than ``max_cgm_gap_min`` anywhere in that history;
      4. stride, on position within the segment so the phase does not drift.
    """
    max_gap_steps = cfg.max_cgm_gap_min // K.GRID_MIN
    L = cfg.min_history_steps
    rows_list: List[np.ndarray] = []
    segs_list: List[np.ndarray] = []
    pos_list: List[np.ndarray] = []
    counts = {"history": 0, "real_cgm": 0, "clean_history": 0, "stride": 0}

    for si, (s, e) in enumerate(zip(starts, ends)):
        s, e = int(s), int(e)
        length = e - s
        if length < L:
            continue
        pos = np.arange(L - 1, length, dtype=np.int64)
        counts["history"] += pos.size

        obs = cgm_observed[s:e]
        ok = obs[pos]
        counts["real_cgm"] += int(ok.sum())

        if cfg.require_clean_history:
            # A window is clean iff no run of missing CGM longer than max_gap_steps
            # starts inside it. Marker: the first sample whose "steps since a real
            # reading" exceeds the allowance. Then compare against the window start.
            since = steps_since(obs, cap=length + 1)
            violation = since > max_gap_steps
            last_viol = last_true_index(violation)          # per position in segment
            ok &= last_viol[pos] < (pos - L + 1)
            counts["clean_history"] += int(ok.sum())

        pos = pos[ok]
        pos = pos[(pos % cfg.anchor_stride) == 0]
        counts["stride"] += pos.size
        if pos.size:
            rows_list.append(pos + s)
            segs_list.append(np.full(pos.size, si, dtype=np.int32))
            pos_list.append(pos.astype(np.int32))

    if not rows_list:
        raise RuntimeError("no anchors survived filtering")
    df = pd.DataFrame({
        "row": np.concatenate(rows_list),
        "seg": np.concatenate(segs_list),
        "pos_in_seg": np.concatenate(pos_list),
    })
    stats["anchor_filter_counts"] = counts
    LOG.info(f"anchors: {counts['history']:,} with history -> "
             f"{counts['real_cgm']:,} real CGM -> {counts['clean_history']:,} clean "
             f"-> {len(df):,} after stride {cfg.anchor_stride}")
    return df


def _build_targets(rows: np.ndarray, cgm_raw: np.ndarray, cgm_observed: np.ndarray,
                   is_test: np.ndarray, seg_end: np.ndarray) -> np.ndarray:
    """CGM at t+h for each anchor and horizon; NaN where the label is unusable.

    A target is valid only if it is (a) inside the anchor's segment, (b) a real
    reading rather than an interpolated one, and (c) on the same side of the
    official train/test boundary as its anchor. (c) matters because 166-169
    subjects are split *temporally* across ``is_test``: without it, a training
    anchor near the boundary would be supervised by a test-period reading.
    """
    out = np.full((rows.size, len(K.HORIZON_STEPS)), np.nan, dtype=np.float32)
    for j, k in enumerate(K.HORIZON_STEPS):
        tgt = rows + k
        inside = tgt < seg_end[rows]
        tgt_safe = np.where(inside, tgt, 0)
        valid = inside & cgm_observed[tgt_safe] & (is_test[tgt_safe] == is_test[rows])
        out[valid, j] = cgm_raw[tgt_safe[valid]]
    return out


def _subject_table(data: Dict[str, np.ndarray], code: np.ndarray,
                   vocab: Dict[str, Dict[str, int]]) -> Tuple[pd.DataFrame, np.ndarray]:
    """One row per subject: keys, official split membership, static covariates.

    Statics are subject-constant by definition, so each one is the subject's first
    *observed* value; a categorical takes the highest observed code, which is any
    recorded level (-1 means the subject never recorded it). Height is feet ->
    metres and weight pounds -> kilograms: the corpus mixes imperial and metric
    without declaring it, and the raw columns are dropped afterwards so nothing
    downstream can read the ambiguous units.

    Aggregation is by boundary reduction over the subject-sorted arrays rather
    than a groupby. With 154.8M rows and ~1,900 subjects, a row-wise groupby is
    the single slowest step in the build for information that is one value per
    patient; ``reduceat`` over the subject boundaries is the same answer in one
    linear pass.
    """
    n = code.size
    if np.any(np.diff(code) < 0):
        raise RuntimeError("subject codes are not sorted; the (subject, time) "
                           "ordering step must run before the subject table")
    starts = np.flatnonzero(np.diff(code, prepend=np.int32(-1))).astype(np.int64)
    n_subj = starts.size
    counts = np.diff(np.append(starts, n))

    def first_observed(arr: np.ndarray) -> np.ndarray:
        """First finite value per subject; NaN where the subject never records one."""
        idx = np.where(np.isfinite(arr), np.arange(n, dtype=np.int64), n)
        pick = np.minimum.reduceat(idx, starts)
        out = np.full(n_subj, np.nan, dtype=np.float64)
        ok = pick < n
        out[ok] = arr[pick[ok]]
        return out

    def any_true(flags: np.ndarray) -> np.ndarray:
        return np.maximum.reduceat(flags.astype(np.uint8), starts).astype(bool)

    tbl = pd.DataFrame({
        "subject": np.arange(n_subj, dtype=np.int32),
        "cohort": data["cohort"][starts],
        "pid": data["pid"][starts],
        "n_rows": counts,
        "any_train": any_true(~data["is_test"]),
        "any_test": any_true(data["is_test"]),
        "known_patient": any_true(data["known"]),
    })
    num = {c: first_observed(data[c]) for c in
           ("age", "age_of_diagnosis", "height", "weight")}
    gender = np.maximum.reduceat(data["gender"], starts)      # -1 = unobserved
    modality = np.maximum.reduceat(data["modality"], starts)

    height_m = num["height"] * K.FEET_TO_M
    weight_kg = num["weight"] * K.LB_TO_KG
    with np.errstate(invalid="ignore", divide="ignore"):
        bmi = weight_kg / np.square(height_m)
    bmi[(bmi < K.BMI_VALID[0]) | (bmi > K.BMI_VALID[1])] = np.nan
    duration = num["age"] - num["age_of_diagnosis"]
    duration[duration < 0] = np.nan

    numeric = np.stack([num["age"], num["age_of_diagnosis"], duration,
                        height_m, weight_kg, bmi], axis=1).astype(np.float32)
    categorical = np.stack([gender, modality, tbl["cohort"].to_numpy()],
                           axis=1).astype(np.float32)
    statics = np.concatenate([numeric, categorical], axis=1).astype(np.float32)

    for col, arr in zip(("age", "age_of_diagnosis", "diabetes_duration",
                         "height_m", "weight_kg", "bmi"), numeric.T):
        tbl[col] = arr
    tbl["gender"] = gender
    tbl["modality"] = modality
    return tbl, statics


# ----------------------------------------------------------------------- CLI
def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    d = PrepareConfig()
    p.add_argument("--corpus", default=d.corpus)
    p.add_argument("--out", default=d.out)
    p.add_argument("--max-cgm-gap-min", type=int, default=d.max_cgm_gap_min)
    p.add_argument("--interp-gap-min", type=int, default=d.interp_gap_min)
    p.add_argument("--min-segment-min", type=int, default=d.min_segment_min)
    p.add_argument("--min-history-steps", type=int, default=d.min_history_steps)
    p.add_argument("--anchor-stride", type=int, default=d.anchor_stride)
    p.add_argument("--allow-dirty-history", action="store_true",
                   help="do not require a gap-free CGM history in the window")
    p.add_argument("--row-groups", type=int, default=None,
                   help="read only the first N parquet row groups (smoke builds)")
    p.add_argument("--max-subjects", type=int, default=None)
    p.add_argument("--cohorts", default=None, help="comma-separated source_file values")
    p.add_argument("--subject-seed", type=int, default=d.subject_seed)
    a = p.parse_args(argv)

    cfg = PrepareConfig(
        corpus=a.corpus, out=a.out, max_cgm_gap_min=a.max_cgm_gap_min,
        interp_gap_min=a.interp_gap_min, min_segment_min=a.min_segment_min,
        min_history_steps=a.min_history_steps, anchor_stride=a.anchor_stride,
        require_clean_history=not a.allow_dirty_history, row_groups=a.row_groups,
        max_subjects=a.max_subjects,
        cohorts=a.cohorts.split(",") if a.cohorts else None,
        subject_seed=a.subject_seed,
    )
    build_store(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
