"""Precompute the per-anchor feature matrix.

    python -m metabonet_forecasting.features.build \
        --store artifacts/store/main --context 288 \
        --anchors artifacts/features/mini_anchor_union.npy \
        --out artifacts/features/mini288

Writes into ``--out``:
    features.npy      float32 memmap [N, F], row i = anchor ``index[i]``
    index.npy         int64 [N], positional anchor ids, ascending
    lookup.npy        int32 [n_anchors], anchor id -> row in features.npy, -1 if absent
    columns.json      the F column names, in order
    manifest.json     store, context, counts, git provenance, timing

The lookup table is the contract: a dataset asking for an anchor that was never
built gets -1 and must raise, rather than silently reading someone else's row.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..common.logging import RunLogger
from ..data.store import SequenceStore
from .spec import build_columns, compute_block


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default="artifacts/store/main")
    ap.add_argument("--context", type=int, default=288)
    ap.add_argument("--anchors", default=None,
                    help="npy of positional anchor ids; omit to build every anchor "
                         "with enough history (large)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--families", default=None,
                    help="comma-separated subset of families; omit for all")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    log = RunLogger(out, name="features")
    st = SequenceStore(a.store)
    L = int(a.context)
    rows_all = st.anchors["row"].to_numpy()

    if a.anchors:
        index = np.sort(np.load(a.anchors).astype(np.int64))
    else:
        index = np.nonzero(rows_all >= L - 1)[0].astype(np.int64)
    # A window is store.channels[row-L+1 : row+1]; anything shorter cannot be built.
    keep = rows_all[index] >= L - 1
    if not keep.all():
        log.warning(f"dropping {(~keep).sum():,} anchors with < {L} steps of history")
        index = index[keep]

    fams = a.families.split(",") if a.families else None
    cols = build_columns(fams)
    N, F = len(index), len(cols)
    log.info(f"store={a.store} context={L} anchors={N:,} features={F} "
             f"-> {N * F * 4 / 1e9:.2f} GB")

    feats = np.lib.format.open_memmap(out / "features.npy", mode="w+",
                                      dtype=np.float32, shape=(N, F))
    rows = rows_all[index]
    off = np.arange(-L + 1, 1)
    t0 = time.time()
    n_bad = 0
    for lo in range(0, N, a.chunk):
        hi = min(lo + a.chunk, N)
        W = st.channels[rows[lo:hi, None] + off[None, :]]          # [n, L, C]
        subj = st.anchors["subject"].to_numpy()[index[lo:hi]]
        blk = compute_block(W, st.anchors, index[lo:hi], fams,
                            statics=st.statics[subj, :6])                # [n, F]
        bad = ~np.isfinite(blk)
        if bad.any():
            n_bad += int(bad.sum())
            blk = np.nan_to_num(blk, nan=0.0, posinf=0.0, neginf=0.0)
        feats[lo:hi] = blk.astype(np.float32)
        if lo and (lo // a.chunk) % 20 == 0:
            el = time.time() - t0
            log.info(f"  {hi:,}/{N:,}  {hi / el:,.0f} anchors/s  "
                     f"eta {(N - hi) / (hi / el) / 60:.1f} min")
    feats.flush()
    secs = time.time() - t0
    if n_bad:
        # Non-finite is expected where a modality is absent for a whole cohort;
        # it is zeroed AFTER standardisation stats are computed, not before, so a
        # constant column stays constant rather than becoming a fake signal.
        log.warning(f"{n_bad:,} non-finite cells zeroed "
                    f"({100 * n_bad / (N * F):.3f}% of the matrix)")

    np.save(out / "index.npy", index)
    lut = np.full(len(rows_all), -1, dtype=np.int32)
    lut[index] = np.arange(N, dtype=np.int32)
    np.save(out / "lookup.npy", lut)
    json.dump(cols, open(out / "columns.json", "w"), indent=1)
    json.dump({"store": a.store, "context": L, "n_anchors": int(N),
               "n_features": int(F), "families": fams or "all",
               "n_nonfinite": int(n_bad), "build_seconds": round(secs, 1)},
              open(out / "manifest.json", "w"), indent=1)
    log.info(f"done in {secs / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
