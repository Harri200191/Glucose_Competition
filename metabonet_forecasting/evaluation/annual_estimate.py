"""What this model would score on the annual holdout, with honest error bars.

    python -m metabonet_forecasting.evaluation.annual_estimate runs/<r>
    python -m metabonet_forecasting.evaluation.annual_estimate runs/<r> --strict
    python -m metabonet_forecasting.evaluation.annual_estimate runs/<r> \
        --strict --freeze data/internal_holdout_strict.parquet

The internal holdout scores every anchor its 89 patients have -- ~216,000 of
them. The annual competition scores 2,228: about 25 per patient. Both estimate
the same quantity; only one of them has the annual set's *noise*.

Two numbers per horizon, and they answer different questions:

``full``          zone A over every anchor. Low variance, so this is what you
                  compare two models on.
``annual-like``   the distribution of zone A over repeated competition-shaped
                  draws. Its spread is what a single 2,228-point scoring run
                  will actually do to you.

A gap that is real in ``full`` but sits inside the ``annual-like`` interval is
real and not bankable on one draw. Both facts matter and neither number says
both.

Three things make a draw "competition-shaped", all on by default under
``--strict``:

* **without replacement.** 25 distinct anchors, not 25 draws from a hat with
  replacement -- the latter inflates variance and is not what the template does.
* **matched patient counts.** The annual template is not a flat 25 per patient:
  it runs 1 to 59, median 24, mean 25.0. Each draw resamples that empirical
  distribution rather than assuming the mean.
* **matched context quality** (``--context-match``). The annual anchors sit on
  gappier history than the corpus at large -- mean CGM observed fraction 0.968
  over the 288-step window against 0.992 corpus-wide, and 2.9% perfectly clean
  windows against 60%. Selecting IOBP2 already matches this to ~0.2 pp, so this
  is a refinement rather than a rescue, but it removes the residual.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .metrics import dts_zones

CTX = 288
DECILES = 10


def horizons_of(df: pd.DataFrame) -> List[int]:
    return sorted(int(c.rsplit("_", 1)[-1]) for c in df.columns
                  if c.startswith("y_true_"))


def _zone_a_flags(df: pd.DataFrame, h: int):
    y = df[f"y_true_{h}"].to_numpy(float)
    p = df[f"y_pred_{h}"].to_numpy(float)
    ok = np.isfinite(y) & np.isfinite(p) & (y > 0)
    flag = np.zeros(len(df), dtype=bool)
    flag[ok] = dts_zones(y[ok], p[ok]) == "A"
    return flag, ok


# ------------------------------------------------------- competition shape
def annual_counts(annual_store: Path) -> np.ndarray:
    """Anchors per patient in the real annual template."""
    a = pd.read_parquet(Path(annual_store) / "anchors.parquet", columns=["subject"])
    return a.groupby("subject").size().to_numpy()


def context_observed(store, anchor_ids: np.ndarray, context: int = CTX,
                     cache: Optional[Path] = None) -> np.ndarray:
    """Fraction of each anchor's context window that is a real CGM reading.

    Cached: it is ~60 M memmap reads for a full holdout and never changes for a
    fixed store and anchor set.
    """
    if cache is not None and Path(cache).exists():
        cached = np.load(cache)
        if cached.shape[0] == anchor_ids.shape[0]:
            return cached
    mid = int(store.channel_ids(["cgm_mask"])[0])
    rows = store.anchors["row"].to_numpy()[anchor_ids]
    out = np.empty(rows.size, dtype=np.float32)
    for i, r in enumerate(rows):
        out[i] = np.asarray(store.channels[r - context + 1:r + 1, mid],
                            dtype=np.float32).mean()
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, out)
    return out


def _decile_weights(pool_obs: np.ndarray, edges: np.ndarray,
                    target: np.ndarray) -> np.ndarray:
    """Per-anchor weights that push a draw's context-quality mix towards
    ``target``. A decile the patient has none of simply cannot be matched; the
    rest absorb it through normalisation."""
    band = np.clip(np.searchsorted(edges, pool_obs, side="right") - 1,
                   0, len(target) - 1)
    have = np.bincount(band, minlength=len(target)).astype(float)
    w = np.zeros(pool_obs.size, dtype=float)
    for b in range(len(target)):
        sel = band == b
        if have[b] > 0:
            w[sel] = target[b] / have[b]
    if not w.any():
        w[:] = 1.0
    return w


def _draw(groups: List[np.ndarray], counts: np.ndarray, rng: np.random.Generator,
          weights: Optional[List[np.ndarray]] = None) -> np.ndarray:
    """One competition-shaped draw: k_i distinct anchors from patient i.

    Weighted sampling without replacement via the Gumbel top-k trick -- one
    vectorised pass per patient instead of a permutation.
    """
    picks = []
    for i, g in enumerate(groups):
        k = int(min(counts[i], g.size))
        if k <= 0:
            continue
        if k == g.size:
            picks.append(g)
            continue
        if weights is None:
            picks.append(rng.choice(g, size=k, replace=False))
        else:
            key = np.log(np.maximum(weights[i], 1e-12)) + rng.gumbel(size=g.size)
            picks.append(g[np.argpartition(-key, k - 1)[:k]])
    return np.concatenate(picks) if picks else np.zeros(0, dtype=int)


def resample(df: pd.DataFrame, horizon: int, groups: List[np.ndarray],
             count_pool: np.ndarray, n_boot: int, rng: np.random.Generator,
             weights: Optional[List[np.ndarray]] = None,
             fixed: Optional[int] = None) -> Dict[str, float]:
    flag, _ = _zone_a_flags(df, horizon)
    scores = np.empty(n_boot, dtype=float)
    sizes = np.empty(n_boot, dtype=int)
    for b in range(n_boot):
        counts = (np.full(len(groups), fixed) if fixed is not None
                  else rng.choice(count_pool, size=len(groups), replace=True))
        idx = _draw(groups, counts, rng, weights)
        sizes[b] = idx.size
        scores[b] = flag[idx].mean() * 100.0 if idx.size else np.nan
    lo, hi = np.nanpercentile(scores, [2.5, 97.5])
    return {"mean": float(np.nanmean(scores)), "sd": float(np.nanstd(scores)),
            "lo95": float(lo), "hi95": float(hi),
            "n_per_draw": int(np.median(sizes))}


def build_groups(df: pd.DataFrame, obs: Optional[np.ndarray],
                 edges: Optional[np.ndarray], target: Optional[np.ndarray]):
    """Scorable anchor indices per patient, plus optional context weights."""
    ok = np.zeros(len(df), dtype=bool)
    for h in horizons_of(df):
        _, o = _zone_a_flags(df, h)
        ok |= o
    subj = df["subject"].to_numpy()
    order = np.argsort(subj, kind="mergesort")
    bounds = np.flatnonzero(np.diff(subj[order])) + 1
    groups, weights = [], []
    for g in np.split(order, bounds):
        g = g[ok[g]]
        if not g.size:
            continue
        groups.append(g)
        if obs is not None:
            weights.append(_decile_weights(obs[g], edges, target))
    return groups, (weights if obs is not None else None)


def report(df: pd.DataFrame, groups, count_pool, n_boot, seed, weights=None,
           fixed=None) -> pd.DataFrame:
    rows = []
    for h in horizons_of(df):
        flag, ok = _zone_a_flags(df, h)
        full = float(flag[ok].mean() * 100) if ok.any() else float("nan")
        r = resample(df, h, groups, count_pool, n_boot,
                     np.random.default_rng(seed), weights, fixed)
        rows.append({"horizon": h, "n_full": int(ok.sum()),
                     "full_dts_A": round(full, 2),
                     "annual_like_mean": round(r["mean"], 2),
                     "annual_like_sd": round(r["sd"], 2),
                     "lo95": round(r["lo95"], 2), "hi95": round(r["hi95"], 2),
                     "n_per_draw": r["n_per_draw"]})
    return pd.DataFrame(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--split", default="internal_holdout")
    ap.add_argument("--strict", action="store_true",
                    help="match the annual per-patient anchor-count distribution "
                         "instead of a flat --per-subject, and turn on "
                         "--context-match")
    ap.add_argument("--per-subject", type=int, default=25,
                    help="flat anchors per patient when --strict is off")
    ap.add_argument("--context-match", action="store_true",
                    help="also match the annual context-quality distribution")
    ap.add_argument("--store", default=None,
                    help="store for --context-match; default: the run's config")
    ap.add_argument("--annual-store", default="artifacts/store/submission_annual")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--freeze", type=Path, default=None,
                    help="also write ONE competition-shaped draw to this parquet, "
                         "as a fixed strict evaluation set")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    pred = a.run / f"predictions_{a.split}.parquet"
    if not pred.exists():
        raise SystemExit(f"{pred} not found")
    df = pd.read_parquet(pred)

    annual_root = Path(a.annual_store)
    counts = (annual_counts(annual_root) if annual_root.exists()
              else np.array([a.per_subject]))
    fixed = None if a.strict else a.per_subject

    obs = edges = target = None
    if a.strict or a.context_match:
        if not annual_root.exists():
            print(f"no annual store at {annual_root}; skipping context match")
        else:
            root = a.store
            if root is None:
                cfg = a.run / "config.resolved.yaml"
                if cfg.exists():
                    import yaml
                    root = (yaml.safe_load(cfg.read_text()) or {}).get(
                        "data", {}).get("store")
            if root is None:
                print("no store for --context-match; skipping")
            else:
                from ..data.store import SequenceStore
                ann = SequenceStore(annual_root)
                a_obs = context_observed(
                    ann, np.arange(len(ann.anchors)),
                    cache=annual_root / "ctx_observed.npy")
                edges = np.unique(np.percentile(
                    a_obs, np.linspace(0, 100, DECILES + 1)))
                target = np.histogram(a_obs, edges)[0].astype(float)
                target /= target.sum()
                store = SequenceStore(root)
                obs = context_observed(
                    store, df["anchor"].to_numpy(),
                    cache=a.run / f"ctx_observed_{a.split}.npy")
                print(f"context match on: annual mean {a_obs.mean():.4f}  "
                      f"holdout mean {obs.mean():.4f}")

    groups, weights = build_groups(df, obs, edges, target)
    tab = report(df, groups, counts, a.boot, a.seed, weights, fixed)

    mode = ("strict: annual count distribution" if a.strict
            else f"flat {a.per_subject}/patient")
    if weights is not None:
        mode += " + context-matched"
    print(f"{a.run.name} [{a.split}]  {len(groups)} patients, {len(df):,} anchors")
    print(f"draw mode: {mode}, {a.boot} draws, without replacement")
    print(tab.to_string(index=False))
    mean_full = tab["full_dts_A"].mean()
    mean_like = tab["annual_like_mean"].mean()
    half = (tab["hi95"] - tab["lo95"]).mean() / 2
    print(f"\nmean DTS-A over horizons:  full {mean_full:.2f}%   "
          f"annual-like {mean_like:.2f}% (+/- {half:.2f} at 95% on one draw)")

    out = a.out or (a.run / f"annual_estimate_{a.split}.csv")
    tab.to_csv(out, index=False)
    out.with_suffix(".json").write_text(json.dumps(
        {"run": a.run.name, "split": a.split, "mode": mode, "n_boot": a.boot,
         "mean_full_dts_A": round(float(mean_full), 3),
         "mean_annual_like_dts_A": round(float(mean_like), 3),
         "half_width_95": round(float(half), 3)}, indent=1))
    print(f"wrote {out}")

    if a.freeze:
        rng = np.random.default_rng(a.seed)
        k = (rng.choice(counts, size=len(groups), replace=True)
             if a.strict else np.full(len(groups), a.per_subject))
        idx = np.sort(_draw(groups, k, rng, weights))
        a.freeze.parent.mkdir(parents=True, exist_ok=True)
        df.iloc[idx].to_parquet(a.freeze, index=False)
        print(f"froze one competition-shaped draw: {a.freeze} "
              f"({len(idx):,} anchors, {df.iloc[idx]['subject'].nunique()} patients)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
