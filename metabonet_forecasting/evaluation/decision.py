"""DTS-grid-optimal decision layer.

The leaderboard's headline metric is the percentage of predictions in DTS error
grid **zone A** — an asymmetric, clinically-shaped region. The RMSE-optimal
point (the conditional mean) is *not* the zone-A-optimal point: around the hypo
boundary the A-region is narrow and skewed, and a mean pulled up by the
right tail steps out of it. Given a predictive distribution we can do better by
construction.

The grid's zone-A condition ``b_low(y) <= p <= b_up(y)`` inverts to an interval
of true values for each candidate prediction ``p``:

    y in [ L(p), U(p) ],   L(p) = -inf          if p <= 60  else 50 + (p-60)/1.2
                           U(p) = 62.5          if p <  50  else 62.5 + (p-50)/0.8

(the constants come straight from ``toolkit/metrics.py``). So the best point is

    argmax_p  P( L(p) <= y <= U(p) )  -  lam * E[(y - p)^2] / 3600

with the probability read off the model's quantiles and ``lam`` trading a
little RMSE for zone-A mass (lam=0 maximises zone A alone; large lam recovers
the conditional mean). Tune ``lam`` on validation predictions, never on test.

Two sources of quantiles:

* a run trained with ``model.head.quantiles`` + the hybrid loss — its
  ``predictions_*.parquet`` carries ``y_qXX_h`` columns;
* any point-only run, via ``ResidualQuantiles``: conditional residual quantiles
  fitted on *validation* predictions in bins of the predicted value.

CLI (writes a sibling run directory ``<run>_dts`` and scores it):

    python -m metabonet_forecasting.evaluation.decision runs/my_run \
        --lam 0.05 [--val-run runs/my_run] [--grid 40 420 2]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

HORIZONS = [30, 60, 90, 120]


# ------------------------------------------------------------------- geometry
def zone_a_bounds(p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """For candidate predictions ``p``: the [L, U] interval of true values for
    which ``p`` falls in DTS zone A. Vectorised; L can be -inf."""
    p = np.asarray(p, dtype=float)
    L = np.where(p <= 60.0, -np.inf, 50.0 + (p - 60.0) / 1.2)
    U = np.where(p < 50.0, 62.5, 62.5 + (p - 50.0) / 0.8)
    return L, U


# ------------------------------------------------- conditional residual model
class ResidualQuantiles:
    """Quantiles of the truth conditional on the predicted value, fitted on
    validation residuals in prediction-value bins."""

    def __init__(self, levels: Sequence[float], n_bins: int = 24,
                 min_bin: int = 500) -> None:
        self.levels = np.asarray(levels, dtype=float)
        self.n_bins = n_bins
        self.min_bin = min_bin
        self.edges: np.ndarray = np.zeros(0)
        self.table: np.ndarray = np.zeros((0, 0))    # [n_bins, n_levels] offsets

    def fit(self, pred: np.ndarray, true: np.ndarray) -> "ResidualQuantiles":
        ok = np.isfinite(pred) & np.isfinite(true)
        pred, true = pred[ok], true[ok]
        qs = np.linspace(0, 1, self.n_bins + 1)
        self.edges = np.unique(np.quantile(pred, qs))
        n_bins = len(self.edges) - 1
        self.table = np.zeros((n_bins, self.levels.size), dtype=float)
        which = np.clip(np.searchsorted(self.edges, pred, side="right") - 1,
                        0, n_bins - 1)
        resid = true - pred
        global_q = np.quantile(resid, self.levels)
        for b in range(n_bins):
            r = resid[which == b]
            self.table[b] = np.quantile(r, self.levels) if r.size >= self.min_bin \
                else global_q
        return self

    def transform(self, pred: np.ndarray) -> np.ndarray:
        """[N] point predictions -> [N, n_levels] conditional quantiles of y."""
        n_bins = len(self.edges) - 1
        which = np.clip(np.searchsorted(self.edges, pred, side="right") - 1,
                        0, n_bins - 1)
        return pred[:, None] + self.table[which]


# ------------------------------------------------------------------- decision
def dts_optimal_point(quant_vals: np.ndarray, levels: Sequence[float],
                      lam: float = 0.05,
                      grid: Tuple[float, float, float] = (40.0, 420.0, 2.0),
                      ) -> np.ndarray:
    """[N, Q] quantile values (mg/dL) -> [N] zone-A-optimal point predictions.

    The distribution is approximated by atoms at the quantile values weighted by
    the level gaps (tails get the outermost levels' mass). Every candidate on
    the grid is scored as P(zone A) − lam·E[(y−p)²]/3600 and the argmax wins.
    """
    levels = np.asarray(levels, dtype=float)
    q = np.asarray(quant_vals, dtype=float)                    # [N, Q]
    # Atom weights: half the gap on each side, tails absorb the rest.
    inner = np.diff(levels)
    w = np.empty_like(levels)
    w[0] = levels[0] + inner[0] / 2
    w[-1] = (1 - levels[-1]) + inner[-1] / 2
    w[1:-1] = (inner[:-1] + inner[1:]) / 2                     # [Q]

    candidates = np.arange(*grid, dtype=float)                 # [P]
    L, U = zone_a_bounds(candidates)                           # [P]
    # in_zone[n, p, k] = L[p] <= q[n, k] <= U[p]  -> P(A) = sum_k w_k * in_zone
    # chunk over n to bound memory.
    best = np.empty(q.shape[0], dtype=float)
    for lo in range(0, q.shape[0], 8192):
        hi = min(lo + 8192, q.shape[0])
        block = q[lo:hi]                                       # [n, Q]
        inz = (block[:, None, :] >= L[None, :, None]) & \
              (block[:, None, :] <= U[None, :, None])
        p_a = np.einsum("npk,k->np", inz.astype(np.float32), w.astype(np.float32))
        if lam:
            se = (block[:, None, :] - candidates[None, :, None]) ** 2
            mse = np.einsum("npk,k->np", se.astype(np.float32),
                            w.astype(np.float32)) / 3600.0
            p_a = p_a - lam * mse
        best[lo:hi] = candidates[np.argmax(p_a, axis=1)]
    return best


# ------------------------------------------------------------------------ CLI
def quantile_columns(df: pd.DataFrame, horizon: int) -> Tuple[List[str], List[float]]:
    cols, levels = [], []
    for c in df.columns:
        if c.startswith("y_q") and c.endswith(f"_{horizon}"):
            cols.append(c)
            levels.append(int(c[3:5]) / 100.0)
    order = np.argsort(levels)
    return [cols[i] for i in order], [levels[i] for i in order]


def apply_to_run(run_dir: Path, lam: float, val_run: Optional[Path],
                 grid: Tuple[float, float, float], suffix: str = "_dts",
                 split: str = "test") -> Path:
    out_dir = run_dir.parent / (run_dir.name + suffix)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(run_dir / f"predictions_{split}.parquet")

    levels_used: Dict[int, int] = {}
    for h in HORIZONS:
        if f"y_pred_{h}" not in df.columns:
            continue
        qcols, levels = quantile_columns(df, h)
        pred = df[f"y_pred_{h}"].to_numpy()
        ok = np.isfinite(pred)
        if qcols:                                   # head quantiles available
            quant = df.loc[ok, qcols].to_numpy()
        else:                                       # residual quantiles from val
            if val_run is None:
                raise SystemExit(f"run has no quantile columns for h={h}; "
                                 f"pass --val-run to fit residual quantiles")
            vdf = pd.read_parquet(Path(val_run) / "predictions_val.parquet")
            levels = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
                      0.6, 0.7, 0.8, 0.9, 0.95, 0.98]
            rq = ResidualQuantiles(levels).fit(
                vdf[f"y_pred_{h}"].to_numpy(), vdf[f"y_true_{h}"].to_numpy())
            quant = rq.transform(pred[ok])
        new = pred.copy()
        new[ok] = dts_optimal_point(quant, levels, lam=lam, grid=grid)
        df[f"y_pred_{h}"] = new
        levels_used[h] = len(levels)

    df.to_parquet(out_dir / f"predictions_{split}.parquet", index=False)
    for aux in ("config.resolved.yaml", "manifest.json"):
        src = run_dir / aux
        if src.exists() and not (out_dir / aux).exists():
            shutil.copy(src, out_dir / aux)
    (out_dir / "decision.json").write_text(json.dumps(
        {"source_run": run_dir.name, "lam": lam, "grid": list(grid),
         "quantile_levels_per_horizon": levels_used}, indent=1))
    return out_dir


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--lam", type=float, default=0.05)
    ap.add_argument("--val-run", type=Path, default=None,
                    help="run directory with predictions_val.parquet, for "
                         "point-only models")
    ap.add_argument("--grid", nargs=3, type=float, default=[40.0, 420.0, 2.0],
                    metavar=("LO", "HI", "STEP"))
    ap.add_argument("--suffix", default="_dts")
    ap.add_argument("--split", default="test",
                    help="which predictions_<split>.parquet to re-decide")
    ap.add_argument("--no-eval", action="store_true")
    a = ap.parse_args(argv)

    out_dir = apply_to_run(a.run, a.lam, a.val_run, tuple(a.grid), a.suffix,
                           split=a.split)
    print(f"wrote {out_dir}")
    if not a.no_eval:
        from .evaluate import score_run
        metrics = score_run(out_dir, split=a.split)
        for h in metrics["horizons_min"]:
            o = metrics["strata"]["all"][str(h)]["overall"]
            print(f"h={h:>3}  RMSE {o['rmse']:6.2f}  DTS-A {o['dts_A']:5.2f}%  "
                  f"DTS-AB {o.get('dts_AB', float('nan')):5.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
