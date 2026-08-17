"""Figures.

    python -m metabonet_forecasting.evaluation.plots results/all_runs.csv --out figures

Four figures, one per question the plan asks:

``context_curve``   performance vs context length per backbone and horizon (E3)
``fusion_ladder``   raw concat -> encoders -> gated -> FiLM, per backbone (E4/E5)
``stratum_heatmap`` error by glycaemic range and event context (E6)
``error_grid``      predictions against reference with the DTS borders drawn

Plain matplotlib, no seaborn, no style sheets: a figure that needs a theme to be
readable is a figure that will not survive a journal's conversion pipeline.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

MARKERS = ["o", "s", "^", "D", "v", "P", "X"]


def _pyplot():
    """Import matplotlib on demand, with an actionable message when it is broken.

    Deliberately not a module-level import: matplotlib built against numpy 1.x
    raises ``numpy.core.multiarray failed to import`` under numpy 2, and that
    should not stop anyone from importing the evaluation package to compute
    metrics. Figures are optional; numbers are not.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as exc:
        raise RuntimeError(
            f"matplotlib is unusable in this environment ({exc}). "
            "Metrics and tables are unaffected. To draw figures: "
            "pip install -U 'matplotlib>=3.8'  (3.5 is not numpy-2 compatible)"
        ) from exc


def _overall(df: pd.DataFrame, stratum: str = "all") -> pd.DataFrame:
    return df[(df["stratum"] == stratum) & (df["slice"] == "overall")]


def context_curve(df: pd.DataFrame, out: Path, metric: str = "rmse",
                  stratum: str = "all") -> Optional[Path]:
    plt = _pyplot()
    sel = _overall(df, stratum).dropna(subset=["context"])
    if sel.empty or sel["context"].nunique() < 2:
        return None
    horizons = sorted(sel["horizon"].unique())
    fig, axes = plt.subplots(1, len(horizons), figsize=(4.2 * len(horizons), 3.6),
                             sharey=False, squeeze=False)
    for ax, h in zip(axes[0], horizons):
        block = sel[sel["horizon"] == h]
        for i, (backbone, g) in enumerate(block.groupby("backbone")):
            agg = g.groupby("context")[metric].agg(["mean", "std", "count"]).reset_index()
            ax.errorbar(agg["context"] * 5 / 60.0, agg["mean"],
                        yerr=agg["std"].fillna(0.0), marker=MARKERS[i % len(MARKERS)],
                        capsize=3, label=str(backbone))
        ax.set_title(f"{h} min ahead")
        ax.set_xlabel("context (hours)")
        ax.set_ylabel(metric.upper() + (" (mg/dL)" if metric in ("rmse", "mae") else " (%)"))
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.suptitle("Does longer history help, and for which backbone?")
    fig.tight_layout()
    path = out / f"context_curve_{metric}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fusion_ladder(df: pd.DataFrame, out: Path, metric: str = "rmse",
                  horizon: int = 60, stratum: str = "all") -> Optional[Path]:
    plt = _pyplot()
    sel = _overall(df, stratum)
    sel = sel[sel["horizon"] == horizon].dropna(subset=["fusion"])
    if sel.empty:
        return None
    order = [f for f in ("raw_concat", "concat", "gated", "film", "native")
             if f in set(sel["fusion"])]
    backbones = sorted(sel["backbone"].dropna().unique())
    width = 0.8 / max(len(backbones), 1)
    fig, ax = plt.subplots(figsize=(1.6 * len(order) + 3, 4))
    for i, backbone in enumerate(backbones):
        vals, errs = [], []
        for fusion in order:
            g = sel[(sel["backbone"] == backbone) & (sel["fusion"] == fusion)][metric]
            vals.append(g.mean() if len(g) else np.nan)
            errs.append(g.std() if len(g) > 1 else 0.0)
        ax.bar(np.arange(len(order)) + i * width, vals, width, yerr=errs,
               capsize=3, label=str(backbone))
    ax.set_xticks(np.arange(len(order)) + width * (len(backbones) - 1) / 2)
    ax.set_xticklabels(order, rotation=15)
    ax.set_ylabel(f"{metric.upper()} at {horizon} min")
    ax.set_title("Fusion ladder: does modality-specific encoding and conditioning help?")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out / f"fusion_ladder_{metric}_h{horizon}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def stratum_heatmap(df: pd.DataFrame, out: Path, run: Optional[str] = None,
                    metric: str = "rmse", stratum: str = "all") -> Optional[Path]:
    plt = _pyplot()
    sel = df[(df["stratum"] == stratum) & (df["slice"] != "overall")]
    if run:
        sel = sel[sel["run"] == run]
    elif sel["run"].nunique() > 1:
        sel = sel[sel["run"] == sorted(sel["run"].unique())[0]]
    if sel.empty:
        return None
    pivot = sel.pivot_table(index="slice", columns="horizon", values=metric, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(1.4 * len(pivot.columns) + 4, 0.36 * len(pivot) + 2))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c} min" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.iloc[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", color="w", fontsize=7)
    fig.colorbar(im, ax=ax, label=metric.upper())
    ax.set_title(f"{metric.upper()} by clinical stratum" + (f" -- {run}" if run else ""))
    fig.tight_layout()
    path = out / f"stratum_heatmap_{metric}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def error_grid(predictions: pd.DataFrame, out: Path, horizon: int = 60,
               max_points: int = 20000, seed: int = 0) -> Path:
    """Scatter with the DTS A/B borders drawn, on a subsample of anchors."""
    plt = _pyplot()
    y = predictions[f"y_true_{horizon}"].to_numpy(dtype=float)
    p = predictions[f"y_pred_{horizon}"].to_numpy(dtype=float)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    if y.size > max_points:
        idx = np.random.default_rng(seed).choice(y.size, max_points, replace=False)
        y, p = y[idx], p[idx]

    ref = np.linspace(0, 450, 500)
    ab_up = np.where(ref <= 50, 60.0, (540.0 / 450.0) * (ref - 50) + 60.0)
    ab_lo = np.where(ref <= 62.5, 0.0, (430.0 / 537.5) * (ref - 62.5) + 50.0)
    bc_up = np.where(ref <= 50, 86.5, (513.5 / 297.0) * (ref - 50) + 86.5)
    bc_lo = np.where(ref <= 97.5, 0.0, (257.0 / 502.5) * (ref - 97.5) + 50.0)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y, p, s=2, alpha=0.15, linewidths=0)
    ax.plot(ref, ref, "k--", lw=0.8)
    ax.plot(ref, ab_up, "g-", lw=1, label="DTS A|B")
    ax.plot(ref, ab_lo, "g-", lw=1)
    ax.plot(ref, bc_up, "orange", lw=1, label="DTS B|C")
    ax.plot(ref, bc_lo, "orange", lw=1)
    ax.set_xlim(0, 450); ax.set_ylim(0, 450)
    ax.set_xlabel("reference CGM (mg/dL)")
    ax.set_ylabel(f"forecast at +{horizon} min (mg/dL)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out / f"error_grid_h{horizon}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_csv", nargs="?", default="results/all_runs.csv")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--metric", default="rmse")
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--predictions", default=None,
                    help="a run's predictions_test.parquet, for the error grid")
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    made = []
    try:
        if Path(a.results_csv).exists():
            df = pd.read_csv(a.results_csv)
            for fn in (lambda: context_curve(df, out, a.metric),
                       lambda: fusion_ladder(df, out, a.metric, a.horizon),
                       lambda: stratum_heatmap(df, out, metric=a.metric)):
                path = fn()
                if path:
                    made.append(path)
        if a.predictions:
            made.append(error_grid(pd.read_parquet(a.predictions), out, a.horizon))
    except RuntimeError as exc:
        # A broken plotting install must not take a sweep down with it.
        print(exc)
        return 1
    print("\n".join(str(p) for p in made) or "nothing to plot")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
