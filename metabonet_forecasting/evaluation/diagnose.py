"""Per-run error diagnostics: where a model is wrong, and under what conditions.

    python -m metabonet_forecasting.evaluation.diagnose runs/<name>
    python -m metabonet_forecasting.evaluation.diagnose runs/<name> --split internal_holdout
    python -m metabonet_forecasting.evaluation.diagnose runs/<name> --no-traces

``metrics.json`` says *how much* error there is. These figures say *where* it is,
which is the part that tells you what to change:

``error_grid``        every horizon on the DTS grid with all five zones drawn, so
                      a miss can be read as clinically harmless (B) or dangerous
                      (D/E) rather than just "large"
``zones_by_glucose``  zone mix across the reference range -- the plot that
                      answers "at what glucose does this fall out of zone A", and
                      the one worth looking at first
``error_profile``     signed bias and mean absolute error across the four
                      conditions known at forecast time: glycaemic band, trend,
                      time since the last bolus, hour of day. Bias matters more
                      than MAE here -- a model that is 20 mg/dL high in
                      hypoglycaemia is dangerous in a way that a symmetric
                      20 mg/dL error is not
``traces``            worked examples: real CGM history, what actually happened,
                      and what the model said would happen, for the worst misses
                      and a matched set of successes

Zones come from ``metrics.dts_zones``, which the test suite checks against the
organisers' scorer point for point; nothing here re-derives them. Only the border
*curves* are recomputed, for drawing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .metrics import dts_zones
from .plots import _pyplot

ZONE_COLOURS = {"A": "#2e7d32", "B": "#9ccc65", "C": "#fdd835",
                "D": "#fb8c00", "E": "#c62828"}
ZONES = ("A", "B", "C", "D", "E")
GRID_MIN = 5


# ------------------------------------------------------------------ geometry
def zone_borders(ref: np.ndarray) -> Dict[str, np.ndarray]:
    """The eight DTS border curves as a function of the reference value.

    Same constants as ``metrics.dts_zones``; kept here only because drawing
    needs the curves themselves rather than a per-point zone label.
    """
    return {
        "ab_up": np.where(ref <= 50, 60.0, (540.0 / 450.0) * (ref - 50) + 60.0),
        "bc_up": np.where(ref <= 50, 86.5, (513.5 / 297.0) * (ref - 50) + 86.5),
        "cd_up": np.where(ref <= 50, 124.0, (476.0 / 191.0) * (ref - 50) + 124.0),
        "de_up": np.where(ref <= 50, 179.0, (421.0 / 117.0) * (ref - 50) + 179.0),
        "ab_lo": np.where(ref <= 62.5, 0.0, (430.0 / 537.5) * (ref - 62.5) + 50.0),
        "bc_lo": np.where(ref <= 97.5, 0.0, (257.0 / 502.5) * (ref - 97.5) + 50.0),
        "cd_lo": np.where(ref <= 153.0, 0.0, (147.0 / 447.0) * (ref - 153.0) + 50.0),
        "de_lo": np.where(ref <= 238.0, 0.0, (76.0 / 362.0) * (ref - 238.0) + 50.0),
    }


def _shade_zones(ax, hi: float = 450.0) -> None:
    """Paint the zones widest-first so the narrow ones end up on top."""
    ref = np.linspace(0, hi, 900)
    b = zone_borders(ref)
    ax.fill_between(ref, 0, hi, color=ZONE_COLOURS["E"], alpha=.16, lw=0)
    for up, lo, z in (("de_up", "de_lo", "D"), ("cd_up", "cd_lo", "C"),
                      ("bc_up", "bc_lo", "B"), ("ab_up", "ab_lo", "A")):
        ax.fill_between(ref, b[lo], np.minimum(b[up], hi),
                        color=ZONE_COLOURS[z], alpha=.20, lw=0)
    for key, style in (("ab_up", "-"), ("ab_lo", "-"), ("bc_up", "--"),
                       ("bc_lo", "--"), ("cd_up", ":"), ("cd_lo", ":")):
        ax.plot(ref, b[key], style, color="#555", lw=.7)
    ax.plot(ref, ref, "k--", lw=.7, alpha=.6)


def horizons_of(df: pd.DataFrame) -> List[int]:
    return sorted(int(c.rsplit("_", 1)[-1]) for c in df.columns
                  if c.startswith("y_true_"))


def _pair(df: pd.DataFrame, h: int):
    y = df[f"y_true_{h}"].to_numpy(float)
    p = df[f"y_pred_{h}"].to_numpy(float)
    ok = np.isfinite(y) & np.isfinite(p) & (y > 0)
    return y, p, ok


# ------------------------------------------------------------------- figures
def error_grid(df: pd.DataFrame, out: Path, max_points: int = 25000,
               seed: int = 0) -> Path:
    """All horizons on the DTS grid, points coloured by the zone they land in."""
    plt = _pyplot()
    hz = horizons_of(df)
    ncol = min(len(hz), 2)
    nrow = int(np.ceil(len(hz) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.4 * ncol, 5.2 * nrow),
                             squeeze=False)
    rng = np.random.default_rng(seed)
    for k, h in enumerate(hz):
        ax = axes[k // ncol][k % ncol]
        y, p, ok = _pair(df, h)
        y, p = y[ok], p[ok]
        z = dts_zones(y, p)
        share = {zz: float(np.mean(z == zz) * 100) for zz in ZONES}
        _shade_zones(ax)
        idx = (rng.choice(y.size, max_points, replace=False)
               if y.size > max_points else np.arange(y.size))
        # Worst zones drawn last: a handful of E points must not be hidden under
        # a hundred thousand A points.
        for zz in ZONES:
            sel = idx[z[idx] == zz]
            if sel.size:
                ax.scatter(y[sel], p[sel], s=3, alpha=.35, linewidths=0,
                           color=ZONE_COLOURS[zz])
        ax.set_xlim(0, 450); ax.set_ylim(0, 450)
        ax.set_xlabel("reference glucose (mg/dL)")
        ax.set_ylabel(f"forecast at +{h} min (mg/dL)")
        ax.set_title(f"h = {h} min   (n = {y.size:,})", loc="left", fontsize=10)
        txt = "\n".join(f"{zz}  {share[zz]:5.2f}%" for zz in ZONES)
        ax.text(.975, .03, txt, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, family="monospace",
                bbox=dict(fc="white", ec="#bbb", alpha=.9, pad=4))
    for k in range(len(hz), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.tight_layout()
    path = out / "error_grid.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return path


def zones_by_glucose(df: pd.DataFrame, out: Path,
                     edges: Sequence[float] = (0, 54, 70, 90, 120, 150, 180,
                                               250, 1000)) -> Path:
    """Zone mix across the reference range, per horizon.

    The single most useful diagnostic: zone A is a narrow corridor at low
    glucose and a wide one at high glucose, so a model can look fine overall
    and still be unusable exactly where it matters.
    """
    plt = _pyplot()
    hz = horizons_of(df)
    labels = [f"{int(edges[i])}-{int(edges[i+1])}" for i in range(len(edges) - 2)]
    labels.append(f">{int(edges[-2])}")
    fig, axes = plt.subplots(1, len(hz), figsize=(3.5 * len(hz), 4.1),
                             squeeze=False, sharey=True)
    for k, h in enumerate(hz):
        ax = axes[0][k]
        y, p, ok = _pair(df, h)
        y, p = y[ok], p[ok]
        z = dts_zones(y, p)
        which = np.clip(np.searchsorted(np.asarray(edges[1:]), y, side="right"),
                        0, len(labels) - 1)
        bottom = np.zeros(len(labels))
        counts = np.array([(which == b).sum() for b in range(len(labels))])
        for zz in ZONES:
            frac = np.array([
                (np.mean(z[which == b] == zz) * 100) if counts[b] else 0.0
                for b in range(len(labels))])
            ax.bar(np.arange(len(labels)), frac, bottom=bottom, width=.82,
                   color=ZONE_COLOURS[zz], label=zz if k == 0 else None)
            bottom += frac
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_title(f"+{h} min", loc="left", fontsize=10, pad=14)
        ax.set_xlabel("reference glucose (mg/dL)")
        ax.set_ylim(0, 107)
        for b, n in enumerate(counts):
            ax.text(b, 101.5, f"{n/max(counts.sum(),1)*100:.0f}%", ha="center",
                    fontsize=6, color="#666")
        if k == 0:
            ax.set_ylabel("% of anchors in each zone")
    # Figure-level legend at the top: an axes legend lands on the rotated tick
    # labels no matter which corner it is given.
    handles = [plt.Rectangle((0, 0), 1, 1, color=ZONE_COLOURS[z]) for z in ZONES]
    fig.legend(handles, list(ZONES), ncol=5, fontsize=9, frameon=False,
               loc="upper right", bbox_to_anchor=(.995, 1.0), title="DTS zone",
               title_fontsize=9)
    fig.suptitle("Zone mix by reference glucose  (grey % = share of all anchors "
                 "in that bin)", fontsize=10, x=.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, .90))
    path = out / "zones_by_glucose.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return path


def _conditions(df: pd.DataFrame) -> Dict[str, Dict[str, np.ndarray]]:
    """Strata known at forecast time, so a finding is actionable."""
    out: Dict[str, Dict[str, np.ndarray]] = {}
    if "cgm_now" in df:
        v = df["cgm_now"].to_numpy(float)
        out["glucose now"] = {"<70": v < 70, "70-120": (v >= 70) & (v < 120),
                              "120-180": (v >= 120) & (v < 180),
                              "180-250": (v >= 180) & (v < 250), ">250": v >= 250}
    if "cgm_delta_30min" in df:
        d = df["cgm_delta_30min"].to_numpy(float)
        out["30-min trend"] = {"falling": d < -10, "steady": np.abs(d) <= 10,
                               "rising": d > 10}
    if "min_since_bolus" in df:
        b = df["min_since_bolus"].to_numpy(float)
        out["since bolus"] = {"<30m": b < 30, "30-90m": (b >= 30) & (b < 90),
                              "90-240m": (b >= 90) & (b < 240), ">4h": b >= 240}
    if "tod_min" in df:
        t = df["tod_min"].to_numpy(float) / 60.0
        out["hour of day"] = {"00-06": t < 6, "06-12": (t >= 6) & (t < 12),
                              "12-18": (t >= 12) & (t < 18), "18-24": t >= 18}
    return out


def error_profile(df: pd.DataFrame, out: Path, min_n: int = 50) -> Path:
    """Signed bias, MAE and zone-A rate across forecast-time conditions."""
    plt = _pyplot()
    hz = horizons_of(df)
    cond = _conditions(df)
    if not cond:
        raise RuntimeError("predictions frame carries no context columns to slice on")
    fig, axes = plt.subplots(3, len(cond), figsize=(3.3 * len(cond), 8.0),
                             squeeze=False)
    for c, (name, strata) in enumerate(cond.items()):
        labels = list(strata)
        x = np.arange(len(labels))
        for h in hz:
            y, p, ok = _pair(df, h)
            bias, mae, za = [], [], []
            for lab in labels:
                m = strata[lab] & ok
                if m.sum() < min_n:
                    bias.append(np.nan); mae.append(np.nan); za.append(np.nan); continue
                e = p[m] - y[m]
                bias.append(float(np.mean(e)))
                mae.append(float(np.mean(np.abs(e))))
                za.append(float(np.mean(dts_zones(y[m], p[m]) == "A") * 100))
            for r, series in enumerate((bias, mae, za)):
                axes[r][c].plot(x, series, marker="o", ms=3.5, lw=1.4,
                                label=f"+{h}m" if (r == 0 and c == 0) else None)
        for r, ylab in enumerate(("signed bias (mg/dL)", "MAE (mg/dL)",
                                  "zone A (%)")):
            ax = axes[r][c]
            ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right",
                                                 fontsize=7)
            ax.grid(alpha=.3)
            if c == 0:
                ax.set_ylabel(ylab, fontsize=9)
            if r == 0:
                ax.axhline(0, color="#c62828", lw=.9, ls="--")
                ax.set_title(name, loc="left", fontsize=10)
    axes[0][0].legend(fontsize=8, frameon=False, ncol=len(hz))
    fig.suptitle("Error by condition known at forecast time  "
                 "(top row: >0 means the model reads high)", fontsize=10,
                 x=.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, .97))
    path = out / "error_profile.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return path


# -------------------------------------------------------------------- traces
def traces(df: pd.DataFrame, store, out: Path, context: int = 96,
           n_bad: int = 6, n_good: int = 2, horizon: Optional[int] = None,
           seed: int = 0) -> Path:
    """Worked examples: history, what happened, and what the model predicted.

    Chosen by the worst zone at ``horizon`` (default: the longest), so the panel
    shows the failures rather than a random sample that is 85% zone A.
    """
    plt = _pyplot()
    hz = horizons_of(df)
    h = horizon or hz[-1]
    y, p, ok = _pair(df, h)
    z = np.where(ok, dts_zones(y, p), "?")
    err = np.where(ok, np.abs(p - y), -np.inf)

    rank = np.lexsort((-err, np.argsort(np.argsort(
        [{"E": 0, "D": 1, "C": 2, "B": 3, "A": 4}.get(s, 9) for s in z]))))
    bad = [i for i in rank if z[i] in ("E", "D", "C", "B")][:n_bad]
    rng = np.random.default_rng(seed)
    good_pool = np.flatnonzero(z == "A")
    good = list(rng.choice(good_pool, min(n_good, good_pool.size), replace=False)) \
        if good_pool.size else []
    picks = bad + good
    if not picks:
        raise RuntimeError("no scorable anchors to draw")

    cgm = int(store.channel_ids(["cgm"])[0])
    rows = store.anchors["row"].to_numpy()
    steps = np.array(hz) // GRID_MIN
    ncol = 4
    nrow = int(np.ceil(len(picks) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.7 * ncol, 2.7 * nrow),
                             squeeze=False)
    for k, i in enumerate(picks):
        ax = axes[k // ncol][k % ncol]
        r = int(rows[int(df["anchor"].to_numpy()[i])])
        lo = max(r - context + 1, 0)
        hist = np.asarray(store.channels[lo:r + 1, cgm], dtype=float)
        fut = np.asarray(store.channels[r + 1:r + 1 + int(steps.max()), cgm],
                         dtype=float)
        th = (np.arange(hist.size) - hist.size + 1) * GRID_MIN
        tf = (np.arange(fut.size) + 1) * GRID_MIN
        ax.plot(th, hist, color="#37474f", lw=1.3, label="CGM history")
        ax.plot(tf, fut, color="#1565c0", lw=1.6, label="what happened")
        pr = [df[f"y_pred_{hh}"].to_numpy(float)[i] for hh in hz]
        ax.plot(list(np.array(hz)), pr, "o--", color="#e8703a", ms=5, lw=1.5,
                label="forecast")
        ax.axvline(0, color="#999", lw=.8)
        ax.axhspan(70, 180, color="#2e7d32", alpha=.07, lw=0)
        ax.axhline(70, color="#c62828", lw=.7, ls=":")
        zone = z[i]
        ax.set_title(f"zone {zone} @ +{h}m   err {p[i]-y[i]:+.0f} mg/dL",
                     loc="left", fontsize=9,
                     color=ZONE_COLOURS.get(zone, "#333"))
        ax.set_xlabel("minutes from anchor", fontsize=8)
        ax.tick_params(labelsize=7)
        if k % ncol == 0:
            ax.set_ylabel("mg/dL", fontsize=8)
        if k == 0:
            ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    for k in range(len(picks), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle(f"Worst {len(bad)} misses and {len(good)} successes at +{h} min",
                 fontsize=10, x=.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, .96))
    path = out / "traces.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return path


# ----------------------------------------------------------------- reporting
def zone_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h in horizons_of(df):
        y, p, ok = _pair(df, h)
        z = dts_zones(y[ok], p[ok])
        row = {"horizon": h, "n": int(ok.sum())}
        row.update({zz: round(float(np.mean(z == zz) * 100), 2) for zz in ZONES})
        row["AB"] = round(row["A"] + row["B"], 2)
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--split", default="test",
                    help="test | internal_holdout | val")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: <run>/figures/<split>")
    ap.add_argument("--store", default=None,
                    help="sequence store for the trace panel; default: the store "
                         "recorded in the run's resolved config")
    ap.add_argument("--horizon", type=int, default=None,
                    help="horizon the traces are ranked by (default: the longest)")
    ap.add_argument("--max-points", type=int, default=25000)
    ap.add_argument("--no-traces", action="store_true")
    a = ap.parse_args(argv)

    pred = a.run / f"predictions_{a.split}.parquet"
    if not pred.exists():
        raise SystemExit(f"{pred} not found")
    df = pd.read_parquet(pred)
    out = a.out or (a.run / "figures" / a.split)
    out.mkdir(parents=True, exist_ok=True)

    table = zone_table(df)
    print(f"{a.run.name} [{a.split}]  n={len(df):,}")
    print(table.to_string(index=False))
    print(f"mean zone A over {len(table)} horizons: {table['A'].mean():.2f}%")
    table.to_csv(out / "zone_table.csv", index=False)

    made = [error_grid(df, out, a.max_points), zones_by_glucose(df, out),
            error_profile(df, out)]

    if not a.no_traces:
        root = a.store
        if root is None:
            cfg = a.run / "config.resolved.yaml"
            if cfg.exists():
                import yaml
                root = (yaml.safe_load(cfg.read_text()) or {}).get(
                    "data", {}).get("store")
        if root is None:
            print("no store available; skipping traces (pass --store)")
        else:
            from ..data.store import SequenceStore
            made.append(traces(df, SequenceStore(root), out, horizon=a.horizon))

    print("\n".join(str(m) for m in made))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
