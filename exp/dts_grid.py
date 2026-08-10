"""The DTS (Diabetes Technology Society) Error Grid — the competition's official
error-grid metric.

Reference
---------
Klonoff DC, Freckmann G, Pleus S, Kovatchev BP, Kerr D, et al.
"The Diabetes Technology Society Error Grid and Trend Accuracy Matrix for Glucose
Monitors." *Journal of Diabetes Science and Technology* 2024;18(6):1346-1361.
DOI 10.1177/19322968241275701 · PMID 39369312 · PMC11531029
Official tool: https://www.diabetestechnology.org/dtseg/

What the paper establishes
--------------------------
* Five risk zones **A-E** with **straight-line borders**, identical for BGMs and CGMs.
* The zones are contours of a continuous risk surface fitted to the older
  Surveillance Error Grid, thresholded at risk scores **0-0.5 (A), >0.5-1.5 (B),
  >1.5-2.5 (C), >2.5-3.5 (D), >3.5 (E)**.
* The surface is deliberately **asymmetric**: a monitor reading *above* the reference
  is scored riskier than the same error below it, because over-reading drives
  over-dosing and then hypoglycaemia. So the borders above the identity line sit
  closer to it than the borders below.
* **The grid is not trademarked and is in the public domain.**
* The exact border coordinates live in **Supplemental Appendix 2** of the paper.

Why the coordinates are not hard-coded here
-------------------------------------------
Supplemental Appendix 2 is not reachable without a subscription or an interactive
browser session: the SAGE copy is paywalled (HTTP 403), the PMC copy is behind a
proof-of-work bot challenge, the article is not in the PMC/Europe-PMC open-access
subset, and the official tool is a server-side Shiny app. Rather than invent
plausible-looking numbers — which would produce a metric that *looks* official and
silently is not — the boundary table is loaded from a data file you supply once.

Because the borders are straight lines, supplying them is purely mechanical: the
same piecewise-linear machinery that implements Parkes in `metrics.py` evaluates
them unchanged.

How to enable it (one-time, ~5 minutes)
---------------------------------------
1. Open the paper and download Supplemental Appendix 2:
   https://journals.sagepub.com/doi/suppl/10.1177/19322968241275701
   (or https://pmc.ncbi.nlm.nih.gov/articles/PMC11531029/ → Supplementary Materials,
   file ``sj-docx-1-dst-10.1177_19322968241275701.docx``)
2. Copy the vertex table into ``exp/dts_boundaries.json`` following
   ``exp/dts_boundaries.template.json``. Coordinates are ``[reference, monitor]``
   pairs in **mg/dL**, listed left to right along each border.
3. Nothing else changes. ``metrics.evaluate()`` picks the file up automatically and
   starts reporting ``dts_A`` … ``dts_E`` alongside Clarke and Parkes.

Verify before trusting it: plot a grid of (ref, pred) pairs against the official
tool at https://www.diabetestechnology.org/dtseg/ and confirm the zone letters agree.
``python -m exp.dts_grid --selftest`` prints a comparison table for exactly that.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DEFAULT_PATH = Path(__file__).resolve().parent / "dts_boundaries.json"

#: Borders, ordered outward from the identity line. "upper" = monitor above
#: reference (the riskier side); "lower" = monitor below reference.
BORDER_ORDER = ("AB", "BC", "CD", "DE")
ZONE_FOR_BORDER = {"AB": "B", "BC": "C", "CD": "D", "DE": "E"}


class DTSBoundariesUnavailable(RuntimeError):
    """Raised when the DTS coordinate file has not been supplied."""


def _piecewise(vertices, x, side: str) -> np.ndarray:
    """Monitor-glucose value of a straight-line border at reference value `x`.

    Beyond the last vertex an upper border returns +inf and a lower border -inf:
    the border has left the grid there, so nothing can be outside it.
    """
    vx = np.asarray([v[0] for v in vertices], dtype=np.float64)
    vy = np.asarray([v[1] for v in vertices], dtype=np.float64)
    order = np.argsort(vx)
    vx, vy = vx[order], vy[order]
    if side == "upper":
        return np.interp(x, vx, vy, left=vy[0], right=np.inf)
    return np.interp(x, vx, vy, left=-np.inf, right=vy[-1])


def load_boundaries(path: Path | str | None = None) -> dict:
    """Load and validate the boundary table."""
    p = Path(path) if path is not None else DEFAULT_PATH
    if not p.exists():
        raise DTSBoundariesUnavailable(
            f"DTS Error Grid boundary file not found at {p}.\n"
            "The exact border coordinates are in Supplemental Appendix 2 of\n"
            "  Klonoff et al. 2024, JDST 18(6):1346-1361, DOI 10.1177/19322968241275701\n"
            "  https://journals.sagepub.com/doi/suppl/10.1177/19322968241275701\n"
            f"Copy them into {p} using the schema in "
            f"{DEFAULT_PATH.with_name('dts_boundaries.template.json').name}.\n"
            "Until then, use parkes_zones() -- the closest published relative -- and do "
            "not report a number as 'DTS zone A'."
        )
    spec = json.loads(p.read_text())
    for side in ("upper", "lower"):
        if side not in spec:
            raise ValueError(f"{p}: missing '{side}' section")
        for b in BORDER_ORDER:
            if b not in spec[side]:
                continue                      # a border may legitimately not exist
            verts = spec[side][b]
            if len(verts) < 2:
                raise ValueError(f"{p}: {side}.{b} needs at least 2 vertices")
            for v in verts:
                if len(v) != 2:
                    raise ValueError(f"{p}: {side}.{b} vertex {v} is not [ref, monitor]")
    return spec


def dts_zones(y_true, y_pred, boundaries: dict | None = None,
              path: Path | str | None = None) -> np.ndarray:
    """DTS Error Grid zone per point as 'A'..'E' ('?' where either value is not finite).

    `y_true` is the reference glucose, `y_pred` the monitor/predicted value, both
    mg/dL. Raises `DTSBoundariesUnavailable` if the coordinate file is absent.
    """
    spec = boundaries if boundaries is not None else load_boundaries(path)
    ref = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    zone = np.full(ref.shape, "A", dtype="<U1")

    above = pred >= ref
    for border in BORDER_ORDER:
        letter = ZONE_FOR_BORDER[border]
        if border in spec.get("upper", {}):
            out = above & (pred > _piecewise(spec["upper"][border], ref, "upper"))
            zone[out] = letter
        if border in spec.get("lower", {}):
            out = (~above) & (pred < _piecewise(spec["lower"][border], ref, "lower"))
            zone[out] = letter

    zone[~np.isfinite(ref) | ~np.isfinite(pred)] = "?"
    return zone


def available(path: Path | str | None = None) -> bool:
    p = Path(path) if path is not None else DEFAULT_PATH
    return p.exists()


# ------------------------------------------------------------------- self-test
def _selftest() -> int:
    """Print zone assignments for a spread of points, to diff against the official
    tool at https://www.diabetestechnology.org/dtseg/."""
    try:
        spec = load_boundaries()
    except DTSBoundariesUnavailable as e:
        print(e)
        return 1

    print(f"loaded {DEFAULT_PATH}")
    for side in ("upper", "lower"):
        for b in BORDER_ORDER:
            if b in spec.get(side, {}):
                print(f"  {side:5s} {b}: {len(spec[side][b])} vertices "
                      f"{spec[side][b][0]} … {spec[side][b][-1]}")

    refs = [40, 60, 70, 100, 140, 180, 250, 300, 400]
    rels = [-0.60, -0.40, -0.20, -0.10, 0.0, 0.10, 0.20, 0.40, 0.60]
    print("\nzone by reference (rows) x relative error (cols)")
    print("ref  " + "".join(f"{int(r*100):>+6d}%" for r in rels))
    for r in refs:
        preds = np.array([r * (1 + q) for q in rels])
        z = dts_zones(np.full(len(preds), float(r)), preds, spec)
        print(f"{r:>4} " + "".join(f"{c:>7}" for c in z))

    # sanity invariants that must hold for any correct table
    g = np.linspace(20, 400, 400)
    same = dts_zones(g, g, spec)
    bad = int((same != "A").sum())
    print(f"\nperfect predictions in zone A: {len(g)-bad}/{len(g)}"
          f"{'  <-- FAIL' if bad else '  ok'}")
    hi = dts_zones(np.full(200, 70.0), np.linspace(70, 400, 200), spec)
    monotone = all(hi[i] <= hi[i + 1] for i in range(len(hi) - 1))
    print(f"zones worsen monotonically as the error grows: {monotone}"
          f"{'' if monotone else '  <-- FAIL'}")
    return 0 if (bad == 0 and monotone) else 1


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true",
                    help="print a zone table to diff against the official tool")
    a = ap.parse_args()
    raise SystemExit(_selftest() if a.selftest else print(__doc__) or 0)
