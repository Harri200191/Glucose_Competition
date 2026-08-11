"""The DTS (Diabetes Technology Society) Error Grid — the competition's official
error-grid metric.

Reference
---------
Klonoff DC, Freckmann G, Pleus S, Kovatchev BP, Kerr D, et al. "The Diabetes
Technology Society Error Grid and Trend Accuracy Matrix for Glucose Monitors."
*J Diabetes Sci Technol* 2024;18(6):1346-1361. DOI 10.1177/19322968241275701 ·
PMID 39369312 · PMC11531029 · tool <https://www.diabetestechnology.org/dtseg/>

The boundaries below are transcribed from the organisers' own scorer,
`toolkit/metrics.py::calculate_dts_error_grid`, vendored in this repo from
<https://github.com/replicahealth/glucose-prediction-submission-toolkit>. They are
straight lines, exactly as the paper describes, so no risk-surface lookup is needed.
The grid is public domain.

Geometry
--------
Four borders on each side of the identity line, each a horizontal segment followed
by a ray. `ref` is the reference (true) glucose, `pred` the monitor/predicted value,
both mg/dL.

    upper (pred above ref -- the riskier side, so these sit closer to the diagonal)
      A|B   60    for ref <= 50,  else (540/450)  *(ref - 50)   + 60
      B|C   86.5  for ref <= 50,  else (513.5/297)*(ref - 50)   + 86.5
      C|D   124   for ref <= 50,  else (476/191)  *(ref - 50)   + 124
      D|E   179   for ref <= 50,  else (421/117)  *(ref - 50)   + 179

    lower (pred below ref)
      A|B   0     for ref <= 62.5, else (430/537.5)*(ref - 62.5) + 50
      B|C   0     for ref <= 97.5, else (257/502.5)*(ref - 97.5) + 50
      C|D   0     for ref <= 153,  else (147/447)  *(ref - 153)  + 50
      D|E   0     for ref <= 238,  else (76/362)   *(ref - 238)  + 50

The step from 0 to 50 on the lower borders is not a discontinuity in the grid; it is
how a **vertical** border segment appears when the border is written as a function of
`ref`. Below the knee there is simply no lower bound for that zone.

Note the asymmetry: at ref = 100 the A|B border sits at 120 above (a +20 % error) but
at 80 below (a −20 % error) — and the upper borders tighten faster as glucose falls,
because over-reading at a low drives over-dosing and then hypoglycaemia.

`--selftest` proves this module agrees exactly with the vendored official scorer.
"""
from __future__ import annotations

import numpy as np

ZONES = ("A", "B", "C", "D", "E")


def _borders(ref: np.ndarray) -> tuple[np.ndarray, ...]:
    """The eight border values at each reference glucose."""
    ab_up = np.where(ref <= 50, 60.0, (540.0 / 450.0) * (ref - 50) + 60.0)
    bc_up = np.where(ref <= 50, 86.5, (513.5 / 297.0) * (ref - 50) + 86.5)
    cd_up = np.where(ref <= 50, 124.0, (476.0 / 191.0) * (ref - 50) + 124.0)
    de_up = np.where(ref <= 50, 179.0, (421.0 / 117.0) * (ref - 50) + 179.0)

    ab_lo = np.where(ref <= 62.5, 0.0, (430.0 / 537.5) * (ref - 62.5) + 50.0)
    bc_lo = np.where(ref <= 97.5, 0.0, (257.0 / 502.5) * (ref - 97.5) + 50.0)
    cd_lo = np.where(ref <= 153.0, 0.0, (147.0 / 447.0) * (ref - 153.0) + 50.0)
    de_lo = np.where(ref <= 238.0, 0.0, (76.0 / 362.0) * (ref - 238.0) + 50.0)
    return ab_up, bc_up, cd_up, de_up, ab_lo, bc_lo, cd_lo, de_lo


def dts_zones(y_true, y_pred) -> np.ndarray:
    """DTS Error Grid zone per point as 'A'..'E', or '?' where either value is
    non-finite. `y_true` is the reference, `y_pred` the prediction, both mg/dL."""
    ref = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    ab_up, bc_up, cd_up, de_up, ab_lo, bc_lo, cd_lo, de_lo = _borders(ref)

    zone = np.full(ref.shape, "E", dtype="<U1")          # outside every border
    zone[((pred <= de_up) & (pred > cd_up)) | ((pred < cd_lo) & (pred >= de_lo))] = "D"
    zone[((pred <= cd_up) & (pred > bc_up)) | ((pred < bc_lo) & (pred >= cd_lo))] = "C"
    zone[((pred <= bc_up) & (pred > ab_up)) | ((pred < ab_lo) & (pred >= bc_lo))] = "B"
    zone[(pred <= ab_up) & (pred >= ab_lo)] = "A"

    zone[~np.isfinite(ref) | ~np.isfinite(pred)] = "?"
    return zone


def dts_fractions(y_true, y_pred) -> dict[str, float]:
    """Percentage of finite points in each zone, plus the A+B total."""
    z = dts_zones(y_true, y_pred)
    valid = z[z != "?"]
    n = max(len(valid), 1)
    out = {f"dts_{k}": float((valid == k).sum()) / n * 100.0 for k in ZONES}
    out["dts_AB"] = out["dts_A"] + out["dts_B"]
    return out


def available(*_a, **_kw) -> bool:
    """Kept for API stability: the grid is now always available."""
    return True


# ------------------------------------------------------------------- self-test
def _selftest() -> int:
    """Assert exact agreement with the organisers' vendored scorer."""
    import importlib.util
    from pathlib import Path

    official_path = Path(__file__).resolve().parent.parent / "toolkit" / "metrics.py"
    if not official_path.exists():
        print(f"official scorer not found at {official_path} -- cannot cross-check.\n"
              "Vendor it from "
              "https://github.com/replicahealth/glucose-prediction-submission-toolkit")
        return 1
    spec = importlib.util.spec_from_file_location("official_metrics", official_path)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)

    rng = np.random.default_rng(0)
    fails = 0
    for name, ref, pred in [
        ("uniform",   rng.uniform(20, 450, 200_000), rng.uniform(20, 450, 200_000)),
        ("realistic", (r := rng.uniform(40, 400, 200_000)),
                      np.clip(r * rng.normal(1.0, 0.18, 200_000), 5, 600)),
        ("hypo",      rng.uniform(30, 80, 100_000),  rng.uniform(20, 250, 100_000)),
        ("perfect",   (g := np.linspace(20, 450, 50_000)), g.copy()),
        ("edges",     np.repeat([50.0, 62.5, 97.5, 153.0, 238.0], 4000),
                      np.tile(np.linspace(0, 600, 4000), 5)),
    ]:
        # Compare the raw percentages, NOT the official function's 2-dp rounded
        # output: at an exact half (e.g. 19.265) Python's round and numpy's round
        # break the tie differently, which looks like a mismatch while the
        # classification is identical. The counts are what must agree.
        mine = dts_fractions(ref, pred)
        theirs = official.calculate_dts_error_grid(pred, ref)
        row = []
        for z in ZONES:
            a = mine[f"dts_{z}"]
            b = theirs[f"DTS_{z}_ZONE_PERCENT"]
            # b is rounded to 2 dp by the official code, so allow half a unit there
            ok = abs(a - b) <= 0.005 + 1e-9
            fails += (not ok)
            row.append(f"{z}={a:6.2f}/{b:6.2f}{'' if ok else ' MISMATCH'}")
        print(f"  {name:<10} n={len(ref):>7,}  " + "  ".join(row))


    # Exact per-point equivalence. Calling the official function one point at a time
    # makes its output unambiguous (a single point is 0 % or 100 % of a zone, which
    # survives the 2-dp rounding), so this compares classifications, not percentages.
    n_pt = 4000
    r1 = np.concatenate([rng.uniform(20, 450, n_pt // 2),
                         rng.choice([50.0, 62.5, 97.5, 153.0, 238.0], n_pt // 2)])
    p1 = np.concatenate([rng.uniform(5, 600, n_pt // 2),
                         r1[n_pt // 2:] * rng.uniform(0.2, 3.0, n_pt // 2)])
    mine_pt = dts_zones(r1, p1)
    bad = 0
    for i in range(len(r1)):
        d = official.calculate_dts_error_grid([p1[i]], [r1[i]])
        theirs_z = next(z for z in ZONES if d[f"DTS_{z}_ZONE_PERCENT"] == 100.0)
        if theirs_z != mine_pt[i]:
            bad += 1
            if bad <= 3:
                print(f"    point mismatch ref={r1[i]:.4f} pred={p1[i]:.4f} "
                      f"mine={mine_pt[i]} official={theirs_z}")
    print(f"\n  exact per-point agreement: {len(r1)-bad}/{len(r1)}"
          f"{'' if bad == 0 else '  <-- FAIL'}")
    fails += (bad > 0)

    perfect = dts_zones(np.linspace(20, 450, 5000), np.linspace(20, 450, 5000))
    all_a = bool((perfect == "A").all())
    print(f"  perfect predictions all in zone A: {all_a}")
    fails += (not all_a)

    print("\n  zone by reference (rows) x relative error (cols)")
    rels = [-0.6, -0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4, 0.6]
    print("  ref  " + "".join(f"{int(q*100):>+7d}%" for q in rels))
    for r in [40, 60, 70, 100, 140, 180, 250, 300, 400]:
        p = np.array([r * (1 + q) for q in rels])
        z = dts_zones(np.full(len(p), float(r)), p)
        print(f"  {r:>4} " + "".join(f"{c:>8}" for c in z))

    print("\n" + ("ALL CHECKS PASSED -- matches the official scorer exactly"
                  if fails == 0 else f"{fails} MISMATCHES"))
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true",
                    help="verify against toolkit/metrics.py, the official scorer")
    a = ap.parse_args()
    raise SystemExit(_selftest() if a.selftest else (print(__doc__) or 0))
