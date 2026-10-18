"""The engineered feature set: 12 families, ~700 columns, all strictly causal.

CONTRACT
--------
``build_columns(families) -> [name, ...]`` and
``compute_block(W, anchors, index, families) -> [n, F] float32`` must agree on
order and width for EVERY subset. They do so by construction: each family returns
an ``OrderedDict[name -> array]`` and ``build_columns`` is that same code run on a
two-row dummy window. There is no second list of names to drift out of sync.

``W`` is ``store.channels[rows[:,None] + arange(-L+1, 1)]`` -> ``[n, L, 23]`` in
RAW PHYSICAL UNITS, oldest first, ``W[:, -1]`` being time t. Nothing else is
readable, which is what makes every column causal by construction rather than by
review -- the same argument prepare.py makes about windows.

RULE N -- NEVER EMIT NaN TO MEAN "ABSENT"
-----------------------------------------
``build.py`` runs ``nan_to_num(blk, nan=0.0)`` in PHYSICAL units, BEFORE the
standardiser is fitted. So a NaN does not become "the training centre" -- it
becomes physical zero, which after robust centring is a large negative outlier
indistinguishable from a real extreme. Every family therefore declares a neutral
fill (either impossible for the real quantity, or exactly its "nothing happened"
value) AND ships a coverage flag. Heart rate is the clean example: an absent
cohort emits 0 bpm, which is physiologically impossible, next to ``hr_cov = 0``.
Never impute a cohort mean -- that manufactures a physiological reading out of a
cohort label, which is the same shortcut hazard as feeding source_file.

WHAT TO EXPECT
--------------
Experiment B in this repo deleted all ELEVEN engineered pharmacology channels and
measured -0.37 / -0.26 / -0.22 DTS-A@60 against seed noise of 0.34 / 0.18 / 0.06.
Statics were worth 0.24. The whole A-F program spans 1.19 points. So 11 features
bought about 0.3, at the noise floor, and the honest prior for 700 is +0.0 to
+0.5 with a real chance of a negative from input dimension alone. Anything above
+1.0 should be assumed a cohort shortcut until the unseen-source holdout agrees.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from typing import Dict, List, Optional

import numpy as np

GRID = 5.0                                   # minutes per lattice step
WIN = (3, 6, 12, 24, 36, 72, 144, 288)       # 15m 30m 1h 2h 3h 6h 12h 24h
WIN7 = (6, 12, 24, 36, 72, 144, 288)

# store channel order, from artifacts/store/*/manifest.json
CGM, CGM_M = 0, 1
BASAL, BASAL_M = 2, 3
BOLUS, BOLUS_M = 4, 5
CARB, CARB_M = 6, 7
HR, HR_M = 8, 9
STEPS, STEPS_M = 10, 11
EXP_BOLUS = (12, 13, 14, 15)                 # 30 / 60 / 120 / 240 min
EXP_BASAL = (16, 17)                         # 60 / 240
EXP_CARB = (18, 19, 20)                      # 20 / 45 / 90
BOLUS_SINCE, CARB_SINCE = 21, 22


# ----------------------------------------------------------------- primitives
def _rcum(a: np.ndarray) -> np.ndarray:
    """Reverse cumulative sum: ``out[:, l-1]`` is the sum over the LAST l steps."""
    return np.cumsum(a[:, ::-1], axis=1)


def _safe(num, den, fill=0.0):
    """Divide, yielding ``fill`` where the denominator vanishes.

    Broadcasts both sides first: several families divide a SCALAR constant by a
    per-anchor vector (ISF = 1800/TDD), and sizing the output from the numerator
    alone silently fails on exactly those.
    """
    num, den = np.broadcast_arrays(np.asarray(num, np.float64),
                                   np.asarray(den, np.float64))
    out = np.full(num.shape, float(fill), dtype=np.float64)
    return np.divide(num, den, out=out, where=np.abs(den) > 1e-9)


def _kernel(shape: str, tp: float, td: float, L: int) -> np.ndarray:
    """One causal pharmacokinetic kernel over window ages, as a length-L vector.

    Row j of the window has age tau = (L-1-j)*5 minutes, so the kernel is
    supported on tau >= 0 only -- causality is in the construction, not a check.

    ``iob``/``act`` are the Loop/Dexcom exponential insulin model; ``cob``/``ra``
    the two-compartment carbohydrate curve. What matters for the experiment is
    that ACT and RA PEAK AT tau > 0: every exposure channel already in the store
    decays monotonically from tau=0, so a non-monotone kernel is genuinely new
    shape rather than a rescaling of something the window already carries.
    """
    tau = (L - 1 - np.arange(L)).astype(np.float64) * GRID
    if shape in ("iob", "act"):
        tc = tp * (1 - tp / td) / (1 - 2 * tp / td)
        a = 2 * tc / td
        S = 1.0 / (1 - a + (1 + a) * math.exp(-td / tc))
        if shape == "iob":
            k = 1 - S * (1 - a) * ((tau ** 2 / (tc * td * (1 - a)) - tau / tc - 1)
                                   * np.exp(-tau / tc) + 1)
        else:
            k = (S / tc ** 2) * tau * np.exp(-tau / tc)
    elif shape == "cob":
        k = (1 + tau / tp) * np.exp(-tau / tp)
    elif shape == "ra":
        k = (tau / tp ** 2) * np.exp(-tau / tp)
    else:
        raise ValueError(shape)
    k = np.where(tau > td, 0.0, np.maximum(k, 0.0))
    return k.astype(np.float32)


def _isf_cr(tdd: np.ndarray):
    """Personal insulin constants from total daily dose, CLAMPED.

    ISF = 1800/TDD and CR = 500/TDD are the standard rules, and both explode when
    TDD -> 0. That is not hypothetical here: 7.5% of anchors have TDD < 0.1 U over
    the window, because the cohort does not log insulin. With an epsilon floor,
    ISF reaches 1.8e6 and the column stops meaning "insulin sensitivity" and
    starts meaning "was this cohort logging insulin" -- a cohort fingerprint, the
    same hazard class as feeding source_file.

    So: floor TDD at 5 U/day (below that the record is absent, not the patient),
    then clip to clinically observed ranges. ISF 10-300 mg/dL per unit and CR
    3-60 g per unit bracket the published adult and paediatric spans with room to
    spare, so the clip only bites where the input was already meaningless.
    """
    t = np.maximum(np.asarray(tdd, np.float64), 5.0)
    return np.clip(1800.0 / t, 10.0, 300.0), np.clip(500.0 / t, 3.0, 60.0)


def _runlen_since(cond: np.ndarray) -> np.ndarray:
    """Steps since the last True in each row; L+1 where it never happened."""
    L = cond.shape[1]
    idx = np.where(cond, np.arange(L)[None, :], -1)
    last = np.maximum.accumulate(idx, axis=1)
    return (L - 1) - last[:, -1]


def _longest_run(cond: np.ndarray) -> np.ndarray:
    """Longest consecutive run of True, per row, without a Python loop."""
    L = cond.shape[1]
    idx = np.where(~cond, np.arange(L)[None, :], -1)
    last = np.maximum.accumulate(idx, axis=1)          # index of last False
    return (np.arange(L)[None, :] - last).max(axis=1)


class Ctx:
    """Everything shared across families, computed once per chunk."""

    def __init__(self, W: np.ndarray, anchors, index: np.ndarray, statics=None):
        self.W = W
        self.n, self.L, _ = W.shape
        L = self.L
        self.X = W[:, :, CGM].astype(np.float64)
        self.OBS = np.isfinite(self.X)                 # has a value (incl. interpolated)
        self.REAL = W[:, :, CGM_M] > 0.5               # is a real reading
        Xz = np.nan_to_num(self.X)
        self._k = _rcum(self.OBS.astype(np.float64))
        self._s1 = _rcum(Xz)
        self._s2 = _rcum(Xz * Xz)
        self._s3 = _rcum(Xz ** 3)
        lg = np.where(self.OBS & (self.X > 0), np.log(np.maximum(self.X, 1e-6)), 0.0)
        self._l1, self._l2 = _rcum(lg), _rcum(lg * lg)
        a = anchors.iloc[index]
        self.cgm_now = a["cgm_now"].to_numpy(np.float64)
        self.tod = a["tod_min"].to_numpy(np.float64)
        self.dow = a["dow"].to_numpy(np.float64)
        self.carb_since0 = a["min_since_carb"].to_numpy(np.float64)
        self.bolus_since0 = a["min_since_bolus"].to_numpy(np.float64)
        self.statics = statics
        # per-row clock, for the night block and meals-by-daypart
        self.tod_row = (self.tod[:, None] + (np.arange(L) - (L - 1)) * GRID) % 1440.0
        self._kern: Dict[tuple, np.ndarray] = {}
        self._cache: Dict[str, np.ndarray] = {}

    # --- moments over the trailing l steps, all from the same three cumsums ---
    def cnt(self, l): return self._k[:, l - 1]
    def mean(self, l): return _safe(self._s1[:, l - 1], np.maximum(self.cnt(l), 1))

    def var(self, l):
        m = self.mean(l)
        return np.maximum(_safe(self._s2[:, l - 1], np.maximum(self.cnt(l), 1)) - m * m, 0.0)

    def std(self, l): return np.sqrt(self.var(l))

    def skew(self, l):
        n = np.maximum(self.cnt(l), 1)
        m = self.mean(l); s = self.std(l)
        m3 = _safe(self._s3[:, l - 1], n) - 3 * m * _safe(self._s2[:, l - 1], n) + 2 * m ** 3
        return _safe(m3, np.maximum(s, 1e-6) ** 3)

    def logmean(self, l): return _safe(self._l1[:, l - 1], np.maximum(self.cnt(l), 1))

    def logstd(self, l):
        n = np.maximum(self.cnt(l), 1)
        m = self.logmean(l)
        return np.sqrt(np.maximum(_safe(self._l2[:, l - 1], n) - m * m, 0.0))

    def slice(self, l): return self.X[:, self.L - l:]

    def q(self, l, p):
        s = self.slice(l)
        with np.errstate(all="ignore"):
            out = np.nanquantile(np.where(self.OBS[:, self.L - l:], s, np.nan), p, axis=1)
        return np.nan_to_num(out, nan=0.0)

    def amin(self, l):
        with np.errstate(all="ignore"):
            return np.nan_to_num(np.nanmin(
                np.where(self.OBS[:, self.L - l:], self.slice(l), np.nan), axis=1), nan=0.0)

    def amax(self, l):
        with np.errstate(all="ignore"):
            return np.nan_to_num(np.nanmax(
                np.where(self.OBS[:, self.L - l:], self.slice(l), np.nan), axis=1), nan=0.0)

    def slope_r2(self, l):
        """Masked OLS over the last l steps: (slope per 5 min, R^2)."""
        key = f"sl{l}"
        if key in self._cache:
            return self._cache[key]
        m = self.OBS[:, self.L - l:].astype(np.float64)
        y = np.nan_to_num(self.slice(l))
        t = np.arange(l, dtype=np.float64)[None, :]
        Sw = m.sum(1); St = (m * t).sum(1); Stt = (m * t * t).sum(1)
        Sy = (m * y).sum(1); Sty = (m * t * y).sum(1); Syy = (m * y * y).sum(1)
        den = Sw * Stt - St * St
        slope = _safe(Sw * Sty - St * Sy, den)
        num = (Sw * Sty - St * Sy) ** 2
        r2 = _safe(num, den * np.maximum(Sw * Syy - Sy * Sy, 1e-9))
        ok = (den > 0) & (Sw >= 3)
        out = (np.where(ok, slope, 0.0), np.where(ok, np.clip(r2, 0, 1), 0.0))
        self._cache[key] = out
        return out

    def conv(self, ch: int, shape: str, tp: float, td: float):
        """One kernel applied to one channel -- the whole PK bank is a matmul."""
        key = (shape, tp, td)
        if key not in self._kern:
            self._kern[key] = _kernel(shape, tp, td, self.L)
        return np.nan_to_num(self.W[:, :, ch].astype(np.float64)) @ self._kern[key]

    def wsum(self, ch: int, l: int):
        if f"c{ch}" not in self._cache:
            self._cache[f"c{ch}"] = _rcum(np.nan_to_num(self.W[:, :, ch].astype(np.float64)))
        return self._cache[f"c{ch}"][:, l - 1]

    def wcount(self, ch: int, l: int, thr=0.0):
        k = f"n{ch}_{thr}"
        if k not in self._cache:
            self._cache[k] = _rcum((np.nan_to_num(self.W[:, :, ch]) > thr).astype(np.float64))
        return self._cache[k][:, l - 1]

    def cov(self, mch: int, l: int):
        """Fraction of the last l slots where this modality was observed."""
        return _safe(_rcum((np.nan_to_num(self.W[:, :, mch]) > 0.5).astype(np.float64))[:, l - 1],
                     float(l))


# --------------------------------------------------------------------- F1..F3
def f_cgm_level(c: Ctx) -> "OrderedDict[str, np.ndarray]":
    """Level, spread and shape of the glucose trace over trailing windows.

    Mostly TIER 2 by design: rolling mean/std/min/max are what an SSM's hidden
    state already is. The log-domain moments are the exception -- DTS zone A is a
    +/-20% RELATIVE band, i.e. constant width in ln(g), and a model trained on
    `delta` targets never forms them.
    """
    o = OrderedDict()
    for l in WIN:
        o[f"cgm__mean_{l}"] = c.mean(l)
        o[f"cgm__std_{l}"] = c.std(l)
        o[f"cgm__min_{l}"] = c.amin(l)
        o[f"cgm__max_{l}"] = c.amax(l)
    for l in WIN[1:]:
        for nm, p in (("median", .5), ("q10", .1), ("q25", .25), ("q75", .75), ("q90", .9)):
            o[f"cgm__{nm}_{l}"] = c.q(l, p)
    for l in (12, 36, 72, 144, 288):
        o[f"cgm__q05_{l}"] = c.q(l, .05)
        o[f"cgm__q95_{l}"] = c.q(l, .95)
    for l in (24, 36, 72, 144, 288):
        o[f"cgm__skew_{l}"] = c.skew(l)
    for l in (12, 36, 72, 144, 288):
        o[f"cgm__logmean_{l}"] = c.logmean(l)
        o[f"cgm__logstd_{l}"] = c.logstd(l)
    return o


def f_cgm_dyn(c: Ctx) -> "OrderedDict[str, np.ndarray]":
    """Differences, slopes, curvature and run structure.

    TIER 2: a causal TCN's first layer is a learned differencing filter. `r2` and
    `accel` are the only genuinely second-order columns -- "is this a clean ramp
    or a noisy plateau with the same slope" is a ratio of two quadratics.
    """
    o = OrderedDict()
    L = c.L
    now = np.nan_to_num(c.X[:, -1], nan=0.0)
    for k in (1, 2, 3, 4, 5, 6, 9, 12, 18, 24, 36, 48, 72, 144, 287):
        o[f"cgm__d_{k}"] = now - np.nan_to_num(c.X[:, L - 1 - k], nan=0.0)
    for l in WIN:
        s, r2 = c.slope_r2(l)
        o[f"cgm__slope_{l}"] = s
        o[f"cgm__r2_{l}"] = r2
    for l in (3, 6, 12, 24, 36):
        s_now, _ = c.slope_r2(l)
        m = c.OBS[:, L - 2 * l:L - l].astype(np.float64)
        y = np.nan_to_num(c.X[:, L - 2 * l:L - l])
        t = np.arange(l, dtype=np.float64)[None, :]
        Sw = m.sum(1); St = (m * t).sum(1); Stt = (m * t * t).sum(1)
        Sy = (m * y).sum(1); Sty = (m * t * y).sum(1)
        prev = _safe(Sw * Sty - St * Sy, Sw * Stt - St * St)
        o[f"cgm__accel_{l}"] = s_now - prev
    for k in (1, 2, 3, 6, 12):
        o[f"cgm__d2_{k}"] = (now - 2 * np.nan_to_num(c.X[:, L - 1 - k])
                             + np.nan_to_num(c.X[:, L - 1 - 2 * k]))
    d = np.diff(np.nan_to_num(c.X), axis=1)
    dm = (c.OBS[:, 1:] & c.OBS[:, :-1]).astype(np.float64)
    for l in (6, 12, 24, 72, 288):
        dd = d[:, -l:]; mm = dm[:, -l:]
        n = np.maximum(mm.sum(1), 1)
        mu = _safe((dd * mm).sum(1), n)
        o[f"cgm__dstd_{l}"] = np.sqrt(np.maximum(_safe((dd * dd * mm).sum(1), n) - mu * mu, 0))
        o[f"cgm__dabs_{l}"] = _safe((np.abs(dd) * mm).sum(1), n)
        o[f"cgm__dmax_{l}"] = (dd * mm).max(1)
        o[f"cgm__dmin_{l}"] = (dd * mm).min(1)
        o[f"cgm__dpos_{l}"] = _safe(((dd > 0) * mm).sum(1), n)
    o["cgm__run_up"] = _runlen_since(d <= 0).astype(np.float64)
    o["cgm__run_dn"] = _runlen_since(d >= 0).astype(np.float64)
    for l in (36, 72, 288):
        sl = np.where(c.OBS[:, L - l:], c.slice(l), -np.inf)
        o[f"cgm__since_max_{l}"] = (l - 1 - np.argmax(sl, axis=1)).astype(np.float64)
        sl2 = np.where(c.OBS[:, L - l:], c.slice(l), np.inf)
        o[f"cgm__since_min_{l}"] = (l - 1 - np.argmin(sl2, axis=1)).astype(np.float64)
    return o


def f_cgm_clin(c: Ctx) -> "OrderedDict[str, np.ndarray]":
    """Clinical glycaemic indices.

    TIER 1 for the RISK sub-block only (LBGI/HBGI/rl_max/rh_max/hypo hinge):
    those are asymmetric, saturating transforms that match the scoring geometry,
    and an RMSE-trained network has no incentive to build a hypo-risk
    representation. GRADE / M / GVP / CV / the dense TIR ladder are monotone
    reparameterisations of a level distribution the model already sees -- kept so
    the per-family ablation table is clinically legible, not for accuracy.
    """
    o = OrderedDict()
    W5 = (12, 24, 72, 144, 288)
    X, M = c.X, c.OBS
    for l in W5:
        s = X[:, c.L - l:]; m = M[:, c.L - l:].astype(np.float64)
        n = np.maximum(m.sum(1), 1)
        for thr, nm in ((54, "lt54"), (70, "lt70"), (80, "lt80")):
            o[f"clin__{nm}_{l}"] = _safe((((s < thr) & (m > 0)).sum(1)), n)
        o[f"clin__tir_{l}"] = _safe((((s >= 70) & (s <= 180) & (m > 0)).sum(1)), n)
        for thr, nm in ((140, "gt140"), (180, "gt180"), (250, "gt250")):
            o[f"clin__{nm}_{l}"] = _safe((((s > thr) & (m > 0)).sum(1)), n)
        o[f"clin__auc_hyper_{l}"] = _safe((np.maximum(np.nan_to_num(s) - 180, 0) * m).sum(1), n)
        o[f"clin__auc_hypo_{l}"] = _safe((np.maximum(70 - np.nan_to_num(s), 0) * m).sum(1), n)
        o[f"clin__cv_{l}"] = _safe(c.std(l), np.maximum(c.mean(l), 1e-6))
        g = np.where(m > 0, np.maximum(np.nan_to_num(s), 1.0), np.nan)
        with np.errstate(all="ignore"):
            f = 1.509 * (np.power(np.log(g), 1.084) - 5.381)
            rl = np.where(f < 0, 10 * f * f, 0.0)
            rh = np.where(f > 0, 10 * f * f, 0.0)
            o[f"clin__lbgi_{l}"] = np.nan_to_num(np.nanmean(rl, axis=1))
            o[f"clin__hbgi_{l}"] = np.nan_to_num(np.nanmean(rh, axis=1))
            o[f"clin__rlmax_{l}"] = np.nan_to_num(np.nanmax(np.nan_to_num(rl), axis=1))
            o[f"clin__rhmax_{l}"] = np.nan_to_num(np.nanmax(np.nan_to_num(rh), axis=1))
            gr = 425 * (np.log10(np.maximum(np.log10(np.maximum(g / 18.0, 1.001)), 1e-6)) + 0.16) ** 2
            o[f"clin__grade_{l}"] = np.nan_to_num(np.nanmean(gr, axis=1))
            mv = np.abs(10 * np.log10(np.maximum(g, 1) / 120.0)) ** 3
            o[f"clin__mvalue_{l}"] = np.nan_to_num(np.nanmean(mv, axis=1))
        dd = np.diff(np.nan_to_num(s), axis=1)
        o[f"clin__gvp_{l}"] = 100 * (_safe(np.sqrt(25 + dd * dd).sum(1), 5.0 * l) - 1)
    # MAGE-like excursion amplitude/rate. NOT MAGE: real MAGE needs turning-point
    # detection, which is ragged per anchor and has no fixed-shape numpy form.
    # Build the lag-6 excursion series ONCE over the whole window, then slice it.
    # Slicing X twice with an offset breaks at l == L (the lagged slice is empty),
    # which is exactly the 24 h case that matters most.
    e_all = np.abs(np.nan_to_num(c.X[:, 6:]) - np.nan_to_num(c.X[:, :-6]))
    for l in (72, 144, 288):
        e = e_all[:, -min(l, e_all.shape[1]):]
        big = e > c.std(l)[:, None]
        o[f"clin__mage_amp_{l}"] = _safe((e * big).sum(1), np.maximum(big.sum(1), 1))
        o[f"clin__mage_rate_{l}"] = _safe(big.sum(1), float(e.shape[1]))
    for n_ in (12, 24, 48):
        dd = np.nan_to_num(c.X[:, n_:]) - np.nan_to_num(c.X[:, :-n_])
        o[f"clin__conga_{n_}"] = dd.std(1)
    return o


# --------------------------------------------------------------------- F4..F6
def f_ins(c: Ctx) -> "OrderedDict[str, np.ndarray]":
    """Insulin: sums, counts, PK kernels, and the basal-suspend block.

    TIER 1 for two sub-blocks. (a) The store's exposure channels all decay
    MONOTONICALLY from tau=0, but real insulin ACTION peaks at 55-90 min -- a
    non-monotone kernel the backbone would have to discover with no inductive
    bias pushing it there. (b) 85.8% of this corpus is closed-loop, and an AID
    pump SUSPENDS BASAL ahead of predicted hypoglycaemia, so the longest
    zero-basal run is a readout of the controller's own forecast -- via a
    max-over-run-lengths op that convolutions and SSMs represent poorly.
    """
    o = OrderedDict()
    for l in WIN7:
        o[f"ins__bolus_sum_{l}"] = c.wsum(BOLUS, l)
        o[f"ins__basal_sum_{l}"] = c.wsum(BASAL, l)
        o[f"ins__bolus_n_{l}"] = c.wcount(BOLUS, l)
        o[f"ins__bolus_max_{l}"] = np.nan_to_num(c.W[:, c.L - l:, BOLUS]).max(1)
        b, ba = c.wsum(BOLUS, l), c.wsum(BASAL, l)
        o[f"ins__bolus_frac_{l}"] = _safe(b, b + ba + 1e-6)
    for ch, nm in list(zip(EXP_BOLUS, ("e30", "e60", "e120", "e240"))) + \
                  list(zip(EXP_BASAL, ("be60", "be240"))):
        o[f"ins__{nm}_now"] = np.nan_to_num(c.W[:, -1, ch]).astype(np.float64)
        for l in (12, 72, 288):
            o[f"ins__{nm}_max_{l}"] = np.nan_to_num(c.W[:, c.L - l:, ch]).max(1)
        for l in (72, 288):
            o[f"ins__{nm}_mean_{l}"] = _safe(c.wsum(ch, l), float(l))
    for tp, td in ((55, 180), (75, 240), (75, 300), (90, 360)):
        for ch, cn in ((BOLUS, "bol"), (BASAL, "bas")):
            o[f"ins__iob_{cn}_{tp}_{td}"] = c.conv(ch, "iob", tp, td)
            o[f"ins__act_{cn}_{tp}_{td}"] = c.conv(ch, "act", tp, td)
    tdd = c.wsum(BOLUS, c.L) + c.wsum(BASAL, c.L)
    o["ins__tdd"] = tdd
    for tp, td in ((55, 180), (75, 240), (75, 300), (90, 360)):
        o[f"ins__iobfrac_{tp}_{td}"] = _safe(c.conv(BOLUS, "iob", tp, td), np.maximum(tdd, 1e-3))
    o["ins__since_now"] = np.nan_to_num(c.W[:, -1, BOLUS_SINCE]).astype(np.float64)
    for thr in (0.5, 1.0, 2.0):
        o[f"ins__since_ge{thr}"] = np.minimum(
            _runlen_since(np.nan_to_num(c.W[:, :, BOLUS]) >= thr) * GRID, 1440.0)
    for thr in (2.0, 4.0):
        for l in (72, 288):
            o[f"ins__n_gt{int(thr)}U_{l}"] = c.wcount(BOLUS, l, thr)
    zero = np.nan_to_num(c.W[:, :, BASAL]) <= 1e-9
    for l in (12, 72, 288):
        o[f"ins__basal_zero_frac_{l}"] = _safe(zero[:, c.L - l:].sum(1), float(l))
    o["ins__basal_zero_run"] = _longest_run(zero).astype(np.float64)
    # Basal modulation vs this window's own typical rate. Clipped: a suspended
    # pump gives a zero median, and an unclipped ratio then reports 1e6 instead of
    # "the pump is running well above its usual rate", which is all it can mean.
    med = np.median(np.nan_to_num(c.W[:, :, BASAL]), axis=1)
    for l in (6, 12, 36):
        o[f"ins__basal_mod_{l}"] = np.clip(
            _safe(_safe(c.wsum(BASAL, l), float(l)), np.maximum(med, 1e-3)), 0.0, 20.0)
    for l in (12, 36, 72):
        m = (np.nan_to_num(c.W[:, c.L - l:, BASAL_M]) > .5).astype(np.float64)
        y = np.nan_to_num(c.W[:, c.L - l:, BASAL]).astype(np.float64)
        t = np.arange(l, dtype=np.float64)[None, :]
        Sw = m.sum(1); St = (m * t).sum(1); Stt = (m * t * t).sum(1)
        Sy = (m * y).sum(1); Sty = (m * t * y).sum(1)
        o[f"ins__basal_slope_{l}"] = _safe(Sw * Sty - St * Sy, Sw * Stt - St * St)
    o["ins__basal_miss"] = 1.0 - c.cov(BASAL_M, c.L)
    o["ins__bolus_miss"] = 1.0 - c.cov(BOLUS_M, c.L)
    o["ins__absent"] = (c.cov(BASAL_M, c.L) + c.cov(BOLUS_M, c.L) <= 0).astype(np.float64)
    return o


def f_carb(c: Ctx) -> "OrderedDict[str, np.ndarray]":
    """Carbohydrate, plus the logging gate.

    TIER 1 for `no_carb_logging` and the COB kernels. DCLP3/DCLP5/Flair and the
    IOBP2 internal holdout log NO carbohydrate at all, so without an explicit
    gate the model cannot tell "ate nothing" from "records nothing" -- and the
    headline holdout is exactly the no-carb cohort.
    """
    o = OrderedDict()
    for l in WIN7:
        o[f"carb__sum_{l}"] = c.wsum(CARB, l)
        o[f"carb__n_{l}"] = c.wcount(CARB, l)
        o[f"carb__max_{l}"] = np.nan_to_num(c.W[:, c.L - l:, CARB]).max(1)
    o["carb__since_now"] = np.nan_to_num(c.W[:, -1, CARB_SINCE]).astype(np.float64)
    for thr in (15.0, 30.0, 50.0):
        o[f"carb__since_ge{int(thr)}"] = np.minimum(
            _runlen_since(np.nan_to_num(c.W[:, :, CARB]) >= thr) * GRID, 1440.0)
    g = np.nan_to_num(c.W[:, :, CARB])
    idx = np.where(g > 0, np.arange(c.L)[None, :], 0)
    last = np.maximum.accumulate(idx, axis=1)[:, -1]
    lastg = np.take_along_axis(g, last[:, None], 1)[:, 0]
    o["carb__last_g"] = lastg
    none = (g.sum(1) <= 0)
    o["carb__bucket_none"] = none.astype(np.float64)
    o["carb__bucket_le15"] = ((~none) & (lastg <= 15)).astype(np.float64)
    o["carb__bucket_15_30"] = ((lastg > 15) & (lastg <= 30)).astype(np.float64)
    o["carb__bucket_30_60"] = ((lastg > 30) & (lastg <= 60)).astype(np.float64)
    o["carb__bucket_gt60"] = (lastg > 60).astype(np.float64)
    for tp, td in ((30, 120), (45, 180), (60, 240), (90, 300)):
        o[f"carb__cob_{tp}_{td}"] = c.conv(CARB, "cob", tp, td)
        o[f"carb__ra_{tp}_{td}"] = c.conv(CARB, "ra", tp, td)
    for ch, nm in zip(EXP_CARB, ("c20", "c45", "c90")):
        o[f"carb__{nm}_now"] = np.nan_to_num(c.W[:, -1, ch]).astype(np.float64)
        for l in (12, 72, 288):
            o[f"carb__{nm}_max_{l}"] = np.nan_to_num(c.W[:, c.L - l:, ch]).max(1)
        for l in (72, 288):
            o[f"carb__{nm}_mean_{l}"] = _safe(c.wsum(ch, l), float(l))
    for nm, lo_, hi_ in (("bfast", 6 * 60, 11 * 60), ("lunch", 11 * 60, 15 * 60),
                         ("dinner", 17 * 60, 22 * 60), ("night", 22 * 60, 6 * 60)):
        if lo_ < hi_:
            sel = (c.tod_row >= lo_) & (c.tod_row < hi_)
        else:
            sel = (c.tod_row >= lo_) | (c.tod_row < hi_)
        o[f"carb__n_{nm}"] = ((g > 0) & sel).sum(1).astype(np.float64)
    # Effective carb ratio. Denominator floored at 1 U, not 1e-6: grams-per-unit
    # is undefined when no insulin was given, and 2.5e8 is not a carb ratio.
    for l in (36, 72, 288):
        o[f"carb__ratio_{l}"] = np.clip(
            _safe(c.wsum(CARB, l), c.wsum(BOLUS, l) + 1.0), 0.0, 100.0)
    o["carb__bolus_gap"] = c.carb_since0 - c.bolus_since0
    o["carb__bolus_after"] = (c.bolus_since0 < c.carb_since0).astype(np.float64)
    o["carb__miss"] = 1.0 - c.cov(CARB_M, c.L)
    o["carb__no_logging"] = ((g.sum(1) <= 0) & (c.cov(CARB_M, c.L) <= 0)).astype(np.float64)
    return o


def f_act(c: Ctx) -> "OrderedDict[str, np.ndarray]":
    """Activity, with coverage flags that are load-bearing rather than decorative.

    TIER 2, and for a structural reason: heart rate and steps are absent for most
    cohorts and ENTIRELY absent from the IOBP2 holdout, so this family's ceiling
    is "helps on cohorts you are not scored on". Its coverage flags will look
    important under permutation testing purely because they are cohort proxies --
    that is not signal.

    Absent-modality rule: where coverage is 0 every HR column emits 0 bpm, which
    is physiologically impossible, so (0, cov=0) is unambiguous. Steps emit 0 too,
    where 0 IS legal -- hence steps features must never be used without their
    coverage column in the same subset.
    """
    o = OrderedDict()
    hr = np.nan_to_num(c.W[:, :, HR]).astype(np.float64)
    hm = (np.nan_to_num(c.W[:, :, HR_M]) > .5).astype(np.float64)
    stp = np.nan_to_num(c.W[:, :, STEPS]).astype(np.float64)
    sm = (np.nan_to_num(c.W[:, :, STEPS_M]) > .5).astype(np.float64)
    for l in (6, 12, 36, 72, 288):
        h, m = hr[:, c.L - l:], hm[:, c.L - l:]
        n = np.maximum(m.sum(1), 1)
        mu = _safe((h * m).sum(1), n)
        o[f"act__hr_mean_{l}"] = mu
        o[f"act__hr_max_{l}"] = (h * m).max(1)
        o[f"act__hr_min_{l}"] = np.where(m.sum(1) > 0,
                                         np.where(m > 0, h, 1e9).min(1), 0.0)
        o[f"act__hr_std_{l}"] = np.sqrt(np.maximum(_safe((h * h * m).sum(1), n) - mu * mu, 0))
    base = _safe((hr * hm).sum(1), np.maximum(hm.sum(1), 1))
    for l in (6, 12, 36):
        o[f"act__hr_rel_{l}"] = _safe(_safe((hr[:, c.L - l:] * hm[:, c.L - l:]).sum(1),
                                            np.maximum(hm[:, c.L - l:].sum(1), 1)),
                                      np.maximum(base, 1e-6))
    with np.errstate(all="ignore"):
        rest = np.nan_to_num(np.nanquantile(np.where(hm > 0, hr, np.nan), 0.1, axis=1))
    o["act__hr_above_rest"] = np.where(hm.sum(1) > 0, hr[:, -1] - rest, 0.0)
    for thr in (100, 120, 140):
        for l in (36, 72, 288):
            m = hm[:, c.L - l:]
            o[f"act__hr_gt{thr}_{l}"] = _safe(((hr[:, c.L - l:] > thr) * m).sum(1),
                                              np.maximum(m.sum(1), 1))
    for l in (6, 12, 36, 72, 288):
        o[f"act__steps_sum_{l}"] = (stp[:, c.L - l:] * sm[:, c.L - l:]).sum(1)
    for l in (36, 288):
        o[f"act__steps_max_{l}"] = (stp[:, c.L - l:] * sm[:, c.L - l:]).max(1)
        o[f"act__steps_pos_{l}"] = _safe(((stp[:, c.L - l:] > 0) * sm[:, c.L - l:]).sum(1),
                                         np.maximum(sm[:, c.L - l:].sum(1), 1))
    o["act__steps_run"] = _longest_run(stp > 0).astype(np.float64)
    o["act__since_steps"] = np.minimum(_runlen_since(stp > 50) * GRID, 1440.0)
    o["act__since_hr120"] = np.minimum(_runlen_since((hr > 120) & (hm > 0)) * GRID, 1440.0)
    tau = (c.L - 1 - np.arange(c.L)).astype(np.float64) * GRID
    for hl in (30.0, 60.0, 120.0):
        k = np.exp(-tau / hl).astype(np.float64)
        o[f"act__steps_exp{int(hl)}"] = (stp * sm) @ k
        o[f"act__hrx_exp{int(hl)}"] = (np.maximum(hr - rest[:, None], 0) * hm) @ k
    for l in (36, 288):
        o[f"act__hr_cov_{l}"] = c.cov(HR_M, l)
        o[f"act__steps_cov_{l}"] = c.cov(STEPS_M, l)
    o["act__absent"] = ((hm.sum(1) + sm.sum(1)) <= 0).astype(np.float64)
    return o


# -------------------------------------------------------------------- F7..F12
def f_qual(c: Ctx) -> "OrderedDict[str, np.ndarray]":
    """Data quality and missingness.

    TIER 1 apart from the device sub-block: run-length maxima over the mask
    channel are an unusual op for a conv/SSM, and they identify windows where the
    LABEL itself is unreliable.

    SHORTCUT HAZARD, stated rather than buried: `int_frac` and the clamp counts
    are device/era fingerprints and correlate with source_file, which this repo
    deliberately withholds from the primary model. They can raise val and only
    hurt the unseen-source holdout. They live behind their own `qual_device`
    family so the sensitivity analysis can drop them with one flag.
    """
    o = OrderedDict()
    for l in (6, 12, 24, 72, 144, 288):
        o[f"qual__real_frac_{l}"] = _safe(c.REAL[:, c.L - l:].sum(1), float(l))
    for l in (12, 24, 72, 288):
        o[f"qual__interp_run_{l}"] = _longest_run(~c.REAL[:, c.L - l:]).astype(np.float64)
    o["qual__since_real"] = _runlen_since(c.REAL).astype(np.float64)
    tr = (c.REAL[:, 1:].astype(np.int8) - c.REAL[:, :-1].astype(np.int8)) > 0
    for l in (24, 72, 288):
        o[f"qual__gaps_{l}"] = tr[:, -l:].sum(1).astype(np.float64)
    o["qual__n_valid"] = c.OBS.sum(1).astype(np.float64)
    for ch, nm in ((BASAL_M, "basal"), (BOLUS_M, "bolus"), (CARB_M, "carb"),
                   (HR_M, "hr"), (STEPS_M, "steps")):
        o[f"qual__cov_{nm}"] = c.cov(ch, c.L)
    for ch, nm in ((BOLUS, "bolus"), (BASAL, "basal"), (CARB, "carb"),
                   (STEPS, "steps"), (HR, "hr")):
        o[f"qual__allzero_{nm}"] = (np.nan_to_num(c.W[:, :, ch]).sum(1) <= 0).astype(np.float64)
    return o


def f_qual_device(c: Ctx) -> "OrderedDict[str, np.ndarray]":
    """Device/era fingerprints. Isolated ON PURPOSE -- see f_qual's docstring."""
    o = OrderedDict()
    X = np.nan_to_num(c.X)
    o["qdev__int_frac"] = _safe((np.abs(X - np.round(X)) < 1e-6).sum(1), float(c.L))
    o["qdev__n_clamp_lo"] = (X <= 40).sum(1).astype(np.float64)
    o["qdev__n_clamp_hi"] = (X >= 400).sum(1).astype(np.float64)
    o["qdev__stuck_run"] = _longest_run(np.abs(np.diff(X, axis=1)) < 1e-9).astype(np.float64)
    d = np.abs(np.diff(X, axis=1))
    boundary = (c.REAL[:, 1:].astype(np.int8) - c.REAL[:, :-1].astype(np.int8)) > 0
    o["qdev__max_jump_boundary"] = (d * boundary).max(1)
    o["qdev__big_jump"] = ((d * boundary) >= 20).any(1).astype(np.float64)
    return o


def f_cal(c: Ctx) -> "OrderedDict[str, np.ndarray]":
    """Circadian and calendar, including day-of-week and is-weekend.

    HONEST CAVEAT: windows.py ALREADY emits [sin tod, cos tod, sin dow, cos dow]
    per timestep. Only the interactions and the two nonlinear timers (minutes to
    06:00, daypart one-hot) are new information; the encodings themselves are
    duplicates under a different name. The 7-way dow one-hot is deliberately NOT
    built -- the real effect is weekend vs weekday meal and exercise timing, and a
    7-way split multiplies dimension for effects below the 0.6-point noise floor.
    """
    o = OrderedDict()
    for k in (1, 2, 3):
        o[f"cal__tod_sin{k}"] = np.sin(2 * np.pi * k * c.tod / 1440.0)
        o[f"cal__tod_cos{k}"] = np.cos(2 * np.pi * k * c.tod / 1440.0)
    o["cal__dow_sin"] = np.sin(2 * np.pi * c.dow / 7.0)
    o["cal__dow_cos"] = np.cos(2 * np.pi * c.dow / 7.0)
    wknd = (c.dow >= 5).astype(np.float64)
    o["cal__is_weekend"] = wknd
    # Explicit 7-way day-of-week. The analysis argues the real effect is weekend
    # vs weekday (meal and exercise timing) and that a 7-way split multiplies
    # dimension for effects below the 0.6-point noise floor -- but it was asked
    # for by name, it is cheap, and `cal` is an ablatable family, so the
    # leave-one-family-out table will settle it rather than an argument.
    for dd in range(7):
        o[f"cal__dow_{dd}"] = (c.dow == dd).astype(np.float64)
    for nm, lo_, hi_ in (("night", 0, 6), ("morn", 6, 11), ("mid", 11, 17),
                         ("eve", 17, 22), ("late", 22, 24)):
        o[f"cal__part_{nm}"] = ((c.tod >= lo_ * 60) & (c.tod < hi_ * 60)).astype(np.float64)
    o["cal__tod_min"] = c.tod
    o["cal__to_dawn"] = (360.0 - c.tod) % 1440.0
    o["cal__is_overnight"] = ((c.tod < 360) | (c.tod >= 1320)).astype(np.float64)
    s1, c1 = o["cal__tod_sin1"], o["cal__tod_cos1"]
    sl6, _ = c.slope_r2(6)
    iob = c.conv(BOLUS, "iob", 75, 300)
    cob = c.conv(CARB, "cob", 45, 180)
    for nm, v in (("cgm", c.cgm_now), ("slope6", sl6), ("iob", iob), ("cob", cob)):
        o[f"cal__sin1_x_{nm}"] = s1 * v
        o[f"cal__cos1_x_{nm}"] = c1 * v
    for nm, v in (("cgm", c.cgm_now), ("carb288", c.wsum(CARB, c.L)),
                  ("bolus288", c.wsum(BOLUS, c.L)), ("mean288", c.mean(c.L))):
        o[f"cal__wknd_x_{nm}"] = wknd * v
    return o


def f_lag24(c: Ctx) -> "OrderedDict[str, np.ndarray]":
    """Same-time-yesterday and long lags -- and the exact reachability limit.

    The window spans t-1435 .. t, so W[:, 0] is t-1435, NOT t-1440: lag-288 lands
    ONE STEP SHORT of 24 h and there is no exact same-time-yesterday sample in a
    288-step window. The closest matched-clock band is rows 0..11 (23h00-23h55
    ago) and it is ONE-SIDED -- the hour before this time yesterday, never after.

    Consequences, stated so nobody computes a surrogate and names it wrongly:
    MODD needs >= 48 h and is NOT computable here; ADRR needs 14 days and is not
    either (f_cgm_clin's rlmax/rhmax are within-window surrogates, named as such);
    nothing day-over-day is computable. All of it becomes reachable at
    context=576/864/1152, stores that already exist. If THIS family scores, that
    is the argument for rebuilding at a longer context -- which is the only
    reason it is worth building at 288.
    """
    o = OrderedDict()
    head = c.X[:, :12]; hm = c.OBS[:, :12].astype(np.float64)
    n = np.maximum(hm.sum(1), 1)
    mu = _safe((np.nan_to_num(head) * hm).sum(1), n)
    o["lag__head_mean"] = mu
    o["lag__head_std"] = np.sqrt(np.maximum(
        _safe((np.nan_to_num(head) ** 2 * hm).sum(1), n) - mu * mu, 0))
    o["lag__head_min"] = np.where(hm.sum(1) > 0, np.where(hm > 0, np.nan_to_num(head), 1e9).min(1), 0)
    o["lag__head_max"] = (np.nan_to_num(head) * hm).max(1)
    o["lag__oldest"] = np.nan_to_num(c.X[:, 0])
    o["lag__now_minus_head"] = c.cgm_now - mu
    o["lag__now_minus_oldest"] = c.cgm_now - np.nan_to_num(c.X[:, 0])
    h24 = c.X[:, :24]; m24 = c.OBS[:, :24].astype(np.float64)
    n24 = np.maximum(m24.sum(1), 1)
    mu24 = _safe((np.nan_to_num(h24) * m24).sum(1), n24)
    o["lag__h24_mean"] = mu24
    o["lag__h24_std"] = np.sqrt(np.maximum(
        _safe((np.nan_to_num(h24) ** 2 * m24).sum(1), n24) - mu24 * mu24, 0))
    night = (c.tod_row < 360) & c.OBS
    nn = np.maximum(night.sum(1), 1)
    nv = np.nan_to_num(c.X) * night
    nmu = _safe(nv.sum(1), nn)
    o["lag__night_mean"] = nmu
    o["lag__night_std"] = np.sqrt(np.maximum(_safe((nv ** 2).sum(1), nn) - nmu * nmu, 0))
    o["lag__night_min"] = np.where(night.sum(1) > 0,
                                   np.where(night, np.nan_to_num(c.X), 1e9).min(1), 0.0)
    o["lag__night_lt70"] = _safe(((np.nan_to_num(c.X) < 70) & night).sum(1), nn)
    o["lag__night_gt180"] = _safe(((np.nan_to_num(c.X) > 180) & night).sum(1), nn)
    # NOT night.sum()/L: at context=288 every window contains exactly 6 h of
    # night, so that is the constant 0.25. What varies is how much of the night
    # block was actually observed.
    # Two traps here, both hit in practice. night.sum()/L is the constant 0.25 at
    # context=288 (every 24 h window holds exactly 6 h of night). And OBSERVED
    # night fraction is the constant 1.0, because CGM is interpolated across gaps
    # <= 30 min so OBS is almost always true. What actually varies is how much of
    # the night block is a REAL reading rather than an interpolated one.
    night_slots = (c.tod_row < 360).sum(1)
    o["lag__night_cov"] = _safe(((c.tod_row < 360) & c.REAL).sum(1),
                                np.maximum(night_slots, 1))
    o["lag__h24_carb"] = np.nan_to_num(c.W[:, :24, CARB]).sum(1)
    o["lag__h24_bolus"] = np.nan_to_num(c.W[:, :24, BOLUS]).sum(1)
    xc = np.nan_to_num(c.X) - c.mean(c.L)[:, None]
    den = np.maximum((xc * xc).sum(1), 1e-6)
    acf = {}
    for k in (6, 12, 36, 72, 144):
        acf[k] = _safe((xc[:, k:] * xc[:, :-k]).sum(1), den)
        o[f"lag__acf_{k}"] = acf[k]
    o["lag__acf_argmax"] = np.array([12, 36, 72], dtype=np.float64)[
        np.argmax(np.stack([acf[12], acf[36], acf[72]], 1), axis=1)]
    return o


def f_xmod(c: Ctx) -> "OrderedDict[str, np.ndarray]":
    """Cross-modality products and ratios. The strongest theoretical case here.

    ReLU-MLP and additive-fusion architectures approximate x*y and x/y badly, and
    models/frontend.py fuses by CGM-centred CONCATENATION, not multiplicative
    gating -- FiLM modulates the CGM state by a context vector, it does not form
    products between arbitrary driver features. `g - ISF*iob + CR*cob` is the
    clinical bolus calculator in one column.
    """
    o = OrderedDict()
    g = c.cgm_now
    s6, _ = c.slope_r2(6); s12, _ = c.slope_r2(12); s24, _ = c.slope_r2(24)
    iob = c.conv(BOLUS, "iob", 75, 300)
    iob_all = iob + c.conv(BASAL, "iob", 75, 300)
    cob = c.conv(CARB, "cob", 45, 180)
    ISF, CR = _isf_cr(c.wsum(BOLUS, c.L) + c.wsum(BASAL, c.L))
    for nm, s in (("s6", s6), ("s12", s12), ("s24", s24)):
        o[f"x__{nm}_x_iob"] = s * iob
        o[f"x__{nm}_x_ioball"] = s * iob_all
        o[f"x__{nm}_x_cob"] = s * cob
    o["x__iob_over_gap"] = _safe(iob, np.maximum(g - 70, 1.0))
    o["x__g_m30iob"] = g - 30 * iob
    o["x__g_m50iob"] = g - 50 * iob
    o["x__g_mISFiob"] = g - ISF * iob
    o["x__nadir"] = g - ISF * iob + CR * cob
    for tp, td in ((55, 180), (75, 240), (75, 300), (90, 360)):
        ib = c.conv(BOLUS, "iob", tp, td)
        o[f"x__nadir_{tp}_{td}"] = g - ISF * ib
    for tp, td in ((30, 120), (45, 180), (60, 240), (90, 300)):
        cb = c.conv(CARB, "cob", tp, td)
        o[f"x__net_load_{tp}_{td}"] = CR * cb - ISF * iob
    o["x__cob_decay"] = cob * np.exp(-np.minimum(c.carb_since0, 1440) / 60.0)
    lastg = np.take_along_axis(
        np.nan_to_num(c.W[:, :, CARB]),
        np.maximum.accumulate(np.where(np.nan_to_num(c.W[:, :, CARB]) > 0,
                                       np.arange(c.L)[None, :], 0), axis=1)[:, -1:], 1)[:, 0]
    o["x__since_x_lastg"] = c.carb_since0 * lastg
    o["x__lastg_rate"] = _safe(lastg, c.carb_since0 + 5.0)
    steps_exp = np.nan_to_num(c.W[:, :, STEPS]) @ np.exp(
        -((c.L - 1 - np.arange(c.L)) * GRID / 60.0))
    gate = (c.cov(STEPS_M, c.L) > 0).astype(np.float64)
    o["x__act_x_iob"] = steps_exp * iob * gate
    o["x__act_x_g"] = steps_exp * g * gate
    o["x__act_x_s6"] = steps_exp * s6 * gate
    o["x__hypo_comp"] = ((g < 90) & (s6 < 0)).astype(np.float64) * iob
    o["x__lt80_x_iob"] = _safe((np.nan_to_num(c.X[:, c.L - 24:]) < 80).sum(1), 24.0) * iob
    for l in (36, 72):
        d = c.cgm_now - np.nan_to_num(c.X[:, c.L - 1 - l])
        # mg/dL moved per unit delivered; floored at 0.5 U and clipped to the
        # range an ISF can plausibly take, for the same reason as _isf_cr.
        o[f"x__sens_{l}"] = np.clip(_safe(d, c.wsum(BOLUS, l) + 0.5), -400.0, 400.0)
        o[f"x__mismatch_{l}"] = CR * c.wsum(CARB, l) - c.wsum(BOLUS, l)
    for l in (12, 24):
        d = c.cgm_now - np.nan_to_num(c.X[:, c.L - 1 - l])
        o[f"x__carbresp_{l}"] = _safe(d, c.wsum(CARB, l) + 1.0)
    on = ((c.tod < 360) | (c.tod >= 1320)).astype(np.float64)
    o["x__since_x_night"] = c.carb_since0 * on
    o["x__iob_x_night"] = iob * on
    o["x__std72_x_iob"] = c.std(72) * iob
    o["x__norm_x_s6"] = ((g - 70.0) / 110.0) * s6
    return o


def f_rel(c: Ctx) -> "OrderedDict[str, np.ndarray]":
    """Patient-relative / self-normalised.

    TIER 1, and the family most likely to transfer to the unseen-source holdout:
    the global standardiser centres on a CORPUS median, while these centre on the
    WINDOW's own distribution, so they are invariant to a patient- or
    cohort-level shift -- exactly the failure mode the IOBP2 holdout tests. The
    percentile rank also requires a sort, which no backbone computes.
    """
    o = OrderedDict()
    g = c.cgm_now
    for l in (12, 24, 72, 144, 288):
        o[f"rel__z_{l}"] = _safe(g - c.mean(l), np.maximum(c.std(l), 1e-6))
    for l in (72, 144, 288):
        sl = c.slice(l); m = c.OBS[:, c.L - l:]
        o[f"rel__pct_{l}"] = _safe(((np.nan_to_num(sl) <= g[:, None]) & m).sum(1),
                                   np.maximum(m.sum(1), 1))
    for l in (72, 288):
        iqr = c.q(l, .75) - c.q(l, .25)
        o[f"rel__riqr_{l}"] = _safe(g - c.q(l, .5), np.maximum(iqr, 1e-6))
    o["rel__g_over_mean288"] = _safe(g, np.maximum(c.mean(288), 1e-6))
    o["rel__g_over_q90"] = _safe(g, np.maximum(c.q(288, .9), 1e-6))
    o["rel__g_over_q10"] = _safe(g, np.maximum(c.q(288, .1), 1e-6))
    for a_, b_ in ((12, 288), (36, 288), (72, 288), (12, 72)):
        o[f"rel__mean_{a_}_{b_}"] = _safe(c.mean(a_), np.maximum(c.mean(b_), 1e-6))
    for a_ in (12, 36, 72):
        o[f"rel__std_{a_}_288"] = _safe(c.std(a_), np.maximum(c.std(288), 1e-6))
    for a_ in (6, 12, 24):
        s, _ = c.slope_r2(a_)
        o[f"rel__slope_{a_}_std288"] = _safe(s, np.maximum(c.std(288), 1e-6))
    o["rel__max_minus_g"] = c.amax(288) - g
    o["rel__g_minus_min"] = g - c.amin(288)
    rng = np.maximum(c.amax(288) - c.amin(288), 1e-6)
    o["rel__max_minus_g_n"] = _safe(c.amax(288) - g, rng)
    o["rel__g_minus_min_n"] = _safe(g - c.amin(288), rng)
    o["rel__mean288_minus_154"] = c.mean(288) - 154.0
    tir288 = _safe(((np.nan_to_num(c.X) >= 70) & (np.nan_to_num(c.X) <= 180) & c.OBS).sum(1),
                   np.maximum(c.OBS.sum(1), 1))
    s72 = c.X[:, c.L - 72:]; m72 = c.OBS[:, c.L - 72:]
    tir72 = _safe(((np.nan_to_num(s72) >= 70) & (np.nan_to_num(s72) <= 180) & m72).sum(1),
                  np.maximum(m72.sum(1), 1))
    o["rel__tir_drift"] = tir288 - tir72
    o["rel__tir_72"] = tir72
    s6a, _ = c.slope_r2(6)
    dall = np.abs(np.diff(np.nan_to_num(c.X), axis=1))
    o["rel__slope_pct"] = _safe((dall <= np.abs(s6a)[:, None]).sum(1), float(c.L - 1))
    o["rel__range_over_mean"] = _safe(rng, np.maximum(c.mean(288), 1e-6))
    o["rel__iqr_over_mean"] = _safe(c.q(288, .75) - c.q(288, .25),
                                    np.maximum(c.mean(288), 1e-6))
    return o


def f_stat(c: Ctx) -> "OrderedDict[str, np.ndarray]":
    """Statics and static x dynamic interactions.

    TIER 2 and partly REDUNDANT BY CONSTRUCTION: the six raw numerics and the two
    categoricals are already in item["static_num"]/["static_cat"] for any run with
    use_static: true, and this repo measured the whole static modality at 0.24
    points. The plausible fraction is the TDD-derived personal constants (ISF/CR)
    and the interactions -- both of which are ratios and products, i.e. the F10
    argument. Exclude this family for the neural arm to avoid double-feeding;
    keep it for a GBDT arm, which has no static path.
    """
    o = OrderedDict()
    n = c.n
    S = c.statics
    if S is None:
        S = np.full((n, 6), np.nan)
    age, dx, dur = S[:, 0], S[:, 1], S[:, 2]
    h, w, bmi = S[:, 3], S[:, 4], S[:, 5]
    for nm, v in (("age", age), ("age_dx", dx), ("duration", dur),
                  ("height", h), ("weight", w), ("bmi", bmi)):
        o[f"stat__{nm}"] = np.nan_to_num(v)
        o[f"stat__{nm}_isnan"] = (~np.isfinite(v)).astype(np.float64)
    o["stat__dx_over_age"] = _safe(np.nan_to_num(dx), np.maximum(np.nan_to_num(age), 1e-6))
    o["stat__dur_over_age"] = _safe(np.nan_to_num(dur), np.maximum(np.nan_to_num(age), 1e-6))
    o["stat__bsa"] = 0.007184 * np.power(np.nan_to_num(h, nan=1.7) * 100, 0.725) \
        * np.power(np.nan_to_num(w, nan=70.0), 0.425)
    o["stat__log_weight"] = np.log1p(np.nan_to_num(w))
    o["stat__is_paed"] = (np.nan_to_num(age, nan=99) < 18).astype(np.float64)
    b = np.nan_to_num(bmi, nan=25.0)
    o["stat__bmi_under"] = (b < 18.5).astype(np.float64)
    o["stat__bmi_norm"] = ((b >= 18.5) & (b < 25)).astype(np.float64)
    o["stat__bmi_over"] = ((b >= 25) & (b < 30)).astype(np.float64)
    o["stat__bmi_obese"] = (b >= 30).astype(np.float64)
    kg = np.maximum(np.nan_to_num(w, nan=70.0), 1.0)
    tdd = c.wsum(BOLUS, c.L) + c.wsum(BASAL, c.L)
    o["stat__tdd_per_kg"] = _safe(tdd, kg)
    o["stat__bolus_per_kg"] = _safe(c.wsum(BOLUS, c.L), kg)
    o["stat__carb_per_kg"] = _safe(c.wsum(CARB, c.L), kg)
    o["stat__basal_per_kg"] = _safe(c.wsum(BASAL, c.L), kg)
    o["stat__iob_per_kg"] = _safe(c.conv(BOLUS, "iob", 75, 300), kg)
    isf, cr = _isf_cr(tdd)
    o["stat__isf"] = isf
    o["stat__cr"] = cr
    o["stat__basal_frac"] = _safe(c.wsum(BASAL, c.L), np.maximum(tdd, 1e-3))
    s6, _ = c.slope_r2(6)
    tir = _safe(((np.nan_to_num(c.X) >= 70) & (np.nan_to_num(c.X) <= 180) & c.OBS).sum(1),
                np.maximum(c.OBS.sum(1), 1))
    for nm, sv in (("age", np.nan_to_num(age)), ("bmi", b), ("dur", np.nan_to_num(dur))):
        for dn, dv in (("cgm", c.cgm_now), ("std288", c.std(288)), ("tir", tir), ("s6", s6)):
            o[f"stat__{nm}_x_{dn}"] = sv * dv
    paed = o["stat__is_paed"]
    for dn, dv in (("std288", c.std(288)), ("carb288", c.wsum(CARB, c.L)), ("tir", tir)):
        o[f"stat__paed_x_{dn}"] = paed * dv
    return o


# ------------------------------------------------------------------- registry
# Order here IS the column order in features.npy. Tier is the honest prior from
# the analysis in the package docstring, not a measurement -- the experiment is
# what turns it into one, via leave-one-family-out.
FEATURE_FAMILIES = OrderedDict([
    ("cgm_level", (f_cgm_level, 2)),
    ("cgm_dyn", (f_cgm_dyn, 2)),
    ("cgm_clin", (f_cgm_clin, 1)),
    ("ins", (f_ins, 1)),
    ("carb", (f_carb, 1)),
    ("act", (f_act, 2)),
    ("qual", (f_qual, 1)),
    ("qual_device", (f_qual_device, 2)),
    ("cal", (f_cal, 2)),
    ("lag24", (f_lag24, 2)),
    ("xmod", (f_xmod, 1)),
    ("rel", (f_rel, 1)),
    ("stat", (f_stat, 2)),
])
TIER1 = [k for k, (_, t) in FEATURE_FAMILIES.items() if t == 1]


def _selected(families: Optional[List[str]]) -> List[str]:
    if not families:
        return list(FEATURE_FAMILIES)
    if list(families) == ["tier1"]:
        return list(TIER1)
    unknown = [f for f in families if f not in FEATURE_FAMILIES]
    if unknown:
        raise ValueError(f"unknown feature families {unknown}; "
                         f"have {list(FEATURE_FAMILIES)} (or 'tier1')")
    return [f for f in FEATURE_FAMILIES if f in families]


def compute_block(W: np.ndarray, anchors, index: np.ndarray,
                  families: Optional[List[str]] = None,
                  statics: Optional[np.ndarray] = None) -> np.ndarray:
    """``[n, L, 23]`` -> ``[n, F]`` float64, column order == ``build_columns``."""
    ctx = Ctx(W, anchors, index, statics)
    cols = []
    for name in _selected(families):
        fn, _ = FEATURE_FAMILIES[name]
        for v in fn(ctx).values():
            cols.append(np.asarray(v, dtype=np.float64).reshape(-1))
    return np.stack(cols, axis=1) if cols else np.zeros((W.shape[0], 0))


def build_columns(families: Optional[List[str]] = None, context: int = 288) -> List[str]:
    """The column names, in order.

    Produced by running the SAME code on a two-row dummy window rather than from a
    second hand-maintained list, so ``len(build_columns(f))`` and
    ``compute_block(..., f).shape[1]`` cannot drift apart for any subset -- which
    matters because build.py writes columns.json from one and features.npy from
    the other.
    """
    import pandas as pd
    rng = np.random.default_rng(0)
    W = np.abs(rng.normal(120, 30, size=(2, context, 23))).astype(np.float32)
    W[:, :, (CGM_M, BASAL_M, BOLUS_M, CARB_M, HR_M, STEPS_M)] = 1.0
    a = pd.DataFrame({"cgm_now": [120.0, 130.0], "tod_min": [600.0, 60.0],
                      "dow": [2.0, 6.0], "min_since_carb": [90.0, 30.0],
                      "min_since_bolus": [45.0, 200.0]})
    ctx = Ctx(W, a, np.array([0, 1]), np.full((2, 6), 30.0))
    names: List[str] = []
    for name in _selected(families):
        fn, _ = FEATURE_FAMILIES[name]
        names.extend(fn(ctx).keys())
    return names


def family_slices(families: Optional[List[str]] = None,
                  context: int = 288) -> "OrderedDict[str, slice]":
    """Column span of each family -- what leave-one-family-out needs."""
    import pandas as pd
    out: "OrderedDict[str, slice]" = OrderedDict()
    start = 0
    for name in _selected(families):
        n = len(build_columns([name], context))
        out[name] = slice(start, start + n)
        start += n
    return out


def select_columns(root, families: str = "all", context: int = 288) -> np.ndarray:
    """Column indices in a BUILT matrix for the requested families.

    The matrix is built once with every family, so an ablation selects columns
    rather than rebuilding -- every arm then reads the identical file and differs
    only in which columns it is shown. Used by BOTH the dataset and the
    standardiser fit, so the two can never disagree about the width.
    """
    import json
    from pathlib import Path as _P
    root = _P(root)
    built = json.load(open(root / "columns.json"))
    want = str(families or "all").strip()
    spans = family_slices(None, context)
    if sum(sl.stop - sl.start for sl in spans.values()) != len(built):
        raise ValueError(
            f"{root/'columns.json'} has {len(built)} columns but the current spec "
            f"produces {sum(sl.stop-sl.start for sl in spans.values())}; the "
            "feature matrix was built by a different version of spec.py -- rebuild it")
    if want in ("all", "*", ""):
        return np.arange(len(built), dtype=np.int64)
    sel = list(TIER1) if want == "tier1" else [f.strip() for f in want.split(",")]
    missing = [f for f in sel if f not in spans]
    if missing:
        raise ValueError(f"data.features.families names {missing}; built families "
                         f"are {list(spans)}")
    return np.sort(np.concatenate(
        [np.arange(spans[f].start, spans[f].stop) for f in sel])).astype(np.int64)
