"""TabPFN -- a prior-fitted transformer that does in-context regression.

TabPFN does not scale to this corpus as-is. Its pretraining regime assumes roughly
<= 10,000 training rows and <= 500 features, and inference cost is O(n_train x
n_test), so 10 M training rows and millions of eval rows are both out of reach.

This adapter makes it usable and honest about the compromise:

1. **Bagged context subsets.** Fit `n_subsets` independent TabPFN models, each on a
   subject-stratified random subsample of `subset_size` rows, and average their
   predictions. This is the standard way to scale TabPFN and it also damps the
   variance of any single context draw.
2. **Feature selection.** Ranked by |Spearman correlation| with the target on a
   sample, keeping `max_features`. `cgm_now` is always kept -- EDA 9.6 shows
   persistence alone is a strong predictor, so losing it would be perverse.
3. **Capped evaluation.** Predicting on every eval row is prohibitive, so the
   adapter scores a random subsample by default. `notes["eval_subsampled"]` records
   this; do not compare its MARD to a full-set MARD without noting the difference.

Version note: `pip install tabpfn==2.2.1`. The 8.x line requires accepting a licence
interactively and setting `TABPFN_TOKEN` before it will download weights.
"""
from __future__ import annotations

import numpy as np

from .base import BaseModel, FitContext


class TabPFNModel(BaseModel):
    name = "tabpfn"
    handles_nan = True          # TabPFN's preprocessor handles NaN internally

    def __init__(self, n_subsets: int = 4, subset_size: int = 10_000,
                 max_features: int = 100, n_estimators: int = 4,
                 max_eval_rows: int | None = 200_000, log_target: bool = True,
                 **params):
        super().__init__(**params)
        self.n_subsets = n_subsets
        self.subset_size = subset_size
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.max_eval_rows = max_eval_rows
        self.log_target = log_target
        self.models_: list = []
        self.keep_: np.ndarray | None = None

    # ------------------------------------------------------------------ utils
    def _select_features(self, X: np.ndarray, y: np.ndarray, names: list[str],
                         rng) -> np.ndarray:
        n_feat = X.shape[1]
        if n_feat <= self.max_features:
            return np.arange(n_feat)
        take = min(200_000, len(y))
        sel = rng.choice(len(y), take, replace=False)
        Xs, ys = np.asarray(X[sel], dtype=np.float64), y[sel].astype(np.float64)
        # rank-based correlation: robust to the extreme skew documented in EDA 5.2
        yr = np.argsort(np.argsort(ys)).astype(np.float64)
        score = np.zeros(n_feat)
        for j in range(n_feat):
            col = Xs[:, j]
            ok = np.isfinite(col)
            if ok.sum() < 100 or np.nanstd(col[ok]) == 0:
                continue
            xr = np.argsort(np.argsort(col[ok])).astype(np.float64)
            score[j] = abs(np.corrcoef(xr, yr[ok])[0, 1])
        score = np.nan_to_num(score)
        keep = set(np.argsort(-score)[: self.max_features].tolist())
        if "cgm_now" in names:
            keep.add(names.index("cgm_now"))
        return np.array(sorted(keep))

    def _fwd(self, y):
        return np.log(np.clip(y, 1.0, None)) if self.log_target else y

    def _inv(self, p):
        return np.exp(np.clip(p, 0.0, 7.0)) if self.log_target else p

    # -------------------------------------------------------------------- api
    def fit(self, X, y, X_val=None, y_val=None, ctx: FitContext | None = None):
        from tabpfn import TabPFNRegressor

        rng = np.random.default_rng(ctx.seed if ctx else 0)
        names = ctx.feature_names if ctx else [f"f{i}" for i in range(X.shape[1])]
        self.keep_ = self._select_features(X, y, names, rng)
        self.kept_names_ = [names[i] for i in self.keep_]

        device = f"cuda:{ctx.gpu}" if (ctx and ctx.gpu is not None) else "cpu"
        n = len(y)
        size = min(self.subset_size, n)

        self.models_ = []
        for s in range(self.n_subsets):
            idx = rng.choice(n, size, replace=False)
            Xs = np.asarray(X[np.sort(idx)][:, self.keep_], dtype=np.float32)
            ys = self._fwd(y[np.sort(idx)]).astype(np.float32)
            m = TabPFNRegressor(device=device, n_estimators=self.n_estimators,
                                random_state=(ctx.seed if ctx else 0) + s,
                                ignore_pretraining_limits=True)
            m.fit(Xs, ys)
            self.models_.append(m)

        self.notes = {
            "n_subsets": self.n_subsets,
            "subset_size": size,
            "context_rows_total": size * self.n_subsets,
            "train_rows_available": int(n),
            "fraction_of_train_seen": round(size * self.n_subsets / max(n, 1), 4),
            "n_features_kept": int(len(self.keep_)),
            "features_kept": self.kept_names_,
            "log_target": self.log_target,
            "device": device,
            "caveat": ("TabPFN sees only a bagged subsample of the training data; its "
                       "score is not comparable to a model trained on all rows."),
        }
        return self

    def predict(self, X, chunk: int = 10_000) -> np.ndarray:
        """Chunked because TabPFN attends every query row against the whole context:
        peak memory is O(chunk x context x n_estimators). At chunk=50,000 this
        reached 31 GB on a 32 GB V100; 10,000 keeps it comfortable."""
        import gc
        Xs = np.asarray(X[:, self.keep_], dtype=np.float32)
        acc = np.zeros(len(Xs), dtype=np.float64)
        for m in self.models_:
            for i in range(0, len(Xs), chunk):
                acc[i:i + chunk] += m.predict(Xs[i:i + chunk])
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:                                    # noqa: BLE001
                pass
            gc.collect()
        return self._inv(acc / len(self.models_)).astype(np.float32)

    def subsample_eval(self, n_rows: int, seed: int = 0) -> np.ndarray | None:
        """Indices to score on, or None to score everything."""
        if self.max_eval_rows is None or n_rows <= self.max_eval_rows:
            return None
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(n_rows, self.max_eval_rows, replace=False))
