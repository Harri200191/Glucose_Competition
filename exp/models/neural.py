"""Neural networks.

Five architectures across two input regimes:

**Tabular** (same feature matrix the trees get)
  * ``mlp``      -- 3-layer MLP, GELU, dropout, categorical embeddings
  * ``resmlp``   -- pre-norm residual blocks, deeper, for the >1 M-sample runs
  * ``ft_transformer`` -- feature-tokeniser + transformer encoder (Gorishniy et al.),
    the strongest published NN family on tabular data

**Sequence** (raw 24 h window gathered on the fly from `channels.npy`)
  * ``tcn``      -- dilated causal 1-D convolutions; receptive field covers 24 h
  * ``gru``      -- bidirectional-free (causal) GRU encoder

Why sequence models at all: EDA 10.2 showed the carbohydrate response peaks 45-75 min
after the event and decays over 3-4 h, and EDA 9.6 showed autocorrelation is still
~0.49 at two hours. The hand-built window features compress that shape into summary
statistics; a sequence model gets to learn the shape itself.

Shared decisions
----------------
* **Target: a residual on top of persistence.** The network predicts
  ``log(glucose_{t+h}) - log(glucose_t)`` -- the log-ratio -- and the prediction is
  ``exp(log(cgm_now) + delta)``. This matters more than any architecture choice
  here. EDA 9.6 measured persistence at MARD 14.03 % with no features at all, so a
  network that regresses the absolute level spends its first epochs rediscovering
  "output roughly the current value". Measured on the 40-subject smoke set, the
  absolute parameterisation started at val MARD 153 % and never settled below
  14.4 %; the residual form starts at approximately persistence by construction and
  improves monotonically from there.
* **Loss.** L1 on the log scale. The gradient of |log p - log t| carries a 1/t
  weighting, which is exactly what MARD applies (EDA 9.2), so the training signal
  matches the scoring metric.
* **NaN.** Networks cannot consume NaN. Inputs get median imputation +
  missingness indicators (fitted on train only) and robust median/IQR scaling --
  EDA 5.2 showed skew in the tens to hundreds, where mean/sd scaling is dominated
  by the tail.
"""
from __future__ import annotations

import math
import time
import numpy as np

from .base import BaseModel, FitContext, NanImputer, Standardiser


# ============================================================== torch modules
def build_modules():
    import torch
    import torch.nn as nn

    class MLP(nn.Module):
        def __init__(self, d_in, hidden=(512, 256, 128), dropout=0.15):
            super().__init__()
            layers, d = [], d_in
            for h in hidden:
                layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.GELU(),
                           nn.Dropout(dropout)]
                d = h
            layers += [nn.Linear(d, 1)]
            self.net = nn.Sequential(*layers)

        def forward(self, x, seq=None):
            return self.net(x).squeeze(-1)

    class ResBlock(nn.Module):
        def __init__(self, d, expansion=2, dropout=0.15):
            super().__init__()
            self.norm = nn.LayerNorm(d)
            self.fc1 = nn.Linear(d, d * expansion)
            self.fc2 = nn.Linear(d * expansion, d)
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            h = self.norm(x)
            h = self.fc2(self.drop(torch.nn.functional.gelu(self.fc1(h))))
            return x + self.drop(h)

    class ResMLP(nn.Module):
        def __init__(self, d_in, d_model=256, n_blocks=6, dropout=0.15):
            super().__init__()
            self.inp = nn.Linear(d_in, d_model)
            self.blocks = nn.ModuleList([ResBlock(d_model, 2, dropout)
                                         for _ in range(n_blocks)])
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, 1)

        def forward(self, x, seq=None):
            h = self.inp(x)
            for b in self.blocks:
                h = b(h)
            return self.head(self.norm(h)).squeeze(-1)

    class FTTransformer(nn.Module):
        """Feature tokeniser + transformer encoder. Each scalar feature becomes a
        token via its own linear map, plus a CLS token that carries the readout."""

        def __init__(self, d_in, d_token=64, n_layers=4, n_heads=8, dropout=0.1):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(d_in, d_token))
            self.bias = nn.Parameter(torch.empty(d_in, d_token))
            nn.init.normal_(self.weight, std=d_token ** -0.5)
            nn.init.normal_(self.bias, std=d_token ** -0.5)
            self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
            layer = nn.TransformerEncoderLayer(
                d_token, n_heads, dim_feedforward=d_token * 2, dropout=dropout,
                activation="gelu", batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, n_layers)
            self.norm = nn.LayerNorm(d_token)
            self.head = nn.Linear(d_token, 1)

        def forward(self, x, seq=None):
            t = x.unsqueeze(-1) * self.weight + self.bias        # B, F, D
            t = torch.cat([self.cls.expand(t.size(0), -1, -1), t], dim=1)
            h = self.enc(t)
            return self.head(self.norm(h[:, 0])).squeeze(-1)

    class TCN(nn.Module):
        """Dilated causal convolutions over the channel window, concatenated with a
        static/tabular branch."""

        def __init__(self, d_static, n_channels=5, width=96, levels=6,
                     kernel=3, dropout=0.1):
            super().__init__()
            convs = []
            c_in = n_channels
            for i in range(levels):
                d = 2 ** i
                convs.append(nn.Sequential(
                    nn.Conv1d(c_in, width, kernel, dilation=d,
                              padding=(kernel - 1) * d),
                    nn.BatchNorm1d(width), nn.GELU(), nn.Dropout(dropout)))
                c_in = width
            self.convs = nn.ModuleList(convs)
            self.kernel, self.levels = kernel, levels
            self.static = nn.Sequential(nn.Linear(d_static, 128), nn.GELU(),
                                        nn.Dropout(dropout))
            self.head = nn.Sequential(nn.Linear(width * 2 + 128, 128), nn.GELU(),
                                      nn.Dropout(dropout), nn.Linear(128, 1))

        def forward(self, x, seq):
            h = seq                                             # B, C, L
            for i, conv in enumerate(self.convs):
                trim = (self.kernel - 1) * (2 ** i)
                h = conv(h)[:, :, :-trim] if trim else conv(h)   # causal
            pooled = torch.cat([h[:, :, -1], h.mean(dim=2)], dim=1)
            return self.head(torch.cat([pooled, self.static(x)], dim=1)).squeeze(-1)

    class GRUNet(nn.Module):
        def __init__(self, d_static, n_channels=5, hidden=128, layers=2, dropout=0.1):
            super().__init__()
            self.rnn = nn.GRU(n_channels, hidden, layers, batch_first=True,
                              dropout=dropout if layers > 1 else 0.0)
            self.static = nn.Sequential(nn.Linear(d_static, 128), nn.GELU(),
                                        nn.Dropout(dropout))
            self.head = nn.Sequential(nn.Linear(hidden + 128, 128), nn.GELU(),
                                      nn.Dropout(dropout), nn.Linear(128, 1))

        def forward(self, x, seq):
            out, _ = self.rnn(seq.transpose(1, 2))               # B, L, C
            return self.head(torch.cat([out[:, -1], self.static(x)], dim=1)).squeeze(-1)

    return dict(mlp=MLP, resmlp=ResMLP, ft_transformer=FTTransformer,
                tcn=TCN, gru=GRUNet)


# ============================================================ sequence gather
class SequenceSource:
    """Gathers a 24 h channel window per anchor, lazily, from a memmap.

    Materialising the windows would cost 288 steps x 5 channels x 4 bytes x 10 M
    samples = 57 GB. Indexing into the memmapped full-resolution channel array
    instead costs nothing until the batch is assembled.
    """

    CHANNELS = ("cgm", "cgm_present", "bolus", "basal", "carbs")

    def __init__(self, channels: np.ndarray, seq_len: int = 96, seq_stride: int = 3):
        self.channels = channels            # [n_rows, 5] float32, memmapped
        self.seq_len = seq_len
        self.seq_stride = seq_stride
        # offsets are <= 0: the window ends at the anchor row
        self.offsets = -(np.arange(seq_len)[::-1] * seq_stride)

    @property
    def span_steps(self) -> int:
        return (self.seq_len - 1) * self.seq_stride

    def gather(self, anchor_rows: np.ndarray) -> np.ndarray:
        """`anchor_rows` are absolute row indices into `channels`, not sample ids."""
        rows = np.asarray(anchor_rows)[:, None] + self.offsets[None, :]
        np.clip(rows, 0, len(self.channels) - 1, out=rows)
        win = np.asarray(self.channels[rows], dtype=np.float32)   # B, L, C
        return np.ascontiguousarray(win.transpose(0, 2, 1))       # B, C, L


# ================================================================= the wrapper
class NeuralModel(BaseModel):
    name = "nn"
    handles_nan = False
    wants_scaling = True
    supports_early_stopping = True

    SEQUENCE_ARCHS = {"tcn", "gru"}

    def __init__(self, arch: str = "mlp", epochs: int = 30, batch_size: int = 4096,
                 lr: float = 2e-3, weight_decay: float = 1e-5, dropout: float = 0.15,
                 patience: int = 8, loss: str = "l1_log", seq_len: int = 96,
                 seq_stride: int = 3, amp: bool = True, num_workers: int = 0,
                 residual_on_persistence: bool = True, **arch_kw):
        super().__init__()
        self.arch = arch
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.patience = patience
        self.loss_name = loss
        self.seq_len = seq_len
        self.seq_stride = seq_stride
        self.amp = amp
        self.num_workers = num_workers
        self.residual_on_persistence = residual_on_persistence
        self.base_col_: int | None = None
        self.arch_kw = arch_kw
        self.seq_source: SequenceSource | None = None
        self.seq_rows_train: np.ndarray | None = None
        self.seq_rows_val: np.ndarray | None = None
        self.imputer = NanImputer(add_indicators=True)
        self.scaler = Standardiser()

    # ------------------------------------------------------------------ data
    def attach_sequences(self, channels: np.ndarray, rows_train: np.ndarray,
                         rows_val: np.ndarray | None = None) -> None:
        """Required for arch in SEQUENCE_ARCHS.

        `channels` is the full-resolution [n_rows, C] array; `rows_train` /
        `rows_val` are the absolute anchor row indices for the X matrices that will
        be handed to `fit`, in the same order. `predict` takes its own `rows`.
        """
        self.seq_source = SequenceSource(channels, self.seq_len, self.seq_stride)
        self.seq_rows_train = np.asarray(rows_train)
        self.seq_rows_val = None if rows_val is None else np.asarray(rows_val)

    def _prep_x(self, X, fit: bool):
        Xn = np.asarray(X, dtype=np.float32)
        if fit:
            Xn = self.imputer.fit_transform(Xn)
            self.scaler.fit(Xn)
        else:
            Xn = self.imputer.transform(Xn)
        return self.scaler.transform(Xn)

    def _baseline(self, X) -> np.ndarray:
        """log of the current glucose -- the persistence anchor the net corrects.

        Read from the RAW feature matrix, before imputation and scaling. `cgm_now`
        is never NaN at an anchor (build_dataset requires a real reading there), but
        the clip is kept as a guard against a caller passing a looser matrix.
        """
        if not self.residual_on_persistence or self.base_col_ is None:
            return np.zeros(len(X), dtype=np.float32)
        v = np.asarray(X[:, self.base_col_], dtype=np.float64)
        v = np.where(np.isfinite(v) & (v > 1.0), v, 100.0)
        return np.log(v).astype(np.float32)

    def _loss_fn(self, torch):
        if self.loss_name == "l1_log":
            return lambda p, t: (p - t).abs().mean()          # t already log-space
        if self.loss_name == "huber_log":
            return torch.nn.SmoothL1Loss(beta=0.05)
        if self.loss_name == "l2_log":
            return torch.nn.MSELoss()
        raise ValueError(f"unknown loss {self.loss_name!r}")

    # ------------------------------------------------------------------- fit
    def fit(self, X, y, X_val=None, y_val=None, ctx: FitContext | None = None):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        seed = ctx.seed if ctx else 0
        torch.manual_seed(seed)
        np.random.seed(seed)
        dev = torch.device(f"cuda:{ctx.gpu}" if (ctx and ctx.gpu is not None
                                                 and torch.cuda.is_available()) else "cpu")
        is_seq = self.arch in self.SEQUENCE_ARCHS
        if is_seq and self.seq_source is None:
            raise RuntimeError(f"arch={self.arch!r} needs attach_sequences() first")

        if self.residual_on_persistence and ctx and "cgm_now" in ctx.feature_names:
            self.base_col_ = ctx.feature_names.index("cgm_now")

        base_tr = self._baseline(X)
        Xtr = self._prep_x(X, fit=True)
        # the network learns the log-RATIO; adding base_tr back recovers the level
        ytr = np.log(np.clip(np.asarray(y, dtype=np.float32), 1.0, None)) - base_tr
        n_tr = len(Xtr)
        has_val = X_val is not None and len(X_val) > 0
        if has_val:
            base_va = self._baseline(X_val)
            Xva = self._prep_x(X_val, fit=False)
            yva = np.log(np.clip(np.asarray(y_val, dtype=np.float32), 1.0, None)) - base_va

        mods = build_modules()
        d_in = Xtr.shape[1]
        self.d_in_ = int(d_in)
        self.n_channels_ = int(self.seq_source.channels.shape[1]) if is_seq else 0
        if is_seq:
            net = mods[self.arch](d_static=d_in, n_channels=self.n_channels_,
                                  dropout=self.dropout, **self.arch_kw)
        else:
            net = mods[self.arch](d_in, dropout=self.dropout, **self.arch_kw)
        net = net.to(dev)

        opt = torch.optim.AdamW(net.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        steps = max(1, math.ceil(n_tr / self.batch_size)) * self.epochs
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr,
                                                    total_steps=steps, pct_start=0.15)
        lossf = self._loss_fn(torch)
        scaler = torch.amp.GradScaler("cuda", enabled=(self.amp and dev.type == "cuda"))

        # sequence models index rows lazily; tabular models can sit on the GPU
        tr_order = np.arange(n_tr)
        best, best_state, bad, history = np.inf, None, 0, []
        t0 = time.time()

        for epoch in range(self.epochs):
            net.train()
            np.random.shuffle(tr_order)
            tot, seen = 0.0, 0
            for i in range(0, n_tr, self.batch_size):
                sel = tr_order[i:i + self.batch_size]
                xb = torch.from_numpy(Xtr[sel]).to(dev, non_blocking=True)
                tb = torch.from_numpy(ytr[sel]).to(dev, non_blocking=True)
                sb = None
                if is_seq:
                    sb = torch.from_numpy(np.nan_to_num(
                        self.seq_source.gather(self.seq_rows_train[sel]))).to(dev)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=(self.amp and dev.type == "cuda")):
                    pred = net(xb, sb)
                    loss = lossf(pred, tb)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                tot += float(loss) * len(sel); seen += len(sel)

            row = {"epoch": epoch, "train_loss": tot / max(seen, 1)}
            if has_val:
                vp = self._raw_predict(net, Xva, dev, is_seq, rows=self.seq_rows_val)
                # report in the metric that matters, not in log-loss units
                p_abs = np.exp(np.clip(vp + base_va, 0.0, 7.0))
                t_abs = np.exp(yva + base_va)
                row["val_MARD"] = float(np.mean(np.abs(p_abs - t_abs) / t_abs) * 100)
                score = row["val_MARD"]
                if score < best - 1e-4:
                    best, bad = score, 0
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in net.state_dict().items()}
                else:
                    bad += 1
            history.append(row)
            if has_val and bad >= self.patience:
                break

        if best_state is not None:
            net.load_state_dict(best_state)
        self.net_ = net
        self.dev_ = dev
        self.is_seq_ = is_seq
        self.notes = {
            "arch": self.arch, "loss": self.loss_name,
            "target": ("log(glucose_t+h) - log(glucose_t)  [residual on persistence]"
                       if self.residual_on_persistence else "log(glucose_t+h)"),
            "epochs_run": len(history), "best_val_MARD": (None if not has_val else best),
            "device": str(dev), "params": int(sum(p.numel() for p in net.parameters())),
            "seq_len": self.seq_len if is_seq else None,
            "seq_stride": self.seq_stride if is_seq else None,
            "seq_hours": (self.seq_len * self.seq_stride * 5 / 60) if is_seq else None,
            "imputation": self.imputer.describe(ctx.feature_names if ctx else []),
            "scaling": "median/IQR, clipped to +-8",
            "train_seconds": round(time.time() - t0, 1),
            "history": history,
        }
        return self

    # --------------------------------------------------------------- predict
    def _raw_predict(self, net, Xn, dev, is_seq, rows=None):
        import torch
        if is_seq and rows is None:
            raise RuntimeError("sequence architectures need the anchor `rows` for the "
                               "matrix being predicted")
        net.eval()
        out = np.empty(len(Xn), dtype=np.float32)
        bs = self.batch_size * 4
        with torch.no_grad():
            for i in range(0, len(Xn), bs):
                sel = np.arange(i, min(i + bs, len(Xn)))
                xb = torch.from_numpy(Xn[sel]).to(dev)
                sb = None
                if is_seq:
                    sb = torch.from_numpy(np.nan_to_num(
                        self.seq_source.gather(np.asarray(rows)[sel]))).to(dev)
                with torch.amp.autocast("cuda", enabled=(self.amp and dev.type == "cuda")):
                    out[sel] = net(xb, sb).float().cpu().numpy()
        return out

    def predict(self, X, rows: np.ndarray | None = None) -> np.ndarray:
        """`rows` = absolute anchor row indices, required for sequence archs."""
        base = self._baseline(X)
        Xn = self._prep_x(X, fit=False)
        raw = self._raw_predict(self.net_, Xn, self.dev_, self.is_seq_, rows=rows)
        return np.exp(np.clip(raw + base, 0.0, 7.0)).astype(np.float32)

    # ------------------------------------------------------------- persistence
    # The architecture classes are defined inside `build_modules()`, so they are
    # not picklable by reference and plain joblib.dump fails. Save the weights plus
    # everything needed to rebuild instead -- which is the durable format anyway.
    def save(self, path) -> None:
        import torch
        torch.save({
            "format": "NeuralModel/1",
            "arch": self.arch, "arch_kw": self.arch_kw,
            "hyper": dict(epochs=self.epochs, batch_size=self.batch_size, lr=self.lr,
                          weight_decay=self.weight_decay, dropout=self.dropout,
                          patience=self.patience, loss=self.loss_name,
                          seq_len=self.seq_len, seq_stride=self.seq_stride,
                          amp=self.amp,
                          residual_on_persistence=self.residual_on_persistence),
            "d_in": self.d_in_, "n_channels": self.n_channels_,
            "is_seq": self.is_seq_, "base_col": self.base_col_,
            "state_dict": {k: v.detach().cpu() for k, v in self.net_.state_dict().items()},
            "imputer": {"medians": self.imputer.medians_,
                        "indicator_cols": self.imputer.indicator_cols_,
                        "add_indicators": self.imputer.add_indicators},
            "scaler": {"center": self.scaler.center_, "scale": self.scaler.scale_,
                       "clip": self.scaler.clip},
            "notes": self.notes,
        }, str(path))

    @classmethod
    def load_checkpoint(cls, path, device: str | None = None) -> "NeuralModel":
        import torch
        ck = torch.load(str(path), map_location="cpu", weights_only=False)
        if ck.get("format") != "NeuralModel/1":
            raise ValueError(f"{path} is not a NeuralModel checkpoint")

        m = cls(arch=ck["arch"], **ck["hyper"], **ck["arch_kw"])
        m.d_in_, m.n_channels_ = ck["d_in"], ck["n_channels"]
        m.is_seq_ = ck["is_seq"]
        m.base_col_ = ck["base_col"]
        m.notes = ck.get("notes", {})

        m.imputer = NanImputer(add_indicators=ck["imputer"]["add_indicators"])
        m.imputer.medians_ = ck["imputer"]["medians"]
        m.imputer.indicator_cols_ = ck["imputer"]["indicator_cols"]
        m.scaler = Standardiser(clip=ck["scaler"]["clip"])
        m.scaler.center_, m.scaler.scale_ = ck["scaler"]["center"], ck["scaler"]["scale"]

        mods = build_modules()
        if m.is_seq_:
            net = mods[m.arch](d_static=m.d_in_, n_channels=m.n_channels_,
                               dropout=m.dropout, **m.arch_kw)
        else:
            net = mods[m.arch](m.d_in_, dropout=m.dropout, **m.arch_kw)
        net.load_state_dict(ck["state_dict"])
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        m.net_ = net.to(dev).eval()
        m.dev_ = dev
        return m
