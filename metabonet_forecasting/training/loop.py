"""The single training loop every architecture shares.

It never branches on the model: it calls ``model(batch)``, gets a
``ForecastOutput``, computes a masked loss in model space, and early-stops on
validation RMSE in mg/dL. Anything an architecture needs beyond that has to live
inside the architecture.

What it writes into the run directory:
    best.pt                the best checkpoint by validation RMSE
    train_log.csv          per-epoch losses, metrics, learning rate, timings
    predictions_*.parquet  per-anchor forecasts in mg/dL for later scoring
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..common.logging import RunLogger
from ..data.splits import seen_in_train
from ..data.windows import batch_to_device
from .callbacks import Checkpointer, EarlyStopping, WarmupCosine
from .losses import build_loss, build_loss_with_zone


class Trainer:
    def __init__(self, model: nn.Module, bundle, cfg, run_dir: Path,
                 logger: Optional[RunLogger] = None) -> None:
        self.cfg = cfg
        self.bundle = bundle
        self.run_dir = Path(run_dir)
        self.log = logger or RunLogger(self.run_dir)

        t = cfg.get_path("training")
        self.device = torch.device(str(t.get("device", "cuda" if torch.cuda.is_available()
                                             else "cpu")))
        self.model = model.to(self.device)
        self.epochs = int(t.get("epochs", 20))
        self.max_steps_per_epoch = t.get("max_steps_per_epoch")
        self.grad_clip = float(t.get("grad_clip", 1.0))
        self.accumulate = int(t.get("accumulate_grad_batches", 1))
        self.amp = bool(t.get("amp", True)) and self.device.type == "cuda"
        self.log_every = int(t.get("log_every", 200))

        loss_cfg = dict(t.get("loss", {"name": "huber"}))
        # Zone A is |log(pred/true)| <= 0.2027. The target standardiser divides
        # model-space values by target_scale, so the window shrinks by the same
        # factor -- which is why the half-width is computed here and not in the
        # config. Only a log target mode makes the surrogate a zone-A stand-in.
        self._zone_cfg = loss_cfg
        self.criterion = build_loss(loss_cfg)
        # Frequency-domain augmentation (FrAug-style). Time-domain transforms --
        # jitter, scaling, warping -- are documented to help classification and
        # NOT forecasting, because they break the fine temporal structure the
        # forecast depends on. Masking spectral components does not.
        self.freq_mask_p = float(t.get("freq_mask_p", 0.0) or 0.0)
        self._init_zone_loss()
        self.quantile_loss = str(loss_cfg.get("name", "")).lower() in ("quantile", "hybrid")
        if self.quantile_loss and not cfg.get_path("model.head.quantiles", None):
            raise ValueError(
                "training.loss.name=quantile/hybrid needs a quantile head; set "
                "model.head.quantiles=[0.1, 0.5, 0.9] to match")
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=float(t.get("lr", 1e-3)),
            weight_decay=float(t.get("weight_decay", 1e-4)),
            betas=tuple(t.get("betas", (0.9, 0.95))))
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.amp)
        self.early = EarlyStopping(patience=int(t.get("patience", 5)),
                                   min_delta=float(t.get("min_delta", 0.0)))
        self.ckpt = Checkpointer(self.run_dir)
        self.scheduler: Optional[WarmupCosine] = None
        self.horizons = list(bundle.spec.horizons_min)

    # ------------------------------------------------------------------ epochs
    def train(self) -> Dict[str, object]:
        train_loader = self.bundle.loader("train", self.cfg)
        val_loader = self.bundle.loader("val", self.cfg, shuffle=False)
        steps_per_epoch = (min(len(train_loader), self.max_steps_per_epoch)
                           if self.max_steps_per_epoch else len(train_loader))
        t = self.cfg.get_path("training")
        self.scheduler = WarmupCosine(
            self.optimizer, total_steps=max(steps_per_epoch * self.epochs, 1),
            warmup_steps=int(t.get("warmup_steps", min(500, steps_per_epoch))),
            min_lr_frac=float(t.get("min_lr_frac", 0.05)))

        self.log.info(f"training {type(self.model).__name__} "
                      f"({self.model.n_parameters():,} params) on {self.device} | "
                      f"{len(train_loader.dataset):,} train windows, "
                      f"{len(val_loader.dataset):,} val windows, "
                      f"{steps_per_epoch:,} steps/epoch, amp={self.amp}")

        history: List[Dict[str, float]] = []
        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch(train_loader, steps_per_epoch, epoch)
            val = self.evaluate_loader(val_loader)
            row = {"epoch": epoch, "train_loss": round(train_loss, 6),
                   "val_loss": round(val["loss"], 6),
                   "val_rmse": round(val["rmse"], 4), "val_mae": round(val["mae"], 4),
                   "val_mard": round(val["mard"], 4),
                   "lr": self.optimizer.param_groups[0]["lr"],
                   "seconds": round(time.time() - t0, 1)}
            for h, v in zip(self.horizons, val["rmse_per_horizon"]):
                row[f"val_rmse_{h}"] = round(float(v), 4)
            history.append(row)
            self.log.epoch(row)

            improved = self.early.step(val["rmse"], epoch)
            if improved:
                self.ckpt.save(self.model, {"epoch": epoch, "val_rmse": val["rmse"],
                                            "horizons_min": self.horizons})
            self.log.info(
                f"epoch {epoch:>3}/{self.epochs}  train {train_loss:.4f}  "
                f"val RMSE {val['rmse']:.2f} mg/dL  MARD {val['mard']:.2f}%  "
                f"{'*best*' if improved else f'({self.early.bad_epochs} since best)'}  "
                f"{row['seconds']:.0f}s")
            if self.early.should_stop:
                self.log.info(f"early stop at epoch {epoch}; best was epoch "
                              f"{self.early.best_epoch} at {self.early.best_value:.3f}")
                break

        if self.ckpt.exists():
            self.ckpt.load_into(self.model, map_location=self.device)
        return {"history": history, "best_epoch": self.early.best_epoch,
                "best_val_rmse": float(self.early.best_value)}

    def _init_zone_loss(self) -> None:
        """Rebuild the criterion with the zone-A surrogate, once the
        standardiser is known."""
        cfg = self._zone_cfg
        if float(cfg.get("zone_weight", 0.0) or 0.0) <= 0:
            return
        std = getattr(self.bundle, "standardizer", None)
        mode = str(getattr(std, "target_mode", "delta"))
        if mode not in ("log_ratio", "log"):
            raise ValueError(
                f"training.loss.zone_weight needs a log target mode (got "
                f"data.target_mode={mode!r}): zone A is a relative band, so the "
                "surrogate only stands in for it in log space.")
        import numpy as _np
        scale = _np.maximum(_np.asarray(std.target_scale, dtype=_np.float32), 1e-6)
        cfg = dict(cfg); cfg["zone_halfwidth"] = (0.2027 / scale).tolist()
        self.criterion = build_loss_with_zone(cfg).to(self.device)
        self.log.info("zone-A surrogate on: weight=%s halfwidth(model space)=%s"
                      % (cfg["zone_weight"],
                         [round(float(v), 4) for v in cfg["zone_halfwidth"]]))

    def _freq_augment(self, batch):
        """Randomly drop spectral components of the CGM context (training only).

        rFFT the value channel, zero a random subset of frequency bins, invert.
        The observation mask and all other channels are untouched, and the
        targets are never touched at all.
        """
        x = batch.get("x_cgm")
        if x is None or x.shape[-1] == 0:
            return batch
        v = x[..., 0].float()
        f = torch.fft.rfft(v, dim=1)
        keep = (torch.rand(f.shape, device=f.device) >= self.freq_mask_p)
        keep[:, 0] = True                    # keep the mean; dropping it shifts level
        v2 = torch.fft.irfft(f * keep, n=v.shape[1], dim=1).to(x.dtype)
        batch = dict(batch)
        batch["x_cgm"] = torch.cat([v2.unsqueeze(-1), x[..., 1:]], dim=-1)
        return batch

    def _train_epoch(self, loader, steps: int, epoch: int) -> float:
        self.model.train()
        total, n, n_bad = 0.0, 0, 0
        self.optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader):
            if step >= steps:
                break
            batch = batch_to_device(batch, self.device)
            if self.freq_mask_p > 0.0:
                batch = self._freq_augment(batch)
            with torch.amp.autocast(self.device.type, enabled=self.amp):
                out = self.model(batch)
                target = batch["y_norm"]
                mask = batch["y_mask"]
                pred = out.quantiles if self.quantile_loss else out.pred_norm
                loss = self.criterion(pred, target, mask)
            # A single non-finite batch used to poison the reported loss for the
            # rest of the epoch, because the log prints a RUNNING MEAN: once
            # `total` is NaN it stays NaN however healthy the next 20,000 steps
            # are. Skip the batch, count it, and leave the average truthful.
            # (GradScaler already declines to apply non-finite gradients, so
            # this is about honest reporting and wasted work, not weight safety.)
            if not torch.isfinite(loss):
                n_bad += 1
                if n_bad in (1, 10, 100, 1000):
                    self.log.warning(
                        f"  epoch {epoch} step {step}: non-finite loss "
                        f"({n_bad} so far this epoch) -- batch skipped")
                self.optimizer.zero_grad(set_to_none=True)
                continue

            self.scaler.scale(loss / self.accumulate).backward()

            if (step + 1) % self.accumulate == 0:
                if self.grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler:
                    self.scheduler.step()

            total += float(loss.detach()) * mask.shape[0]
            n += mask.shape[0]
            if self.log_every and step and step % self.log_every == 0:
                bad = f"  [{n_bad} non-finite skipped]" if n_bad else ""
                self.log.info(f"  epoch {epoch} step {step}/{steps} "
                              f"loss {total / max(n, 1):.4f}{bad}")
        if n_bad:
            self.log.warning(
                f"  epoch {epoch}: {n_bad}/{steps} batches had a non-finite loss "
                f"and were skipped. A handful is a bad batch; a rising count is "
                f"divergence -- lower training.lr or raise training.warmup_steps.")
        return total / max(n, 1)

    # -------------------------------------------------------------- inference
    @torch.no_grad()
    def predict(self, loader) -> pd.DataFrame:
        """Per-anchor forecasts in mg/dL, ready for scoring."""
        self.model.eval()
        std = self.bundle.standardizer
        idx: List[np.ndarray] = []
        preds: List[np.ndarray] = []
        trues: List[np.ndarray] = []
        masks: List[np.ndarray] = []
        quants: List[np.ndarray] = []
        for batch in loader:
            batch = batch_to_device(batch, self.device)
            with torch.amp.autocast(self.device.type, enabled=self.amp):
                out = self.model(batch)
            pred_mgdl = std.inverse_targets(out.pred_norm.float(), batch["cgm_now"].float())
            idx.append(batch["index"].cpu().numpy())
            preds.append(pred_mgdl.cpu().numpy())
            trues.append(batch["y"].cpu().numpy())
            masks.append(batch["y_mask"].cpu().numpy())
            if out.quantiles is not None:
                q = torch.stack([
                    std.inverse_targets(out.quantiles[..., k].float(),
                                        batch["cgm_now"].float())
                    for k in range(out.quantiles.shape[-1])], dim=-1)
                quants.append(q.cpu().numpy())

        index = np.concatenate(idx)
        pred = np.concatenate(preds)
        true = np.concatenate(trues)
        mask = np.concatenate(masks)
        true = np.where(mask > 0, true, np.nan)
        pred = np.where(mask > 0, pred, np.nan)

        anchors = self.bundle.store.anchors
        df = pd.DataFrame({"anchor": index})
        for col in ("subject", "cohort", "is_test", "known_patient", "cgm_now",
                    "cgm_delta_30min", "min_since_carb", "min_since_bolus", "tod_min"):
            df[col] = anchors[col].to_numpy()[index]
        df["seen_in_train"] = seen_in_train(self.bundle.store, self.bundle.split, index)
        for j, h in enumerate(self.horizons):
            df[f"y_true_{h}"] = true[:, j]
            df[f"y_pred_{h}"] = pred[:, j]
        if quants:
            levels = list(self.cfg.get_path("model.head.quantiles", []) or [])
            quant = np.concatenate(quants)
            for j, h in enumerate(self.horizons):
                for k, lv in enumerate(levels):
                    df[f"y_q{int(round(lv * 100)):02d}_{h}"] = \
                        np.where(mask[:, j] > 0, quant[:, j, k], np.nan)
        return df

    @torch.no_grad()
    def evaluate_loader(self, loader) -> Dict[str, object]:
        """Validation metrics in mg/dL plus the model-space loss."""
        self.model.eval()
        std = self.bundle.standardizer
        loss_sum, n_items = 0.0, 0
        se = np.zeros(len(self.horizons))
        ae = np.zeros(len(self.horizons))
        rel = np.zeros(len(self.horizons))
        cnt = np.zeros(len(self.horizons))
        for batch in loader:
            batch = batch_to_device(batch, self.device)
            with torch.amp.autocast(self.device.type, enabled=self.amp):
                out = self.model(batch)
                pred = out.quantiles if self.quantile_loss else out.pred_norm
                loss = self.criterion(pred, batch["y_norm"], batch["y_mask"])
            loss_sum += float(loss) * batch["y_mask"].shape[0]
            n_items += batch["y_mask"].shape[0]

            p = std.inverse_targets(out.pred_norm.float(), batch["cgm_now"].float())
            p = p.cpu().numpy()
            y = batch["y"].cpu().numpy()
            m = batch["y_mask"].cpu().numpy() > 0
            err = np.where(m, p - y, 0.0)
            se += (err ** 2).sum(axis=0)
            ae += np.abs(err).sum(axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                rel += np.where(m & (y > 0), np.abs(err) / np.where(y > 0, y, 1), 0.0).sum(axis=0)
            cnt += m.sum(axis=0)

        cnt_safe = np.maximum(cnt, 1)
        rmse_h = np.sqrt(se / cnt_safe)
        return {
            "loss": loss_sum / max(n_items, 1),
            "rmse": float(np.sqrt(se.sum() / max(cnt.sum(), 1))),
            "mae": float(ae.sum() / max(cnt.sum(), 1)),
            "mard": float(rel.sum() / max(cnt.sum(), 1) * 100.0),
            "rmse_per_horizon": rmse_h.tolist(),
            "n_per_horizon": cnt.tolist(),
        }
