"""The single training loop every architecture shares.

It never branches on the model: it calls ``model(batch)``, gets a
``ForecastOutput``, computes a masked loss in model space, and early-stops on a
validation metric. Anything an architecture needs beyond that has to live inside
the architecture. What the criterion is handed comes from the criterion's own
``input_kind`` (``losses.loss_inputs``), never from which head produced it.

``training.select_metric`` chooses what ``best.pt`` is selected on. The default is
``val_rmse``, which is what every historical run used -- but it is the wrong
criterion for an arm whose loss deliberately trades RMSE for zone A, and until
this existed every such experiment was confounded by it. Validation zone A is
computed and logged for EVERY run either way, so the diagnostic is always there.

What it writes into the run directory:
    best.pt                the best checkpoint by training.select_metric
    train_log.csv          per-epoch losses, metrics (incl. val zone A), lr, timings
    predictions_*.parquet  per-anchor forecasts in mg/dL for later scoring
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..common.logging import RunLogger
from ..data.splits import seen_in_train
from ..data.windows import batch_to_device
from .callbacks import Checkpointer, EarlyStopping, WarmupCosine
from .losses import band_constants, build_loss_with_zone, loss_inputs, needs_band


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

        # One construction, one derivation of the model-space band. _band_cfg
        # needs the fitted standardiser, which is why it cannot live in a config.
        self.criterion = build_loss_with_zone(
            self._band_cfg(dict(t.get("loss", {"name": "huber"})))).to(self.device)
        # Frequency-domain augmentation (FrAug-style). Time-domain transforms --
        # jitter, scaling, warping -- are documented to help classification and
        # NOT forecasting, because they break the fine temporal structure the
        # forecast depends on. Masking spectral components does not.
        # MEASURED NEGATIVE: -0.08 at p=0.10 and -0.43 at p=0.25 (results/physio).
        self.freq_mask_p = float(t.get("freq_mask_p", 0.0) or 0.0)
        if getattr(self.criterion, "input_kind", "point") == "quantiles":
            head_q = list(cfg.get_path("model.head.quantiles", None) or [])
            loss_q = [float(v) for v in
                      (t.get("loss", {}).get("quantiles") or [0.1, 0.5, 0.9])]
            if not head_q:
                raise ValueError(
                    "training.loss.name=quantile/hybrid needs a quantile head; set "
                    "model.head.quantiles=[0.1, 0.5, 0.9] to match")
            # Previously only PRESENCE was checked, so a mismatch produced a
            # wrong-length pinball against the head's columns: either a shape error
            # or, at equal lengths, silently the wrong levels.
            if [float(v) for v in head_q] != loss_q:
                raise ValueError(
                    f"model.head.quantiles={head_q} and training.loss.quantiles="
                    f"{loss_q} must be the same levels in the same order -- the "
                    "pinball term is applied column-by-column to the head's ladder.")
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=float(t.get("lr", 1e-3)),
            weight_decay=float(t.get("weight_decay", 1e-4)),
            betas=tuple(t.get("betas", (0.9, 0.95))))
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.amp)
        # What best.pt is selected on. min_delta CHANGES UNITS with this: mg/dL
        # of RMSE, or percentage points of zone A. 0.02 is a sensible stinginess
        # in the first and absurd in the second (3% of the 0.6 noise floor).
        self.select_metric = str(t.get("select_metric", "val_rmse"))
        if self.select_metric not in ("val_rmse", "val_zone_a"):
            raise ValueError("training.select_metric must be val_rmse|val_zone_a, "
                             f"got {self.select_metric!r}")
        self.early = EarlyStopping(
            patience=int(t.get("patience", 5)),
            min_delta=float(t.get("min_delta", 0.0)),
            mode="min" if self.select_metric == "val_rmse" else "max")
        self._init_ema(dict(t.get("ema", {}) or {}))
        # Resume from last.pt when one is present. Default ON: the failure it
        # guards against is a multi-day run being killed, and restarting from
        # scratch is never what anyone wants at that point. It is a no-op on a
        # fresh run dir.
        self.resume = bool(t.get("resume", True))
        self.ckpt = Checkpointer(self.run_dir)
        # Resume points INSIDE an epoch. A full-data ctx1152 epoch is ~1 day on a
        # V100, so an epoch-end-only last.pt loses up to a day to a crash or a
        # preemption. At each fraction, last.pt is rewritten (exact resume: the
        # train order is a seeded permutation, see EpochPermutationSampler) and
        # the raw weights are kept as ckpt/epEE_qK.pt.
        fr = [float(f) for f in (t.get("checkpoint_fractions") or [])]
        if any(not 0.0 < f < 1.0 for f in fr):
            raise ValueError(f"training.checkpoint_fractions must lie in (0, 1), got {fr}")
        self.ckpt_fractions = sorted(set(fr))
        # Stop LOUDLY once this many batches in one epoch had a non-finite loss.
        # Skipping them is not neutral: in the Mamba runs they were the
        # hypoglycaemia-heavy windows (ReplaceBG, HUPA-UCM, IOBP2), 7-23% of
        # batches, so the model silently stopped learning exactly those patients.
        # null = old behaviour (skip and count forever).
        mnb = t.get("max_nonfinite_batches")
        self.max_nonfinite = None if mnb is None else int(mnb)
        self._resume_step, self._resume_partial = 0, None
        self._history, self._best_val = [], {}
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

        start_epoch, history, best_val = self._try_resume()
        self._history, self._best_val = history, best_val
        if start_epoch > self.epochs:
            self.log.info(f"last.pt is already at epoch {start_epoch - 1} of "
                          f"{self.epochs}; nothing to train")
        for epoch in range(start_epoch, self.epochs + 1):
            t0 = time.time()
            start_step = self._resume_step if epoch == start_epoch else 0
            train_loss = self._train_epoch(train_loader, steps_per_epoch, epoch, start_step)
            val = self.evaluate_loader(val_loader)
            weights, winner = self.model, "raw"
            if self.ema is not None and epoch >= self.ema_start:
                if self.ema_eval_both:
                    # Score both and keep the better. Costs one extra forward
                    # pass over val (~6% wall) and cannot lose to the raw weights.
                    ema_val = self.evaluate_loader(val_loader, self.ema.module)
                    if self._score(ema_val) > self._score(val):
                        val, weights, winner = ema_val, self.ema.module, "ema"
                else:
                    # Trust the average. eval_both=false must NOT mean "build an
                    # EMA and then ignore it", which is a silent no-op.
                    val = self.evaluate_loader(val_loader, self.ema.module)
                    weights, winner = self.ema.module, "ema"
            row = {"epoch": epoch, "train_loss": round(train_loss, 6),
                   "val_nonfinite": int(val.get("n_nonfinite", 0)),
                   "val_loss": round(val["loss"], 6),
                   "val_rmse": round(val["rmse"], 4), "val_mae": round(val["mae"], 4),
                   "val_mard": round(val["mard"], 4),
                   "val_zone_a": round(val["zone_a"], 4),
                   "lr": self.optimizer.param_groups[0]["lr"],
                   "seconds": round(time.time() - t0, 1)}
            for h, v in zip(self.horizons, val["rmse_per_horizon"]):
                row[f"val_rmse_{h}"] = round(float(v), 4)
            for h, v in zip(self.horizons, val["zone_a_per_horizon"]):
                row[f"val_zone_a_{h}"] = round(float(v), 4)
            if self.ema is not None:
                row["weights"] = winner
            history.append(row)
            self.log.epoch(row)

            selected = val["rmse"] if self.select_metric == "val_rmse" else val["zone_a"]
            if val.get("n_nonfinite", 0):
                # val_zone_a counts a NaN prediction as a miss and stays finite, so
                # without this a checkpoint that emits NaN can still be "best"
                # (c96t_mamba2 marked epochs 6, 8 and 9 best while doing exactly that).
                self.log.warning(f"epoch {epoch}: {val['n_nonfinite']} non-finite "
                                 "val predictions -- this epoch cannot be selected")
                selected = float("nan")          # EarlyStopping: NaN never improves
            improved = self.early.step(selected, epoch)
            if improved:
                best_val = dict(val)
                self._best_val = best_val
                self.ckpt.save(weights, {"epoch": epoch, "val_rmse": val["rmse"],
                                         "val_zone_a": val["zone_a"],
                                         "select_metric": self.select_metric,
                                         "weights": winner,
                                         "horizons_min": self.horizons})
            self.log.info(
                f"epoch {epoch:>3}/{self.epochs}  train {train_loss:.4f}  "
                f"val RMSE {val['rmse']:.2f} mg/dL  zone A {val['zone_a']:.2f}%  "
                f"MARD {val['mard']:.2f}%  "
                f"{'*best*' if improved else f'({self.early.bad_epochs} since best)'}"
                f"{'' if self.ema is None else '  [' + winner + ']'}  "
                f"{row['seconds']:.0f}s")
            # Written AFTER the checkpoint decision, so last.pt and best.pt can
            # never disagree about which epoch won.
            self._save_resume(epoch, history, best_val)
            if self.ckpt_fractions:
                self._snapshot(epoch, "end", {"val_zone_a": val["zone_a"],
                                              "val_rmse": val["rmse"]})
            if self.early.should_stop:
                self.log.info(f"early stop at epoch {epoch}; best was epoch "
                              f"{self.early.best_epoch} at {self.early.best_value:.3f}")
                break

        if self.ckpt.exists():
            self.ckpt.load_into(self.model, map_location=self.device)
        # best_val_rmse stays an RMSE whatever selected the checkpoint --
        # evaluation/aggregate.py reads it straight into all_runs.csv.
        return {"history": history, "best_epoch": self.early.best_epoch,
                "best_val_rmse": float(best_val.get("rmse", float("nan"))),
                "best_val_zone_a": float(best_val.get("zone_a", float("nan"))),
                "select_metric": self.select_metric,
                "best_val_score": float(self.early.best_value)}

    def _fingerprint(self) -> str:
        """What must match for a resume to be a continuation and not a new run."""
        import hashlib
        keys = ["data.store", "data.context", "data.target_mode", "data.horizons_min",
                "data.modalities", "data.future_modalities", "data.use_source_identity",
                "data.split.scheme", "data.split.val_frac", "data.split.seed",
                "data.sampling.max_train_windows", "data.sampling.mix",
                "data.internal_holdout.enabled", "data.internal_holdout.exclude_whole_source",
                "model", "training.loss", "training.lr",
                "training.select_metric", "seed"]
        # training.epochs is deliberately NOT here: extending a run is a
        # legitimate reason to resume. It does change the cosine schedule's
        # horizon though, so _try_resume warns when it moved.
        blob = "|".join(f"{k}={self.cfg.get_path(k, None)!r}" for k in keys)
        return hashlib.sha1(blob.encode()).hexdigest()[:16]

    def _save_resume(self, epoch: int, history, best_val, step_in_epoch: int = 0,
                     partial: Optional[Dict] = None) -> None:
        """``epoch`` = last COMPLETED epoch; ``step_in_epoch`` > 0 = mid-way
        through the next one, with ``partial`` the running loss totals."""
        self.ckpt.save_resume({
            "epoch": epoch,
            "step_in_epoch": int(step_in_epoch),
            "partial": partial,
            "batch_size": int(self.cfg.get_path("data.loader.batch_size", 0)),
            # Dropout draws from these. Without them a resumed run is a valid
            # continuation but not the SAME one (measured on a 600-step mamba2
            # smoke: resumed epoch-2 val zone A 67.20 vs 67.32 uninterrupted).
            "rng": {"torch": torch.get_rng_state(),
                    "cuda": (torch.cuda.get_rng_state_all()
                             if torch.cuda.is_available() else None),
                    "numpy": np.random.get_state(),
                    "python": random.getstate()},
            "fingerprint": self._fingerprint(),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "scaler": self.scaler.state_dict(),
            "early": self.early.state_dict(),
            "ema": self.ema.state_dict() if self.ema is not None else None,
            "history": history,
            "best_val": best_val,
            "epochs_planned": self.epochs,
        })

    def _try_resume(self):
        """-> (start_epoch, history, best_val). (1, [], {}) means start fresh."""
        if not self.resume:
            return 1, [], {}
        blob = self.ckpt.load_resume(map_location=self.device)
        if blob is None:
            return 1, [], {}
        fp = self._fingerprint()
        if blob.get("fingerprint") != fp:
            # Refuse rather than silently continue a DIFFERENT experiment from
            # another run's weights -- that would be unattributable.
            raise ValueError(
                f"{self.ckpt.last_path} was written by a different configuration "
                f"(fingerprint {blob.get('fingerprint')} != {fp}). Delete it to "
                "start fresh, or point --runs-dir/--name at the original run.")
        self.model.load_state_dict(blob["model"])
        self.optimizer.load_state_dict(blob["optimizer"])
        if self.scheduler is not None and blob.get("scheduler"):
            self.scheduler.load_state_dict(blob["scheduler"])
        self.scaler.load_state_dict(blob["scaler"])
        self.early.load_state_dict(blob["early"])
        if self.ema is not None and blob.get("ema"):
            self.ema.load_state_dict(blob["ema"])
        planned = blob.get("epochs_planned")
        if planned is not None and int(planned) != self.epochs:
            # Not an error -- but the cosine decays over total_steps, so a resumed
            # run with a different horizon follows a different LR curve from here
            # than one uninterrupted run of either length would have.
            self.log.warning(
                f"training.epochs changed {planned} -> {self.epochs} since last.pt. "
                "The LR schedule is stepped from the restored position but decays "
                "over the NEW horizon, so this is not identical to an uninterrupted "
                f"{self.epochs}-epoch run.")
        ep = int(blob["epoch"])
        rng = blob.get("rng")
        if rng:
            torch.set_rng_state(rng["torch"].cpu())
            if rng.get("cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([t.cpu() for t in rng["cuda"]])
            np.random.set_state(rng["numpy"])
            random.setstate(rng["python"])
        kept = self.log.adopt_existing_csv() if hasattr(self.log, "adopt_existing_csv") else 0
        if kept:
            self.log.info(f"train_log.csv: kept {kept} earlier row(s), appending")
        self._resume_step = int(blob.get("step_in_epoch", 0) or 0)
        self._resume_partial = blob.get("partial")
        if self._resume_step:
            bs_then = int(blob.get("batch_size", 0) or 0)
            bs_now = int(self.cfg.get_path("data.loader.batch_size", 0))
            if bs_then and bs_then != bs_now:
                raise ValueError(
                    f"mid-epoch resume needs the same batch size (last.pt {bs_then}, "
                    f"now {bs_now}): the offset into the epoch is counted in batches")
            self.log.info(f"RESUMING MID-EPOCH: epoch {ep + 1} from step "
                          f"{self._resume_step:,} (same seeded order, no window repeated)")
        self.log.info(
            f"RESUMED from {self.ckpt.last_path.name} at epoch {ep} "
            f"(best {self.select_metric} {self.early.best_value:.4f} at epoch "
            f"{self.early.best_epoch}, {self.early.bad_epochs} since; "
            f"lr schedule at step {getattr(self.scheduler, 'step_num', 0):,})")
        return ep + 1, list(blob.get("history") or []), dict(blob.get("best_val") or {})

    def _mid_checkpoint(self, epoch: int, step_done: int, k: int, steps: int,
                        total: float, n: int, n_bad: int) -> None:
        """Resume point + weight snapshot at fraction k of the epoch."""
        self._save_resume(epoch - 1, self._history, self._best_val,
                          step_in_epoch=step_done,
                          partial={"total": total, "n": n, "n_bad": n_bad})
        self._snapshot(epoch, f"q{k}", {"step_in_epoch": step_done, "steps": steps,
                                        "train_loss_so_far": total / max(n, 1)})
        self.log.info(f"  checkpoint q{k}: epoch {epoch} step {step_done:,}/{steps:,} "
                      f"-> last.pt (resumable) + ckpt/ep{epoch:02d}_q{k}.pt")

    def _snapshot(self, epoch: int, tag: str, meta: Dict) -> None:
        """Raw weights at this point, loadable like best.pt ({state_dict, meta})."""
        d = self.run_dir / "ckpt"
        d.mkdir(exist_ok=True)
        tmp = d / f"ep{epoch:02d}_{tag}.pt.tmp"
        torch.save({"state_dict": self.model.state_dict(),
                    "meta": {"epoch": epoch, "tag": tag, **meta}}, tmp)
        tmp.replace(d / f"ep{epoch:02d}_{tag}.pt")

    def _score(self, val: Dict[str, object]) -> float:
        """The selection metric, always as "higher is better"."""
        return (-float(val["rmse"]) if self.select_metric == "val_rmse"
                else float(val["zone_a"]))

    def _band_cfg(self, loss_cfg: Dict) -> Dict:
        """Inject the zone-A band in MODEL space. THE one place it is derived.

        Zone A is ``|log(p/y) - CENTRE| <= HALFWIDTH``. The target standardiser
        divides model-space values by ``target_scale``, so the window shrinks by
        the same factor -- which is why these are computed here, from the frozen
        standardiser, and not in a config file. Only a log target mode makes any
        band-aware loss meaningful, and ``band_constants`` raises otherwise.
        """
        if not needs_band(loss_cfg):
            return loss_cfg
        std = getattr(self.bundle, "standardizer", None)
        cfg = dict(loss_cfg)
        cfg.update(band_constants(getattr(std, "target_scale", None),
                                  str(getattr(std, "target_mode", "delta"))))
        # The centre is part of the band's definition, so zone_t always gets it --
        # its whole objective is "how much mass is inside [c-r, c+r]". For the
        # point losses it is an opt-in SHIFT of the residual, off by default so
        # that every existing run reproduces bit for bit.
        if not bool(cfg.get("band_offset", False)) \
                and str(cfg.get("name", "")).lower() != "zone_t":
            cfg["band_centre"] = None
        self.log.info(
            "zone-A band (model space): halfwidth=%s centre=%s"
            % ([round(float(v), 4) for v in cfg["band_halfwidth"]],
               None if cfg["band_centre"] is None
               else [round(float(v), 4) for v in cfg["band_centre"]]))
        if cfg.get("band_centre") is not None and \
                str(cfg.get("name", "")).lower() in ("hybrid", "quantile"):
            self.log.warning(
                "band_offset with a quantile loss: the median column is both the "
                "point forecast and the q50 target, so the Huber and pinball terms "
                "pull it 2% apart. Huber dominates ~28:1 so the point lands where "
                "the band wants it, but training.loss.name=zone_t is the version "
                "that separates the decision from the law properly.")
        return cfg

    def _init_ema(self, cfg: Dict) -> None:
        """Optional exponential moving average of the weights.

        The diagnosis in run_ideas.sh is that the remaining error is SPREAD, not
        bias -- at h=120 the sd of log(pred/true) is 0.2068 against a pass window
        of +/-0.2027, so one sd IS the pass mark. The one intervention that
        reliably reduces spread here is ensembling (+0.5, reproduced every time),
        and averaging weights along the trajectory is its free approximation.

        The conditions favour it unusually strongly: several shipped runs never
        converged (ss_ctrl_s0's val RMSE was still falling at epoch 6 of 6; all
        four c96 runs ended their final epoch on *best*), and a model still
        descending is exactly where trajectory averaging pays.

        The average never escapes the Trainer: whichever weights win are written
        to best.pt as an ordinary state_dict, so predict(), --init-from and
        submission/ are untouched.
        """
        self.ema = None
        if not bool(cfg.get("enabled", False)):
            return
        from torch.optim import swa_utils
        bad = [n for n, m in self.model.named_modules()
               if isinstance(m, nn.modules.batchnorm._BatchNorm)]
        if bad:
            raise ValueError(
                f"training.ema.enabled with BatchNorm layers ({bad[:3]}...): their "
                "running statistics need swa_utils.update_bn over the training set "
                "after averaging, which this loop does not do. No model in models/ "
                "used BatchNorm when this was written.")
        self.ema_start = int(cfg.get("start_epoch", 1))
        self.ema_eval_both = bool(cfg.get("eval_both", True))
        self.ema = swa_utils.AveragedModel(
            self.model, use_buffers=True,
            multi_avg_fn=swa_utils.get_ema_multi_avg_fn(float(cfg.get("decay", 0.999))))
        self.log.info("EMA on: decay=%s start_epoch=%s eval_both=%s"
                      % (cfg.get("decay", 0.999), self.ema_start, self.ema_eval_both))

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

    def _train_epoch(self, loader, steps: int, epoch: int, start_step: int = 0) -> float:
        self.model.train()
        total, n, n_bad = 0.0, 0, 0
        if start_step and self._resume_partial:
            total = float(self._resume_partial.get("total", 0.0))
            n = int(self._resume_partial.get("n", 0))
            n_bad = int(self._resume_partial.get("n_bad", 0))
        sampler = getattr(loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch, start=start_step * int(loader.batch_size))
        elif start_step:
            raise RuntimeError("mid-epoch resume needs the seeded train sampler")
        marks = {max(1, int(round(f * steps))): k
                 for k, f in enumerate(self.ckpt_fractions, 1)}
        self.optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader, start=start_step):
            if step >= steps:
                break
            batch = batch_to_device(batch, self.device)
            if self.freq_mask_p > 0.0:
                batch = self._freq_augment(batch)
            with torch.amp.autocast(self.device.type, enabled=self.amp):
                out = self.model(batch)
                target = batch["y_norm"]
                mask = batch["y_mask"]
                loss = self.criterion(loss_inputs(self.criterion, out), target, mask)
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
                if self.max_nonfinite is not None and n_bad > self.max_nonfinite:
                    raise RuntimeError(
                        f"epoch {epoch}: {n_bad} non-finite batches exceeds "
                        f"training.max_nonfinite_batches={self.max_nonfinite}. "
                        "Stopping rather than silently training on a skewed subset. "
                        f"last.pt is from before this point; inspect, then resume.")
                self.optimizer.zero_grad(set_to_none=True)
                if (step + 1) in marks:
                    self._mid_checkpoint(epoch, step + 1, marks[step + 1], steps,
                                         total, n, n_bad)
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
                if self.ema is not None and epoch >= self.ema_start:
                    # After scaler.update(), so the averaged weights are the fp32
                    # master weights the optimiser actually stepped.
                    self.ema.update_parameters(self.model)

            total += float(loss.detach()) * mask.shape[0]
            n += mask.shape[0]
            if (step + 1) in marks:
                self._mid_checkpoint(epoch, step + 1, marks[step + 1], steps,
                                     total, n, n_bad)
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
    def evaluate_loader(self, loader, model: Optional[nn.Module] = None
                        ) -> Dict[str, object]:
        """Validation metrics in mg/dL plus the model-space loss.

        ``zone_a`` is pooled over every (anchor, horizon) pair rather than being
        the mean of four per-horizon rates -- that is the leaderboard's own
        definition. It is computed for every run whatever ``select_metric`` is,
        because it costs one boolean mask and it is the number every loss
        experiment is actually about.
        """
        from ..evaluation.metrics import dts_zone_a_mask
        model = self.model if model is None else model
        model.eval()
        std = self.bundle.standardizer
        loss_sum, n_items = 0.0, 0
        se = np.zeros(len(self.horizons))
        ae = np.zeros(len(self.horizons))
        rel = np.zeros(len(self.horizons))
        cnt = np.zeros(len(self.horizons))
        zone = np.zeros(len(self.horizons))
        n_nonfinite = 0
        for batch in loader:
            batch = batch_to_device(batch, self.device)
            with torch.amp.autocast(self.device.type, enabled=self.amp):
                out = model(batch)
                loss = self.criterion(loss_inputs(self.criterion, out),
                                      batch["y_norm"], batch["y_mask"])
            loss_sum += float(loss) * batch["y_mask"].shape[0]
            n_items += batch["y_mask"].shape[0]

            p = std.inverse_targets(out.pred_norm.float(), batch["cgm_now"].float())
            p = p.cpu().numpy()
            y = batch["y"].cpu().numpy()
            m = batch["y_mask"].cpu().numpy() > 0
            n_nonfinite += int((m & ~np.isfinite(p)).sum())
            err = np.where(m, p - y, 0.0)
            se += (err ** 2).sum(axis=0)
            ae += np.abs(err).sum(axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                rel += np.where(m & (y > 0), np.abs(err) / np.where(y > 0, y, 1), 0.0).sum(axis=0)
            cnt += m.sum(axis=0)
            zone += (dts_zone_a_mask(y, p) & m).sum(axis=0)

        cnt_safe = np.maximum(cnt, 1)
        rmse_h = np.sqrt(se / cnt_safe)
        return {
            "loss": loss_sum / max(n_items, 1),
            "rmse": float(np.sqrt(se.sum() / max(cnt.sum(), 1))),
            "mae": float(ae.sum() / max(cnt.sum(), 1)),
            "mard": float(rel.sum() / max(cnt.sum(), 1) * 100.0),
            "zone_a": float(zone.sum() / max(cnt.sum(), 1) * 100.0),
            "rmse_per_horizon": rmse_h.tolist(),
            "zone_a_per_horizon": (zone / cnt_safe * 100.0).tolist(),
            "n_per_horizon": cnt.tolist(),
            "n_nonfinite": n_nonfinite,
        }
