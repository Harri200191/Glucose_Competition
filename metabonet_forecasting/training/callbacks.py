"""Early stopping, checkpointing, and the learning-rate schedule.

Early stopping watches validation **RMSE in mg/dL**, not the training loss: the
loss is in standardised model space, and a run must be stopped on the quantity it
will be judged on.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional

import torch


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0,
                 mode: str = "min") -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.sign = 1.0 if mode == "min" else -1.0
        self.best: float = math.inf
        self.best_epoch: int = -1
        self.bad_epochs: int = 0

    def step(self, value: float, epoch: int) -> bool:
        """Returns True when this is a new best."""
        scaled = self.sign * value
        if scaled < self.best - self.min_delta:
            self.best, self.best_epoch, self.bad_epochs = scaled, epoch, 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience

    @property
    def best_value(self) -> float:
        return self.sign * self.best


class Checkpointer:
    """Keeps exactly one file: the best model by the monitored metric."""

    def __init__(self, run_dir: Path, filename: str = "best.pt") -> None:
        self.path = Path(run_dir) / filename
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, model: torch.nn.Module, meta: Dict) -> Path:
        torch.save({"state_dict": model.state_dict(), "meta": meta}, self.path)
        return self.path

    def load_into(self, model: torch.nn.Module, map_location="cpu") -> Dict:
        blob = torch.load(self.path, map_location=map_location, weights_only=False)
        model.load_state_dict(blob["state_dict"])
        return blob.get("meta", {})

    def exists(self) -> bool:
        return self.path.exists()


class WarmupCosine:
    """Linear warmup then cosine decay, stepped per optimiser step."""

    def __init__(self, optimizer, total_steps: int, warmup_steps: int = 0,
                 min_lr_frac: float = 0.05) -> None:
        self.opt = optimizer
        self.total = max(total_steps, 1)
        self.warmup = max(min(warmup_steps, self.total - 1), 0)
        self.min_frac = min_lr_frac
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.step_num = 0

    def step(self) -> float:
        self.step_num += 1
        if self.step_num <= self.warmup and self.warmup > 0:
            frac = self.step_num / self.warmup
        else:
            progress = (self.step_num - self.warmup) / max(self.total - self.warmup, 1)
            progress = min(max(progress, 0.0), 1.0)
            frac = self.min_frac + (1 - self.min_frac) * 0.5 * (1 + math.cos(math.pi * progress))
        for group, base in zip(self.opt.param_groups, self.base_lrs):
            group["lr"] = base * frac
        return self.opt.param_groups[0]["lr"]
