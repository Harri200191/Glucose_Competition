"""Seeding.

Plan section 8.3 asks for multiple training seeds per neural model with reported
seed variability, which only means something if a seed actually pins the run down.
``seed_everything`` seeds python/numpy/torch and, when ``deterministic`` is set,
also switches cuDNN out of benchmark mode.
"""
from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, deterministic: bool = False) -> int:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32 - 1))
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dep in practice
        return seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    else:
        torch.backends.cudnn.benchmark = True
    return seed


def worker_init_fn(worker_id: int) -> None:
    """Give every dataloader worker a distinct, reproducible numpy stream."""
    import torch
    base = torch.initial_seed() % (2 ** 32 - 1)
    np.random.seed((base + worker_id) % (2 ** 32 - 1))
    random.seed(base + worker_id)
