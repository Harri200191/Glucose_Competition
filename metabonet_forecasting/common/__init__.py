"""Shared plumbing: configuration, registries, seeding, run logging."""

from .config import Config, load_config, save_config, apply_overrides
from .registry import BACKBONES, FUSIONS, register_backbone, register_fusion
from .seeding import seed_everything
from .logging import RunLogger, get_console_logger, provenance

__all__ = [
    "Config", "load_config", "save_config", "apply_overrides",
    "BACKBONES", "FUSIONS", "register_backbone", "register_fusion",
    "seed_everything", "RunLogger", "get_console_logger", "provenance",
]
