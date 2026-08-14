"""Name -> constructor registries.

The point of a registry here is not extensibility for its own sake: it is that
``training/train.py`` must be unable to special-case an architecture. It receives
a name from the config, looks it up, and builds it. Any per-architecture handling
has to live inside that architecture's module, where it is visible.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Type


class Registry(dict):
    def __init__(self, kind: str) -> None:
        super().__init__()
        self.kind = kind

    def register(self, name: str) -> Callable[[Type], Type]:
        def deco(cls: Type) -> Type:
            if name in self:
                raise KeyError(f"{self.kind} {name!r} already registered by {self[name]}")
            self[name] = cls
            return cls
        return deco

    def build(self, name: str, **kwargs: Any) -> Any:
        if name not in self:
            raise KeyError(
                f"unknown {self.kind} {name!r}; available: {sorted(self)}"
            )
        return self[name](**kwargs)


BACKBONES: Registry = Registry("backbone")
FUSIONS: Registry = Registry("fusion")
ENCODERS: Registry = Registry("encoder")


def register_backbone(name: str) -> Callable[[Type], Type]:
    return BACKBONES.register(name)


def register_fusion(name: str) -> Callable[[Type], Type]:
    return FUSIONS.register(name)


def register_encoder(name: str) -> Callable[[Type], Type]:
    return ENCODERS.register(name)
