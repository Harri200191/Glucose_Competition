"""Configuration: YAML files, deep merge, dotted overrides, frozen provenance.

Discipline enforced here (plan section 9.3):

* every experiment setting lives in a YAML file under ``configs/``;
* a config may ``include:`` other configs, which are deep-merged in listed order
  with the including file winning;
* anything not in a file is a dotted ``--set key=value`` override, and those are
  recorded verbatim so a run is reconstructible from its manifest alone;
* the fully resolved config is written next to the run's outputs.

There is deliberately no implicit default anywhere else in the codebase: modules
read what the config says and raise if a required key is absent.
"""
from __future__ import annotations

import ast
import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "configs"


class Config(dict):
    """A dict with attribute access and dotted get/set.

    Subclassing ``dict`` keeps it trivially YAML/JSON serialisable, which matters
    because the resolved config is part of every run's provenance record.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:  # pragma: no cover - attribute protocol
            raise AttributeError(name) from exc
        return Config(value) if isinstance(value, dict) and not isinstance(value, Config) else value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    # ------------------------------------------------------------------ dotted
    def get_path(self, path: str, default: Any = ...) -> Any:
        """``cfg.get_path("model.backbone.d_model")``."""
        node: Any = self
        for part in path.split("."):
            if isinstance(node, Mapping) and part in node:
                node = node[part]
            elif default is not ...:
                return default
            else:
                raise KeyError(f"config key {path!r} not found (missing at {part!r})")
        return Config(node) if isinstance(node, dict) and not isinstance(node, Config) else node

    def set_path(self, path: str, value: Any) -> None:
        parts = path.split(".")
        node: Dict[str, Any] = self
        for part in parts[:-1]:
            nxt = node.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                node[part] = nxt
            node = nxt
        node[parts[-1]] = value

    def to_dict(self) -> Dict[str, Any]:
        return json.loads(json.dumps(self, default=str))

    def copy(self) -> "Config":  # type: ignore[override]
        return Config(copy.deepcopy(dict(self)))


# --------------------------------------------------------------------- merging
def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursive dict merge. Lists and scalars are replaced, never concatenated.

    Replacement is the right rule for this project: a config that says
    ``modalities: [cgm]`` means *only* CGM, and a merge that appended would
    silently turn a modality ablation into the full model.
    """
    out: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], Mapping) and isinstance(value, Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _coerce(text: str) -> Any:
    """Parse a CLI override value: ``3`` -> int, ``[1,2]`` -> list, ``foo`` -> str."""
    lowered = text.strip().lower()
    if lowered in {"none", "null", "~"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def apply_overrides(cfg: Config, overrides: Optional[Iterable[str]]) -> Config:
    """Apply ``key.path=value`` strings in order, recording them in the config."""
    applied: List[str] = list(cfg.get("_overrides", []))
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override {item!r} is not of the form key.path=value")
        key, _, raw = item.partition("=")
        cfg.set_path(key.strip(), _coerce(raw))
        applied.append(item)
    cfg["_overrides"] = applied
    return cfg


# --------------------------------------------------------------------- loading
def _resolve(path_like: str, relative_to: Optional[Path] = None) -> Path:
    """Resolve a config reference.

    Accepted, in order: an existing path as given; a path relative to the
    including file; a path under ``configs/``; the same with ``.yaml`` appended.
    """
    candidates: List[Path] = []
    raw = Path(path_like)
    candidates.append(raw)
    if relative_to is not None:
        candidates.append(relative_to / raw)
    candidates.append(CONFIG_ROOT / raw)
    with_ext = [c.with_suffix(".yaml") for c in candidates if c.suffix == ""]
    for cand in candidates + with_ext:
        if cand.is_file():
            return cand.resolve()
    raise FileNotFoundError(
        f"config {path_like!r} not found (looked in ., {relative_to}, {CONFIG_ROOT})"
    )


def load_config(path_like: str, overrides: Optional[Sequence[str]] = None,
                _seen: Optional[List[Path]] = None) -> Config:
    """Load a YAML config, resolving ``include:`` recursively, then apply overrides.

    ``include`` entries are merged in listed order and the including file wins,
    so ``configs/experiment/e4_fusion_film.yaml`` can include the mamba-2 model
    config and then change three keys.
    """
    path = _resolve(path_like)
    seen = list(_seen or [])
    if path in seen:
        raise ValueError(f"circular include: {' -> '.join(str(p) for p in seen + [path])}")
    seen.append(path)

    with open(path, "r") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config {path} must be a mapping at top level")

    includes = raw.pop("include", []) or []
    if isinstance(includes, str):
        includes = [includes]

    merged: Dict[str, Any] = {}
    for inc in includes:
        merged = deep_merge(merged, load_config(inc, None, seen))
    merged = deep_merge(merged, raw)

    cfg = Config(merged)
    cfg["_config_files"] = list(cfg.get("_config_files", [])) + [str(path.relative_to(REPO_ROOT))]
    return apply_overrides(cfg, overrides)


def save_config(cfg: Config, path: os.PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        yaml.safe_dump(cfg.to_dict(), fh, sort_keys=False, default_flow_style=False)


def config_from_args(argv_config: str, sets: Optional[Sequence[str]]) -> Config:
    """Convenience wrapper used by every entry point."""
    return load_config(argv_config, sets)
