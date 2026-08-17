"""Run logging and provenance.

Every run directory ends up with:

    config.resolved.yaml   the exact config after includes and overrides
    manifest.json          git commit, dataset manifest hash, seed, versions, timing
    train_log.csv          one row per epoch
    events.jsonl           append-only structured event log
    metrics.json           written by evaluation, not by training

Provenance is not decoration. Plan section 9.3 requires that a number in a table
can be traced to the dataset version, commit and seed that produced it.
"""
from __future__ import annotations

import csv
import json
import logging
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

_CONSOLE_CONFIGURED = False


def get_console_logger(name: str = "metabonet", level: int = logging.INFO) -> logging.Logger:
    global _CONSOLE_CONFIGURED
    logger = logging.getLogger(name)
    if not _CONSOLE_CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname).1s %(name)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
        _CONSOLE_CONFIGURED = True
    return logger


def _git(*args: str) -> Optional[str]:
    try:
        out = subprocess.run(["git"] + list(args), capture_output=True, text=True,
                             timeout=10, cwd=str(Path(__file__).resolve().parents[2]))
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:  # pragma: no cover - git may be absent
        return None


def provenance(extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Everything needed to re-create a run except the data itself."""
    versions: Dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("numpy", "pandas", "pyarrow", "torch", "sklearn", "scipy"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = None
    try:
        import torch
        versions["cuda"] = torch.version.cuda
        versions["gpu"] = (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except Exception:
        pass
    rec: Dict[str, Any] = {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "argv": sys.argv,
        "versions": versions,
    }
    if extra:
        rec.update(dict(extra))
    return rec


class RunLogger:
    """Writes the four files above and mirrors progress to the console."""

    def __init__(self, run_dir: Path, name: str = "metabonet") -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.console = get_console_logger(name)
        self._events = open(self.run_dir / "events.jsonl", "a", buffering=1)
        self._csv_path = self.run_dir / "train_log.csv"
        self._csv_fields: List[str] = []
        self._t0 = time.time()

    # ------------------------------------------------------------------ output
    def info(self, msg: str) -> None:
        self.console.info(msg)
        self.event("log", message=msg)

    def event(self, kind: str, **fields: Any) -> None:
        rec = {"t": round(time.time() - self._t0, 3), "kind": kind}
        rec.update(fields)
        self._events.write(json.dumps(rec, default=str) + "\n")

    def epoch(self, row: Mapping[str, Any]) -> None:
        """Append one row to train_log.csv, creating the header on first use."""
        row = dict(row)
        if not self._csv_fields:
            self._csv_fields = list(row)
            with open(self._csv_path, "w", newline="") as fh:
                csv.DictWriter(fh, self._csv_fields).writeheader()
        with open(self._csv_path, "a", newline="") as fh:
            csv.DictWriter(fh, self._csv_fields, extrasaction="ignore").writerow(row)
        self.event("epoch", **row)

    def write_json(self, name: str, payload: Any) -> Path:
        path = self.run_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2, default=str)
        return path

    def close(self) -> None:
        try:
            self._events.close()
        except Exception:
            pass


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)
