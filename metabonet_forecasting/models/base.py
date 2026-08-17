"""The common model API (plan 9.2).

Every architecture in this repository implements the same contract:

    inputs  = the batch dict produced by data/windows.py
    outputs = ForecastOutput(pred_norm[B, H], aux)

``pred_norm`` is in model space (delta from the last CGM reading, standardised);
converting to mg/dL is the standardizer's job and happens once, in evaluation.
Nothing in ``training`` may branch on which architecture it holds.

Two families implement it:

* ``FrontEndModel`` -- shallow modality encoders -> fusion -> temporal backbone ->
  multi-horizon head. TCN, PatchTST and Mamba-2 differ *only* in the backbone,
  which is what makes the comparison controlled.
* native models (TFT, TiDE) -- consume the same batch but arrange their own
  internals, because variable selection (TFT) and dense encoding (TiDE) are the
  thing being tested and wrapping them in the common front end would test nothing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn


@dataclass
class ForecastOutput:
    pred_norm: torch.Tensor                      # [B, H] point forecast, model space
    quantiles: Optional[torch.Tensor] = None     # [B, H, Q] if a quantile head is on
    aux: Dict[str, torch.Tensor] = field(default_factory=dict)


class ForecastModel(nn.Module):
    """Base class: shape bookkeeping plus a uniform ``forward``."""

    def __init__(self, n_horizons: int) -> None:
        super().__init__()
        self.n_horizons = n_horizons

    def forward(self, batch: Dict[str, torch.Tensor]) -> ForecastOutput:  # pragma: no cover
        raise NotImplementedError

    # ------------------------------------------------------------------ helpers
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def receptive_field(self) -> Optional[int]:
        """Steps of history the architecture can actually see, where finite.

        Reported next to every result: a TCN whose receptive field is shorter than
        the context window is not testing long context, it is testing a shorter one.
        """
        return None

    def describe(self) -> Dict[str, object]:
        return {"class": type(self).__name__,
                "parameters": self.n_parameters(),
                "receptive_field": self.receptive_field()}


class TemporalBackbone(nn.Module):
    """[B, L, d_model] -> [B, L, d_model], strictly causal.

    Backbones return the full sequence rather than a pooled vector so the head can
    choose (last step, mean, or both) identically for every architecture.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def receptive_field(self) -> Optional[int]:
        return None
