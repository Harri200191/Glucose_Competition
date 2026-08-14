"""PatchTST backbone (Nie et al., ICLR 2023).

Tests: *does patch-based attention benefit longer context?*

Deviation from the paper, made deliberately and recorded here because it affects
interpretation: the original patches each raw channel independently
(channel-independence) and shares one Transformer across channels. Here the
patches are taken over the **fused latent sequence**, as the plan specifies, so
that PatchTST receives exactly the representation TCN and Mamba-2 receive. That
keeps the backbone comparison controlled at the cost of not reproducing
channel-independence; the raw-concat fusion setting is the closest available
approximation to the paper's input treatment.

Patch length and stride are logged with every run: patching is a form of temporal
downsampling, so a result at patch length 16 is not the same experiment as one at
patch length 8.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from ..common.registry import register_backbone
from .base import TemporalBackbone


@register_backbone("patchtst")
class PatchTSTBackbone(TemporalBackbone):
    def __init__(self, d_model: int, patch_len: int = 16, stride: Optional[int] = None,
                 n_layers: int = 3, n_heads: int = 8, d_ff: Optional[int] = None,
                 dropout: float = 0.1, context: Optional[int] = None,
                 causal_attention: bool = True) -> None:
        super().__init__(d_model)
        self.patch_len = patch_len
        self.stride = stride or patch_len          # non-overlapping by default
        self.causal_attention = causal_attention
        d_ff = d_ff or 4 * d_model

        self.embed = nn.Linear(patch_len * d_model, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self._n_patches(context or 288), d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def _n_patches(self, length: int) -> int:
        return max(1, 1 + (length - self.patch_len) // self.stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, L, d] -> [B, P, d]: one token per patch, not per step.

        The head pools the returned sequence, so returning patch tokens rather
        than re-expanding to L steps is both cheaper and faithful to the model.
        """
        b, length, d = x.shape
        if length < self.patch_len:
            raise ValueError(f"context {length} is shorter than patch_len {self.patch_len}")
        # Trailing-aligned patches: the window ends exactly at the anchor, so the
        # last patch must end at the last step. Any remainder is dropped from the
        # *start*, never the end -- dropping from the end would silently discard
        # the most recent readings.
        n_patches = self._n_patches(length)
        used = self.patch_len + (n_patches - 1) * self.stride
        x = x[:, length - used:]
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        # unfold -> [B, P, d, patch_len]; flatten channel-time inside each patch
        tokens = patches.permute(0, 1, 3, 2).reshape(b, n_patches, self.patch_len * d)
        h = self.embed(tokens) + self.pos[:, :n_patches]

        mask = None
        if self.causal_attention:
            # Patches are ordered in time; a causal mask keeps a patch from
            # attending to later patches. Within a patch the model does see all
            # its steps, which is inherent to patching (the paper does the same)
            # and is safe here because a patch never extends past the anchor.
            mask = torch.triu(torch.full((n_patches, n_patches), float("-inf"),
                                         device=x.device), diagonal=1)
        return self.norm(self.encoder(h, mask=mask))

    def receptive_field(self) -> Optional[int]:
        return None       # attention over all patches: the whole context

    def extra_repr(self) -> str:
        return f"patch_len={self.patch_len}, stride={self.stride}"
