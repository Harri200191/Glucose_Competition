"""CGM-JEPA (Muhammad et al., arXiv:2605.00933) -- faithful re-implementation.

Why re-implement rather than import their package: their ``models/encoder.py``
pulls in ``utils.mask_utils``, ``root.PROJECT_ROOT`` and a ``config`` package,
none of which exist here, and adding their repo to PYTHONPATH would shadow our
own ``models``/``utils``/``config`` module names. The architecture is 120 lines,
so we carry it, and we carry it *exactly*: every submodule and parameter name
below matches theirs so ``load_state_dict(..., strict=True)`` succeeds against
the published ``model.safetensors`` with nothing renamed or reshaped.

Verified against the real checkpoint (49 tensors, 1,002,160 params):

    data_embedding.value_embedding.proj.weight        (96, 1, 3)
    data_embedding.value_embedding.fc.weight          (96, 384)
    data_embedding.timefeature_embedding.proj.weight  (96, 5)
    data_embedding.positional_embedding.pos_emb       (1, 5000, 96)
    predictor_blocks.{0,1,2}.{norm1,attn.qkv,attn.proj,norm2,mlp.fc1,mlp.fc2}
    encoder_norm.{weight,bias}
    proj.{fc1,fc2}                                    (96 -> 1024 -> 48)

Two things about this architecture that matter downstream and are easy to
misread:

* ``dim_in`` (12) is the PATCH LENGTH in 5-minute steps, and ``kernel_size`` (3)
  is the stride of the intra-patch convolution -- not the other way round. A
  24 h window is 288 steps = 24 patches of 12; each patch is convolved with
  kernel 3 stride 3 into 4 positions x 96, flattened to 384 and projected to 96.
* the transformer is **bidirectional** over the 24 patches. That is not a leak
  here: every one of the 288 steps is at or before the forecast anchor, so
  attending across all of them is attending across history only.

Missing readings are the sentinel value -1 *after* normalisation, which is how
they trained it (``data_loaders/data_transformer.py``: ``mask = x != -1``).
"""
from __future__ import annotations

import json
import math
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None, act_layer=nn.GELU,
                 drop: float = 0.0) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False,
                 qk_scale: Optional[float] = None, attn_drop: float = 0.0,
                 proj_drop: float = 0.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = qk_scale or (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(out)), attn


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = False, qk_scale: Optional[float] = None,
                 drop: float = 0.0, attn_drop: float = 0.0,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MultiHeadAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                       qk_scale=qk_scale, attn_drop=attn_drop,
                                       proj_drop=drop)
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        y, _ = self.attn(self.norm1(x), mask=mask)
        x = x + y
        return x + self.mlp(self.norm2(x))


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pos_emb = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pos_emb[:, 0::2] = torch.sin(position * div)
        pos_emb[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pos_emb", pos_emb.unsqueeze(0))

    def forward(self, x_len: int) -> torch.Tensor:
        return self.pos_emb[:, :x_len]


class ValueEmbedding(nn.Module):
    """One patch of ``in_channels`` steps -> one ``dim``-vector."""

    def __init__(self, dim: int, in_channels: int, patch_size: int,
                 bias: bool = True) -> None:
        super().__init__()
        self.proj = nn.Conv1d(1, dim, kernel_size=patch_size, stride=patch_size,
                              bias=bias, padding=0)
        n_out = (in_channels - patch_size) // patch_size + 1
        self.fc = nn.Linear(dim * n_out, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, P = x.shape
        h = self.proj(x.reshape(-1, 1, P))
        return self.fc(h.reshape(h.size(0), -1)).view(B, N, -1)


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model: int, d_inp: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_inp, d_model)

    def forward(self, x_mark: torch.Tensor) -> torch.Tensor:
        # [B, N, patch, d_inp] -> [B, N, D], averaged within the patch
        B, N, P, D = x_mark.shape
        return self.proj(x_mark.reshape(-1, D).float()).view(B, N, P, -1).mean(dim=2)


class DataEmbedding(nn.Module):
    def __init__(self, dim: int, in_channels: int, patch_size: int,
                 time_inp_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.value_embedding = ValueEmbedding(dim, in_channels, patch_size)
        self.positional_embedding = PositionalEmbedding(dim)
        self.timefeature_embedding = TimeFeatureEmbedding(dim, time_inp_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, x_mark: Optional[torch.Tensor]) -> torch.Tensor:
        val = self.value_embedding(x)
        out = val + self.positional_embedding(val.size(1))
        if x_mark is not None:
            out = out + self.timefeature_embedding(x_mark)
        return self.dropout(out)


class Encoder(nn.Module):
    """The published CGM-JEPA / X-CGM-JEPA encoder.

    ``forward(x, x_mark)`` with ``x: [B, n_patches, patch_len]`` returns
    ``(embedding [B, n_patches, embed_dim], projection [B, n_patches, 48])``.
    We use the embedding; the 48-d projection head is the SSL target space and
    is loaded only so that ``strict=True`` holds.
    """

    def __init__(self, dim_in: int = 12, kernel_size: int = 3, embed_dim: int = 96,
                 embed_bias: bool = True, nhead: int = 6, num_layers: int = 3,
                 mlp_ratio: float = 4.0, qkv_bias: bool = True,
                 qk_scale: Optional[float] = None, drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0, norm_layer=nn.LayerNorm,
                 jepa: bool = False, time_inp_dim: int = 5, **_ignored) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_len = dim_in
        self.time_inp_dim = time_inp_dim
        self.data_embedding = DataEmbedding(embed_dim, dim_in, kernel_size,
                                            time_inp_dim, dropout=drop_rate)
        self.predictor_blocks = nn.ModuleList([
            Block(embed_dim, nhead, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                  norm_layer=norm_layer)
            for _ in range(num_layers)])
        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.jepa = jepa
        self.proj = MLP(embed_dim, hidden_features=1024, out_features=48)

    def forward(self, x: torch.Tensor,
                x_mark: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.data_embedding(x, x_mark)
        for blk in self.predictor_blocks:
            h = blk(h)
        h = self.encoder_norm(h)
        return h, self.proj(h)


# ------------------------------------------------------------------ checkpoint
HF_REPO = "CRUISEResearchGroup/CGM-JEPA"
VARIANTS = ("cgm_jepa", "x_cgm_jepa")


def load_pretrained(root: str, variant: str = "cgm_jepa") -> Encoder:
    """Build the encoder from ``<root>/<variant>/config.json`` + safetensors.

    ``root`` is the local mirror written by ``scripts/download_cgm_fm_weights.sh``
    (the same layout ``huggingface-cli download ... --local-dir`` produces).
    Loading is strict on purpose: a silently-renamed tensor would leave part of
    the network at its random initialisation and we would report a "pretrained"
    number that is nothing of the sort.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    d = os.path.join(root, variant)
    cfg_path, w_path = os.path.join(d, "config.json"), os.path.join(d, "model.safetensors")
    if not (os.path.isfile(cfg_path) and os.path.isfile(w_path)):
        raise FileNotFoundError(
            f"no CGM-JEPA checkpoint at {d}. Fetch it once with\n"
            f"    scripts/download_cgm_fm_weights.sh\n"
            f"(mirrors https://huggingface.co/{HF_REPO}, MIT licence, ~4 MB per variant).")
    with open(cfg_path) as fh:
        cfg = json.load(fh)
    model = Encoder(**cfg)
    from safetensors.torch import load_file
    missing, unexpected = model.load_state_dict(load_file(w_path), strict=False)
    # pos_emb is a deterministic buffer; everything else must come from the file.
    hard = [k for k in missing if "pos_emb" not in k]
    if hard or unexpected:
        raise RuntimeError(f"checkpoint mismatch for {variant}: "
                           f"missing={hard} unexpected={list(unexpected)}")
    return model
