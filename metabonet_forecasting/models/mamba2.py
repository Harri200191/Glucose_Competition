"""Mamba-2 backbone: selective state-space modelling (Dao & Gu, 2024).

Tests: *does selective state propagation improve long-context and
event-conditioned forecasting?* -- the reason this project cares is that insulin
and carbohydrate effects persist for hours after a sparse event, which is exactly
the regime where a selective state should beat a fixed convolutional window.

Implementation
--------------
The state-space duality (SSD) recurrence is implemented here in pure PyTorch, in
the chunked form from the paper: quadratic attention-like computation *within* a
chunk, linear recurrence *across* chunks. This is a transcription of the reference
``ssd_minimal_discrete`` algorithm, and ``tests/test_models.py`` checks it against
a step-by-step recurrence to 1e-4.

Pure PyTorch is the default rather than a fallback because the official CUDA
kernels require Ampere-class hardware and the machines this benchmark runs on are
Volta (V100). Set ``use_kernels: true`` in the config to use
``mamba_ssm.modules.mamba2.Mamba2`` where it is installed and supported; the
maths is the same, the speed is not.

The Mamba2 blocks are used as a **temporal backbone**, never as a language-model
head: no embedding table, no vocabulary projection, no next-token loss.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.registry import register_backbone
from .base import TemporalBackbone


# ------------------------------------------------------------------ SSD kernel
def segsum(x: torch.Tensor) -> torch.Tensor:
    """Segment sum: ``out[..., i, j] = sum(x[..., j+1:i+1])`` for j <= i, else -inf.

    ``exp(segsum(A))`` is the lower-triangular decay matrix of a 1-semiseparable
    operator, i.e. how much the state at step j still contributes at step i.
    """
    T = x.size(-1)
    cs = torch.cumsum(x, dim=-1)
    out = cs.unsqueeze(-1) - cs.unsqueeze(-2)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=0)
    return out.masked_fill(~mask, float("-inf"))


def ssd_chunked(X: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
                chunk_size: int,
                initial_states: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Chunked state-space duality scan.

    X  [b, l, h, p]   input, already multiplied by the discretisation step dt
    A  [b, l, h]      log-decay per step (negative), already multiplied by dt
    B  [b, l, h, n]   input projection to state space
    C  [b, l, h, n]   output projection from state space
    ->  Y [b, l, h, p], final_state [b, h, p, n]

    The sequence is left-padded to a multiple of ``chunk_size`` with zeros. Left
    rather than right because the window ends at the forecast anchor: padding the
    end would shift the "now" step.
    """
    b, l, h, p = X.shape
    pad = (-l) % chunk_size
    if pad:
        X = F.pad(X, (0, 0, 0, 0, pad, 0))
        A = F.pad(A, (0, 0, pad, 0))
        B = F.pad(B, (0, 0, 0, 0, pad, 0))
        C = F.pad(C, (0, 0, 0, 0, pad, 0))
    L = X.shape[1]
    c = L // chunk_size

    Xc = X.reshape(b, c, chunk_size, h, p)
    Bc = B.reshape(b, c, chunk_size, h, -1)
    Cc = C.reshape(b, c, chunk_size, h, -1)
    Ac = A.reshape(b, c, chunk_size, h).permute(0, 3, 1, 2)      # [b, h, c, chunk]
    A_cumsum = torch.cumsum(Ac, dim=-1)

    # 1. within-chunk (the "quadratic", attention-like term)
    Lmat = torch.exp(segsum(Ac))                                  # [b,h,c,q,q]
    Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", Cc, Bc, Lmat, Xc)

    # 2. state contributed by each chunk
    decay_states = torch.exp(A_cumsum[..., -1:] - A_cumsum)       # [b,h,c,q]
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", Bc, decay_states, Xc)

    # 3. linear recurrence across chunks
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)           # [b,c+1,h,p,n]
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[..., -1], (1, 0))))
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. chunk state -> outputs
    Y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", Cc, states, torch.exp(A_cumsum))

    Y = (Y_diag + Y_off).reshape(b, L, h, p)
    return Y[:, pad:], final_state


def ssd_recurrent_reference(X: torch.Tensor, A: torch.Tensor, B: torch.Tensor,
                            C: torch.Tensor) -> torch.Tensor:
    """The same operator written as an explicit step-by-step recurrence.

    Slow and obviously correct; exists so the chunked kernel can be tested against
    something whose causality is self-evident.
    """
    b, l, h, p = X.shape
    n = B.shape[-1]
    state = torch.zeros(b, h, p, n, dtype=X.dtype, device=X.device)
    out = []
    for t in range(l):
        decay = torch.exp(A[:, t]).reshape(b, h, 1, 1)
        state = decay * state + X[:, t].unsqueeze(-1) * B[:, t].unsqueeze(-2)
        out.append(torch.einsum("bhpn,bhn->bhp", state, C[:, t]))
    return torch.stack(out, dim=1)


# ----------------------------------------------------------------------- block
class RMSNormGated(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        if z is not None:
            x = x * F.silu(z)
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class Mamba2Block(nn.Module):
    """One Mamba-2 mixer: in-projection, causal depthwise conv, SSD, gated norm."""

    def __init__(self, d_model: int, d_state: int = 64, d_conv: int = 4,
                 expand: int = 2, headdim: int = 64, chunk_size: int = 32,
                 dt_min: float = 0.001, dt_max: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        if self.d_inner % headdim != 0:
            raise ValueError(f"d_inner {self.d_inner} not divisible by headdim {headdim}")
        self.nheads = self.d_inner // headdim
        self.headdim = headdim
        self.d_state = d_state
        self.chunk_size = chunk_size

        d_in_proj = 2 * self.d_inner + 2 * d_state + self.nheads
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False)

        conv_dim = self.d_inner + 2 * d_state
        self.d_conv = d_conv
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, kernel_size=d_conv,
                                groups=conv_dim, padding=0)

        # dt bias initialised so softplus(dt_bias) lands in [dt_min, dt_max]
        dt = torch.exp(torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min))
                       + math.log(dt_min)).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        # A is negative-real and per head: the state decays, never explodes.
        self.A_log = nn.Parameter(torch.log(torch.empty(self.nheads).uniform_(1, 16)))
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.norm = RMSNormGated(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        b, l, _ = u.shape
        zxbcdt = self.in_proj(u)
        z, xBC, dt = torch.split(
            zxbcdt, [self.d_inner, self.d_inner + 2 * self.d_state, self.nheads], dim=-1)

        # causal depthwise convolution: left pad only
        xBC = F.pad(xBC.transpose(1, 2), (self.d_conv - 1, 0))
        xBC = F.silu(self.conv1d(xBC).transpose(1, 2))
        x, B, C = torch.split(xBC, [self.d_inner, self.d_state, self.d_state], dim=-1)

        dt = F.softplus(dt + self.dt_bias)                        # [b, l, h]
        A = -torch.exp(self.A_log.float())                        # [h]
        x = x.reshape(b, l, self.nheads, self.headdim)
        B = B.unsqueeze(2).expand(-1, -1, self.nheads, -1)        # ngroups = 1
        C = C.unsqueeze(2).expand(-1, -1, self.nheads, -1)

        # The scan runs in fp32 even under autocast: it is a cumulative product of
        # decays over hundreds of steps, and fp16 loses the tail of the state,
        # which is precisely the long-range information this backbone is here to
        # test. The projections around it stay in half precision.
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            y, _ = ssd_chunked((x * dt.unsqueeze(-1)).float(),
                               (A.reshape(1, 1, -1) * dt).float(),
                               B.float(), C.float(), self.chunk_size)
        y = y.to(x.dtype) + x * self.D.reshape(1, 1, -1, 1)
        y = y.reshape(b, l, self.d_inner)
        return self.out_proj(self.norm(y, z))


class Mamba2Layer(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, **block_kwargs) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mixer = Mamba2Block(d_model, **block_kwargs)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.mixer(self.norm(x)))


class _KernelLayer(nn.Module):
    """Wraps the official CUDA Mamba2 module with the same residual structure."""

    def __init__(self, d_model: int, dropout: float = 0.0, **block_kwargs) -> None:
        super().__init__()
        from mamba_ssm.modules.mamba2 import Mamba2 as OfficialMamba2
        self.norm = nn.LayerNorm(d_model)
        self.mixer = OfficialMamba2(
            d_model=d_model,
            d_state=block_kwargs.get("d_state", 64),
            d_conv=block_kwargs.get("d_conv", 4),
            expand=block_kwargs.get("expand", 2),
            headdim=block_kwargs.get("headdim", 64),
            chunk_size=block_kwargs.get("chunk_size", 32),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.mixer(self.norm(x)))


@register_backbone("mamba2")
class Mamba2Backbone(TemporalBackbone):
    def __init__(self, d_model: int, n_layers: int = 4, d_state: int = 64,
                 d_conv: int = 4, expand: int = 2, headdim: int = 32,
                 chunk_size: int = 32, dropout: float = 0.1,
                 use_kernels: bool = False, context: Optional[int] = None) -> None:
        super().__init__(d_model)
        block_kwargs = dict(d_state=d_state, d_conv=d_conv, expand=expand,
                            headdim=headdim, chunk_size=chunk_size)
        layer_cls = Mamba2Layer
        self.using_kernels = False
        if use_kernels:
            try:
                import mamba_ssm  # noqa: F401
                layer_cls = _KernelLayer
                self.using_kernels = True
            except Exception:
                layer_cls = Mamba2Layer
        self.layers = nn.ModuleList([
            layer_cls(d_model, dropout=dropout, **block_kwargs) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

    def receptive_field(self) -> Optional[int]:
        return None      # a state, not a window: unbounded in principle
