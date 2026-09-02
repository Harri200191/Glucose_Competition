"""Forecast heads (plan stage 5): one shared design for every backbone,
plus an opt-in cross-attention decoder for the future-covariate condition.

The head is shared on purpose. If PatchTST used a flatten-all-patches head and
the TCN used a last-step head, part of any measured difference would be the head.
Here every backbone hands over ``[B, L, d_model]`` and the head pools it the same
way.

``pooling``:
    last       the final step -- the natural "now" representation
    mean       average over the window
    last_mean  both, concatenated (default): the last step carries the current
               state, the mean carries the recent regime
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn


class MultiHorizonHead(nn.Module):
    """[B, L, d] -> [B, H] point forecasts, optionally [B, H, Q] quantiles."""

    def __init__(self, d_model: int, n_horizons: int, hidden: Optional[int] = None,
                 pooling: str = "last_mean", dropout: float = 0.0,
                 quantiles: Optional[Sequence[float]] = None,
                 d_extra: int = 0) -> None:
        super().__init__()
        if pooling not in ("last", "mean", "last_mean"):
            raise ValueError(f"unknown pooling {pooling!r}")
        self.pooling = pooling
        self.n_horizons = n_horizons
        self.quantiles = list(quantiles) if quantiles else []
        # d_extra is the planned-action embedding, when a run opted into one. It
        # enters here, at the single point every architecture shares, so that the
        # future-covariate comparison is not confounded by *where* the
        # information was injected.
        self.d_extra = d_extra
        in_dim = d_model * (2 if pooling == "last_mean" else 1) + d_extra
        hidden = hidden or max(d_model, 64)
        n_out = n_horizons * (1 + len(self.quantiles))
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_out),
        )
        # Start from "no change": with delta targets, a zero output is the
        # persistence forecast, which is a sane and honest initial state.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def pool(self, seq: torch.Tensor) -> torch.Tensor:
        if self.pooling == "last":
            return seq[:, -1]
        if self.pooling == "mean":
            return seq.mean(dim=1)
        return torch.cat([seq[:, -1], seq.mean(dim=1)], dim=-1)

    def forward(self, seq: torch.Tensor, extra: Optional[torch.Tensor] = None):
        pooled = self.pool(seq)
        if self.d_extra:
            if extra is None:
                raise ValueError("head was built with d_extra but got no extra input")
            pooled = torch.cat([pooled, extra], dim=-1)
        out = self.net(pooled)
        point = out[:, :self.n_horizons]
        if not self.quantiles:
            return point, None
        q = out[:, self.n_horizons:].reshape(-1, self.n_horizons, len(self.quantiles))
        return point, _monotone_quantiles(point, q, len(self.quantiles))


def _monotone_quantiles(point: torch.Tensor, raw: torch.Tensor,
                        n_q: int) -> torch.Tensor:
    """Monotone quantiles by construction: cumulative softplus offsets around
    the point forecast, so the 90th can never fall below the 10th."""
    offsets = torch.cumsum(torch.nn.functional.softplus(raw), dim=-1)
    centre = offsets[..., n_q // 2: n_q // 2 + 1]
    return point.unsqueeze(-1) + (offsets - centre)


class HorizonCrossHead(nn.Module):
    """Per-horizon query tokens cross-attending over history and planned doses.

    The pooled head compresses 24 h of context into one vector and appends the
    future-insulin block as one flat embedding — the single most informative
    signal in the competition condition enters the network after all sequence
    processing is over. Here each horizon owns a query token, initialised from
    the anchor state, that attends over (a) every step of the backbone's output
    sequence and (b) one token per future 5-minute slot of the planned doses.
    A 60-minute query can weigh the bolus landing at +25 min differently from
    the 120-minute one — position-resolved credit assignment the flat embedding
    cannot express.

    Output contract matches ``MultiHorizonHead``: ``[B, H]`` point (zero-init,
    so training starts at persistence) plus optional monotone quantiles.

    ``chain`` lets the horizons condition on each other instead of being read
    off the shared memory independently, which is what they do by default --
    the query tokens cross-attend to history but never to one another, so
    nothing ties the 120-minute answer to the 60-minute one.

      none        the default; horizons are independent given the memory.
      causal      self-attention among the query tokens under a causal mask, so
                  the 60-minute query sees the 30-minute query's state, 90 sees
                  30 and 60, and so on. This is the *latent* form of feeding a
                  prediction forward: it passes the whole representation rather
                  than one scalar, and because every horizon is still supervised
                  from ground truth in the same step there is no exposure bias
                  and no error accumulation -- the two things that sink literal
                  recursive forecasting.
      full        the same but unmasked. Not a leak: all four are outputs at the
                  same anchor, so letting them see each other reveals nothing
                  about the future that the memory did not already contain.
      cumulative  parameterise the output as a running sum, pred_k = sum_{j<=k}
                  delta_j, so the head emits increments between horizons instead
                  of four unrelated levels. Encodes trajectory continuity, and
                  under target_mode=log_ratio the sum is a product of growth
                  factors, which is the physically right composition.
      causal_cum  both.
      value       the LITERAL version of the idea: the model's own predicted
                  value at horizon k-1 is projected and added to query k, so
                  the number forecast for +30 min is an input when forecasting
                  +60. Unlike the latent modes this is genuinely recursive, and
                  it is included so the difference can be measured rather than
                  argued about. It uses its OWN prediction at train time too --
                  never the ground truth -- so there is no exposure bias, but
                  error still accumulates along the chain, which is the cost the
                  latent modes avoid.
      dist        ``value``, but the whole predictive DISTRIBUTION goes forward
                  instead of the single number: the point plus the quantile
                  ladder recentred on it, i.e. level AND spread.

                  The motivation is measured, not assumed. On the internal
                  holdout an oracle allowed to pick the best of the seven
                  predicted quantiles per anchor scores 98.30 zone A at 120 min
                  against the median's 74.15 -- the right answer is inside the
                  predicted distribution 98% of the time and the model reports
                  the wrong part of it. Passing only the median discards exactly
                  the information that says whether to commit to a large move.

                  The ladder is passed as ``offsets - centre`` (what
                  ``_monotone_quantiles`` adds to the point), NOT as raw
                  quantile levels. That split is deliberate: the level is
                  already in the query through ``anchor_proj``, so handing over
                  raw quantiles would make the projection re-learn a
                  subtraction it can be given for free, and the spread is then
                  level-invariant -- a 20 mg/dL spread means the same thing at
                  90 and at 250.
    """

    CHAIN_MODES = ("none", "causal", "full", "cumulative", "causal_cum",
                   "value", "dist")

    def __init__(self, d_model: int, n_horizons: int,
                 future_steps: int = 0, future_channels: int = 0,
                 n_heads: int = 4, n_layers: int = 2, dropout: float = 0.0,
                 quantiles: Optional[Sequence[float]] = None,
                 chain: str = "none") -> None:
        super().__init__()
        self.chain = str(chain)
        if self.chain not in self.CHAIN_MODES:
            raise ValueError(f"head.chain must be one of {self.CHAIN_MODES}, "
                             f"got {chain!r}")
        self.chain_attends = self.chain in ("causal", "full", "causal_cum")
        self.chain_causal = self.chain in ("causal", "causal_cum")
        self.chain_cumulative = self.chain in ("cumulative", "causal_cum")
        self.chain_value = self.chain == "value"
        self.chain_dist = self.chain == "dist"
        self.chain_feeds = self.chain_value or self.chain_dist
        self.n_horizons = n_horizons
        self.quantiles = list(quantiles) if quantiles else []
        if self.chain_dist and not self.quantiles:
            raise ValueError(
                "head.chain='dist' passes the predictive distribution between "
                "horizons, so the head must emit quantiles. Set "
                "model.head.quantiles (and a quantile-aware loss), or use "
                "chain='value' to pass the point forecast alone.")
        self.future_steps = future_steps
        self.queries = nn.Parameter(torch.randn(n_horizons, d_model) * 0.02)
        self.anchor_proj = nn.Linear(d_model, d_model)
        if future_steps and future_channels:
            self.future_proj = nn.Linear(future_channels, d_model)
            self.future_pos = nn.Parameter(
                torch.randn(future_steps, d_model) * 0.02)
        else:
            self.future_proj = None
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(nn.ModuleDict({
                "ln_q": nn.LayerNorm(d_model),
                "ln_m": nn.LayerNorm(d_model),
                "attn": nn.MultiheadAttention(d_model, n_heads,
                                              dropout=dropout, batch_first=True),
                "ln_f": nn.LayerNorm(d_model),
                "ffn": nn.Sequential(
                    nn.Linear(d_model, 2 * d_model), nn.GELU(),
                    nn.Dropout(dropout), nn.Linear(2 * d_model, d_model)),
            }))
            if self.chain_attends:
                # Zero-init the output projection so an untrained chain layer is
                # the identity: chain=causal starts life numerically equal to
                # chain=none and can only earn its way in.
                sa = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                           batch_first=True)
                nn.init.zeros_(sa.out_proj.weight)
                nn.init.zeros_(sa.out_proj.bias)
                self.blocks[-1]["ln_c"] = nn.LayerNorm(d_model)
                self.blocks[-1]["chain"] = sa
        # Feeds the previous horizon's predicted value back in. Zero-init, so
        # chain=value also starts numerically identical to chain=none.
        if self.chain_feeds:
            # dist hands over the point AND the recentred ladder; value only
            # the point. Zero-init either way, so both start numerically
            # identical to chain=none and have to earn their way in.
            self.chain_in = 1 + (len(self.quantiles) if self.chain_dist else 0)
            self.value_proj = nn.Linear(self.chain_in, d_model)
            nn.init.zeros_(self.value_proj.weight)
            nn.init.zeros_(self.value_proj.bias)
        self.out_norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, 1 + len(self.quantiles))
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, seq: torch.Tensor, future_block: Optional[torch.Tensor] = None):
        B = seq.shape[0]
        memory = seq
        if self.future_proj is not None:
            if future_block is None:
                raise ValueError("head was built for future covariates but got none")
            fut = self.future_proj(future_block) + self.future_pos.unsqueeze(0)
            memory = torch.cat([memory, fut], dim=1)
        q = self.queries.unsqueeze(0).expand(B, -1, -1) \
            + self.anchor_proj(seq[:, -1]).unsqueeze(1)
        cmask = None
        if self.chain_causal:
            cmask = torch.triu(torch.ones(self.n_horizons, self.n_horizons,
                                          dtype=torch.bool, device=q.device),
                               diagonal=1)
        for blk in self.blocks:
            attn_out, _ = blk["attn"](blk["ln_q"](q), blk["ln_m"](memory),
                                      blk["ln_m"](memory), need_weights=False)
            q = q + attn_out
            if self.chain_attends:
                h = blk["ln_c"](q)
                chain_out, _ = blk["chain"](h, h, h, attn_mask=cmask,
                                            need_weights=False)
                q = q + chain_out
            q = q + blk["ffn"](blk["ln_f"](q))
        if self.chain_feeds:
            # Sequential by necessity: horizon k's input contains horizon k-1's
            # output. Four steps, so the cost is negligible.
            n_q = len(self.quantiles)
            steps = []
            prev = q.new_zeros(q.shape[0], self.chain_in)
            for k in range(self.n_horizons):
                qk = self.out_norm(q[:, k] + self.value_proj(prev))
                ok = self.out(qk)                            # [B, 1+Q]
                steps.append(ok)
                if self.chain_dist:
                    # The same offsets _monotone_quantiles will build, so what
                    # goes forward is the ladder that is actually reported --
                    # not a second, differently-shaped estimate of it.
                    off = torch.cumsum(
                        torch.nn.functional.softplus(ok[..., 1:]), dim=-1)
                    centre = off[..., n_q // 2: n_q // 2 + 1]
                    prev = torch.cat([ok[..., :1], off - centre], dim=-1)
                else:
                    prev = ok[..., :1]
            out = torch.stack(steps, dim=1)                  # [B, H, 1+Q]
        else:
            out = self.out(self.out_norm(q))                 # [B, H, 1+Q]
        point = out[..., 0]
        if self.chain_cumulative:
            point = torch.cumsum(point, dim=1)
        if not self.quantiles:
            return point, None
        return point, _monotone_quantiles(point, out[..., 1:],
                                          len(self.quantiles))
