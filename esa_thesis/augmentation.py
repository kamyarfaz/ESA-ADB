"""Augmentation components of the Mission 1 research pipeline.

Extracted without changing numerical behavior; see docs/thesis-research-context.md.
"""
from __future__ import annotations

import random
from functools import lru_cache
from itertools import accumulate
import torch
from .config import (PA_SUBWIN)


@lru_cache(maxsize=32)
def _overlapping_starts(mask, lower, upper):
    counts = [0, *accumulate(mask)]
    return {length: [s for s in range(len(mask)-length+1)
                     if counts[s+length] > counts[s]]
            for length in range(lower, upper+1)}


def make_pseudo_anomaly(xb: "torch.Tensor", *, target_mask=None) -> "torch.Tensor":
    """Synthesise localized pseudo-anomalies from NORMAL windows (no labels used).
    Mixes three corruption types motivated by the two anomaly mechanisms found on
    Mission 1 — magnitude/offset spikes (Cluster B), inter-channel relationship
    breaks (Cluster A), and noise bursts. Returns a corrupted COPY of xb; the model
    is trained to reconstruct these badly. xb: (B, C, SEQ_LEN, 1)."""
    B, C, T, _ = xb.shape
    # Optional placement constraint only: preserve type/length/channel/amplitude laws.
    # Overlap is geometric; a temporal roll on constant data may still be a no-op.
    valid_starts = None
    if target_mask is not None:
        mask = torch.as_tensor(target_mask, dtype=torch.bool).detach().cpu().flatten()
        if len(mask) != T or not mask.any():
            raise ValueError("target_mask must contain a target and match sequence length")
        lower = max(4, int(PA_SUBWIN[0] * T))
        upper = max(lower + 1, int(PA_SUBWIN[1] * T))
        valid_starts = _overlapping_starts(tuple(mask.tolist()), lower, upper)
    x = xb.detach().clone()
    dev = x.device
    lo = max(4, int(PA_SUBWIN[0] * T)); hi = max(lo + 1, int(PA_SUBWIN[1] * T))
    for i in range(B):
        typ  = random.randint(0, 2)
        wlen = random.randint(lo, hi)
        t0   = random.randint(0, T - wlen) if valid_starts is None else random.choice(valid_starts[wlen])
        nch  = random.randint(1, max(1, C // 2))
        chs  = torch.randperm(C, device=dev)[:nch]
        seg  = x[i, chs, t0:t0 + wlen, 0]                       # (nch, wlen)
        std  = seg.std(dim=1, keepdim=True) + 1e-6
        if typ == 0:                                           # magnitude offset / spike
            sign = torch.randint(0, 2, (nch, 1), device=dev).float() * 2 - 1
            amp  = torch.empty(nch, 1, device=dev).uniform_(3.0, 8.0)
            x[i, chs, t0:t0 + wlen, 0] = seg + sign * amp * std
        elif typ == 1:                                         # relationship break (temporal roll)
            shift = random.randint(1, wlen - 1) if wlen > 2 else 1
            x[i, chs, t0:t0 + wlen, 0] = torch.roll(seg, shifts=shift, dims=1)
        else:                                                  # noise burst
            x[i, chs, t0:t0 + wlen, 0] = seg + torch.randn_like(seg) * std * random.uniform(2.0, 5.0)
    return x
