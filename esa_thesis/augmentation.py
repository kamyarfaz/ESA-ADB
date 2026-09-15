"""Augmentation components of the Mission 1 research pipeline.

Extracted without changing numerical behavior; see docs/thesis-research-context.md.
"""
from __future__ import annotations

import random
import torch
from .config import (PA_SUBWIN)


def make_pseudo_anomaly(xb: "torch.Tensor") -> "torch.Tensor":
    """Synthesise localized pseudo-anomalies from NORMAL windows (no labels used).
    Mixes three corruption types motivated by the two anomaly mechanisms found on
    Mission 1 — magnitude/offset spikes (Cluster B), inter-channel relationship
    breaks (Cluster A), and noise bursts. Returns a corrupted COPY of xb; the model
    is trained to reconstruct these badly. xb: (B, C, SEQ_LEN, 1)."""
    B, C, T, _ = xb.shape
    x = xb.detach().clone()
    dev = x.device
    lo = max(4, int(PA_SUBWIN[0] * T)); hi = max(lo + 1, int(PA_SUBWIN[1] * T))
    for i in range(B):
        typ  = random.randint(0, 2)
        wlen = random.randint(lo, hi)
        t0   = random.randint(0, T - wlen)
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
