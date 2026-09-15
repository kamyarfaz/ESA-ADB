"""Scoring components of the Mission 1 research pipeline.

Extracted without changing numerical behavior; see docs/thesis-research-context.md.
"""
from __future__ import annotations

from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Tuple
import gc
import numpy as np
import torch
import torch.nn as nn
from .config import (DEVICE, EVAL_BATCH_SIZE, MLP_PRED_LEN, SCORE_STRIDE, SEQ_LEN)
from .data import (AllWindows, ForecastWindows)
from .models import (MLPForecaster)


@torch.no_grad()
def score_series(model: nn.Module, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ds = AllWindows(x)
    dl = DataLoader(ds, EVAL_BATCH_SIZE, shuffle=False,
                    num_workers=0, pin_memory=False)  # 1497427FIX: pin_memory=False avoids CUDA driver crashes on large inference passes
    C, T = x.shape[1], len(x)
    # FIX: use float32 instead of float64 — halves accumulator memory on large test sets
    score_sum = np.zeros((C, T), dtype=np.float32)
    count     = np.zeros(T,      dtype=np.float32)

    model.eval()
    for bi, xb in enumerate(dl):
        xb   = xb.to(DEVICE, non_blocking=True)
        # mse: (B, C)  — mean over time dimension
        mse  = ((xb - model(xb)) ** 2).squeeze(-1).mean(dim=-1).detach().cpu().numpy()
        starts = ds.starts[bi*EVAL_BATCH_SIZE:(bi+1)*EVAL_BATCH_SIZE]
        B_actual = len(starts)

        # Build index arrays for vectorised scatter-add
        # idx shape: (B_actual,) for count, (B_actual, SEQ_LEN) for score_sum
        start_arr = starts[:B_actual]               # (B,)
        offsets   = np.arange(SEQ_LEN, dtype=np.int64)  # (SEQ_LEN,)
        idx       = (start_arr[:, None] + offsets[None, :]).ravel()  # (B*SEQ_LEN,)

        np.add.at(count, idx, 1.0)
        # mse broadcast: (B, C) → per-channel, per-window scatter
        # score_sum shape: (C, T)
        for c in range(C):
            np.add.at(score_sum[c], idx, np.repeat(mse[:B_actual, c], SEQ_LEN))

    count     = np.where(count == 0, 1.0, count)
    per_ch    = (score_sum / count[None, :]).astype(np.float32)
    return per_ch, per_ch.max(axis=0).astype(np.float32)


@torch.no_grad()
def score_series_mlp(model: MLPForecaster,
                     x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Score a series using MLP prediction error (MAE).
    Scores are assigned to the PREDICTED positions (future timesteps),
    so the anomaly score at time t reflects how surprising the value at t
    was given the SEQ_LEN context ending at t-1.
    Returns (per_channel_score, aggregate_score), both shape (T,).
    """
    T, C   = x.shape
    ds     = ForecastWindows(x, stride=SCORE_STRIDE, normal_only=False)
    dl     = DataLoader(ds, EVAL_BATCH_SIZE, shuffle=False,
                        num_workers=0, pin_memory=False)
    score_sum = np.zeros((C, T), dtype=np.float32)
    count     = np.zeros(T,      dtype=np.float32)

    model.eval()
    ptr = 0
    for ctx_b, tgt_b in dl:
        ctx_b = ctx_b.to(DEVICE, non_blocking=True)   # (B, C, SEQ_LEN, 1)
        tgt_b = tgt_b.to(DEVICE, non_blocking=True)   # (B, C, PRED_LEN, 1)
        pred  = model(ctx_b)                           # (B, C, PRED_LEN, 1)
        # MAE per (batch, channel, step)
        mae   = (pred - tgt_b).abs().squeeze(-1).detach().cpu().numpy()  # (B,C,PRED_LEN)
        B_act = ctx_b.shape[0]
        # Assign scores to predicted positions
        starts_batch = ds.starts[ptr: ptr + B_act]
        for b in range(B_act):
            base = int(starts_batch[b]) + SEQ_LEN     # first predicted position
            end  = min(base + MLP_PRED_LEN, T)
            length = end - base
            if length <= 0:
                continue
            idx = np.arange(base, end, dtype=np.int64)
            np.add.at(count,       idx, 1.0)
            for c in range(C):
                np.add.at(score_sum[c], idx, mae[b, c, :length])
        ptr += B_act

    gc.collect()
    count     = np.where(count == 0, 1.0, count)
    per_ch    = (score_sum / count[None, :]).astype(np.float32)
    return per_ch, per_ch.max(axis=0).astype(np.float32)


def ensemble_scores(ae_score: np.ndarray,
                    mlp_score: np.ndarray) -> np.ndarray:
    """
    Combine AE and MLP scores into a single ensemble score.
    Strategy: OR-logic via element-wise max of min-max normalised scores.
    Both scores are normalised to [0,1] independently so their scales
    don't bias the combination.
    """
    def norm(s: np.ndarray) -> np.ndarray:
        lo, hi = float(s.min()), float(s.max())
        return (s - lo) / (hi - lo + 1e-12)

    return np.maximum(norm(ae_score), norm(mlp_score)).astype(np.float32)
