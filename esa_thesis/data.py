"""Data components of the Mission 1 research pipeline.

Extracted without changing numerical behavior; see docs/thesis-research-context.md.
"""
from __future__ import annotations

from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
from .config import (CLIP_Z, MAX_TRAIN_WINDOWS, MLP_PRED_LEN, SCORE_STRIDE, SEED, SEQ_LEN, STRIDE)
from .runtime import (log)


def read_columns(path: Path) -> List[str]:
    return pd.read_csv(path, nrows=0, low_memory=False).columns.tolist()


def get_y_any(df: pd.DataFrame) -> np.ndarray:
    y = np.zeros(len(df), dtype=np.int8)
    for c in df.columns:
        if c.startswith("is_anomaly_"):
            vals = pd.to_numeric(df[c], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
            y = np.maximum(y, (vals > 0).astype(np.int8))
    return y


def load_frame(path: Path, features: List[str]) -> pd.DataFrame:
    cols   = read_columns(path)
    labels = [c for c in cols if c.startswith("is_anomaly_")]
    need   = list(dict.fromkeys(["timestamp"] + features + labels))
    df     = pd.read_csv(path, usecols=lambda c: c in need, low_memory=False)
    for f in features:
        if f not in df.columns:
            df[f] = 0.0
        df[f] = pd.to_numeric(df[f], errors="coerce").ffill().bfill().fillna(0.0)
    for c in labels:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(np.int8)
    return df


class RobustChannelScaler:
    def fit(self, x: np.ndarray, y_any: np.ndarray):
        normal = y_any == 0
        meds, iqrs = [], []
        for j in range(x.shape[1]):
            vals = x[:, j]
            pts  = vals[normal]
            pts  = pts[np.isfinite(pts)]
            if len(pts) < 100:
                pts = vals[np.isfinite(vals)]
            if len(pts) == 0:
                pts = np.array([0.0], dtype=np.float32)
            med        = float(np.median(pts))
            q75, q25   = np.percentile(pts, [75, 25])
            iqr        = float(q75 - q25)
            if not np.isfinite(iqr) or abs(iqr) < 1e-8:
                iqr = 1.0
            meds.append(med); iqrs.append(iqr)
        self.median_ = np.asarray(meds, dtype=np.float32)
        self.iqr_    = np.asarray(iqrs, dtype=np.float32)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        z = (x.astype(np.float32) - self.median_) / self.iqr_
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(z, -CLIP_Z, CLIP_Z).astype(np.float32)


class NormalWindows(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        starts = np.arange(0, len(y) - SEQ_LEN + 1, STRIDE, dtype=np.int64)
        # Vectorised normal-window check
        y8      = (y > 0).astype(np.int8)
        cum     = np.zeros(len(y8) + 1, dtype=np.int32)
        np.cumsum(y8, out=cum[1:])
        any_anom = (cum[starts + SEQ_LEN] - cum[starts]) > 0
        normal   = starts[~any_anom]
        if len(normal) == 0:
            log("WARNING: no normal windows; using all windows")
            normal = starts
        if MAX_TRAIN_WINDOWS and len(normal) > MAX_TRAIN_WINDOWS:
            rng    = np.random.default_rng(SEED)
            normal = np.sort(rng.choice(normal, size=MAX_TRAIN_WINDOWS, replace=False))
        self.x = x.astype(np.float32); self.starts = normal

    def __len__(self): return len(self.starts)

    def __getitem__(self, i):
        s = int(self.starts[i])
        return torch.from_numpy(self.x[s:s+SEQ_LEN].T[:, :, None])


class AllWindows(Dataset):
    def __init__(self, x: np.ndarray):
        self.x      = x.astype(np.float32)
        self.starts = np.arange(0, len(x) - SEQ_LEN + 1, SCORE_STRIDE, dtype=np.int64)

    def __len__(self): return len(self.starts)

    def __getitem__(self, i):
        s = int(self.starts[i])
        return torch.from_numpy(self.x[s:s+SEQ_LEN].T[:, :, None])


class ForecastWindows(Dataset):
    """
    Sliding-window dataset for MLP training and scoring.
    context: x[i : i + SEQ_LEN]
    target:  x[i + SEQ_LEN : i + SEQ_LEN + MLP_PRED_LEN]
    Score is assigned to the *predicted* positions (future timesteps).
    """
    def __init__(self, x: np.ndarray, stride: int = SCORE_STRIDE,
                 normal_only: bool = False, y: np.ndarray = None):
        self.x        = x.astype(np.float32)
        self.pred_len = MLP_PRED_LEN
        total         = len(x) - SEQ_LEN - MLP_PRED_LEN + 1
        starts        = np.arange(0, total, stride, dtype=np.int64)

        if normal_only and y is not None:
            y8  = (y > 0).astype(np.int8)
            cum = np.zeros(len(y8) + 1, dtype=np.int32)
            np.cumsum(y8, out=cum[1:])
            # Window is normal if neither context nor target contain anomaly
            win_end  = starts + SEQ_LEN + MLP_PRED_LEN
            win_end  = np.minimum(win_end, len(y8))
            any_anom = (cum[win_end] - cum[starts]) > 0
            starts   = starts[~any_anom]
            if len(starts) == 0:
                starts = np.arange(0, total, stride, dtype=np.int64)
            if MAX_TRAIN_WINDOWS and len(starts) > MAX_TRAIN_WINDOWS:
                rng    = np.random.default_rng(SEED)
                starts = np.sort(rng.choice(starts, size=MAX_TRAIN_WINDOWS, replace=False))

        self.starts = starts

    def __len__(self):  return len(self.starts)

    def __getitem__(self, i):
        s   = int(self.starts[i])
        ctx = self.x[s : s + SEQ_LEN]
        tgt = self.x[s + SEQ_LEN : s + SEQ_LEN + MLP_PRED_LEN]
        # Return same layout as AE: (C, T, 1)
        ctx_t = torch.from_numpy(ctx.T[:, :, None])   # (C, SEQ_LEN, 1)
        tgt_t = torch.from_numpy(tgt.T[:, :, None])   # (C, MLP_PRED_LEN, 1)
        return ctx_t, tgt_t
