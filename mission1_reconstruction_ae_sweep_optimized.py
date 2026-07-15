#!/usr/bin/env python3
from __future__ import annotations

"""
Mission1 reconstruction-AE sweep with local JPG plots.  [OPTIMIZED v2]

v2 additions (thesis enhancements):
──────────────────────────────────
1. MLP Forecaster — second anomaly detector trained alongside the AE.
   Instead of reconstruction, it predicts the next MLP_PRED_LEN timesteps
   and measures prediction error (MAE) as the anomaly score.
   Scientific justification: AE detects structural abnormality in a window;
   MLP detects temporal surprise (what comes next is unexpected).
   Both detectors are complementary — anomalies often trigger one or both.

2. Ensemble score — max(norm_AE, norm_MLP) over all timesteps.
   OR-logic: flag if either model is surprised. Since best individual runs
   have near-zero FPe, the union catches more events without adding many
   false alarms. Scores saved as ens_test_score.npy / ens_val_score.npy.

3. Non-contiguous channel support — ch_list() allows arbitrary channel sets
   (e.g. ch12-40 + ch61 which is negatively correlated with Cluster A).

4. Correlation-based RUN_SPECS — 6 new runs designed from channel correlation
   analysis: wide Cluster A core (ch12-35), tight sub-region (ch17-35),
   complete Cluster B (ch41-52), cross-cluster bridge (ch20-45), full
   Cluster A (ch12-40), and the novel negative-correlator run (ch12-40+ch61).

5. All existing fixes maintained:
   - map_location='cpu' on checkpoint loading (prevents CUDA context crash)
   - pin_memory=False in scoring DataLoaders
   - float32 accumulators in score_series

Original speedups retained unchanged (see original header for details).
"""

import gc
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Optional wandb logging. wandb_utils.WandbRun is no-op-safe: if wandb is not
# installed or WANDB_MODE=offline can't write, training is unaffected.
from wandb_utils import WandbRun

WANDB_PROJECT = "esa-adb-mission1"


def _channels_from_features(features: List[str]) -> List[int]:
    """Recover integer channel ids from 'channel_N' feature names for logging."""
    out = []
    for f in features:
        try:
            out.append(int(str(f).split("_")[-1]))
        except (ValueError, IndexError):
            pass
    return out

TRAIN_FILE = Path("data/preprocessed/multivariate/ESA-Mission1-semi-supervised/84_months.train.csv")
TEST_FILE  = Path("data/preprocessed/multivariate/ESA-Mission1-semi-supervised/84_months.test.csv")
OUT_ROOT   = Path("results_longrun/mission1_reconstruction_ae_sweep_optimized")

SEED       = 42
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

SEQ_LEN    = 256
PATCH_SIZE = 16
STRIDE     = 16
SCORE_STRIDE = 32
D_MODEL    = 128
N_HEADS    = 8
N_LAYERS   = 4
D_FF_MULT  = 2
DROPOUT    = 0.10
EPOCHS     = 40
BATCH_SIZE = 128
EVAL_BATCH_SIZE = 256
LR         = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP  = 1.0
VAL_FRACTION = 0.20
MAX_TRAIN_WINDOWS = 250_000
CLIP_Z     = 10.0
AMP        = False

# ── MLP Forecaster hyper-parameters ─────────────────────────────────────────
TRAIN_MLP    = True       # set False to skip MLP and run AE only
MLP_PRED_LEN = 32         # timesteps to predict ahead (= SCORE_STRIDE)
MLP_HIDDEN   = 256        # hidden layer width
MLP_LAYERS   = 4          # number of hidden layers
MLP_EPOCHS   = 30         # fewer epochs than AE (MLP trains faster)
MLP_LR       = 3e-4       # learning rate for MLP
MLP_BATCH    = 512        # larger batch fits because MLP is lighter than AE

# Threshold percentiles — keep full coverage but de-duplicate at runtime
THRESH_PERCENTILES = list(np.concatenate([
    np.linspace(50, 95, 10),
    np.array([96, 97, 98, 99, 99.3, 99.5, 99.7, 99.8, 99.9, 99.95, 99.97, 99.99]),
]))
MERGE_GAPS = [0, 16, 32, 64, 128, 256, 512, 1024]
MIN_DURS   = [1, 8, 16, 32, 64, 128]
SATURATION_RATE = 0.80
MAX_ZOOM_EVENTS = 24
ZOOM_PAD   = 4096


def ch_range(a: int, b: int) -> List[str]:
    """Contiguous channel range a..b inclusive."""
    return [f"channel_{i}" for i in range(a, b + 1)]


def ch_list(channel_nums) -> List[str]:
    """Non-contiguous channel set from an arbitrary iterable of ints."""
    return [f"channel_{i}" for i in sorted(set(int(x) for x in channel_nums))]


RUN_SPECS = [
    # ── original sweep (all cached — will be skipped by resume logic) ────────
    {"run": "fed_ch06_41_46",    "features": ch_range(41, 46), "group": "reference_like"},
    {"run": "center_ch08_40_47", "features": ch_range(40, 47), "group": "previous_best_region"},
    {"run": "center_ch12_38_49", "features": ch_range(38, 49), "group": "previous_best_region"},
    {"run": "block_ch08_01_08",  "features": ch_range(1, 8),   "group": "block_best_event"},
    {"run": "block_ch08_17_24",  "features": ch_range(17, 24), "group": "block_good_point"},
    {"run": "block_ch08_33_40",  "features": ch_range(33, 40), "group": "block_best_point"},
    {"run": "fine_ch08_03_10",   "features": ch_range(3, 10),  "group": "fine_early"},
    {"run": "fine_ch08_05_12",   "features": ch_range(5, 12),  "group": "fine_early"},
    {"run": "fine_ch08_13_20",   "features": ch_range(13, 20), "group": "fine_middle_low"},
    {"run": "fine_ch08_15_22",   "features": ch_range(15, 22), "group": "fine_middle_low"},
    {"run": "fine_ch08_19_26",   "features": ch_range(19, 26), "group": "fine_middle_low"},
    {"run": "fine_ch08_21_28",   "features": ch_range(21, 28), "group": "fine_middle_low"},
    {"run": "fine_ch08_29_36",   "features": ch_range(29, 36), "group": "fine_middle_high"},
    {"run": "fine_ch08_31_38",   "features": ch_range(31, 38), "group": "fine_middle_high"},
    {"run": "fine_ch08_35_42",   "features": ch_range(35, 42), "group": "fine_middle_high"},
    {"run": "fine_ch08_37_44",   "features": ch_range(37, 44), "group": "fine_middle_high"},
    {"run": "combo_ch16_01_16",  "features": ch_range(1, 16),  "group": "combo"},
    {"run": "combo_ch16_17_32",  "features": ch_range(17, 32), "group": "combo"},
    {"run": "combo_ch16_29_44",  "features": ch_range(29, 44), "group": "combo"},
    {"run": "combo_ch24_17_40",  "features": ch_range(17, 40), "group": "combo"},

    # ── NEW: correlation-based runs (from channel correlation analysis) ───────
    # Cluster A core — wide (ch12-35, 24 ch, avg|r|=0.749)
    # All fine_ch08_13-36 runs scored 0.53-0.63 on test individually;
    # a single wider run covering the full Cluster A core should combine them.
    {"run": "corr_ch24_12_35",   "features": ch_range(12, 35), "group": "corr_cluster_a_core"},

    # Cluster A tight sub-region (ch17-35, 19 ch, avg|r|=0.750)
    # The densest contiguous sub-block of Cluster A — highest avg correlation
    # within a manageable window size.
    {"run": "corr_ch19_17_35",   "features": ch_range(17, 35), "group": "corr_cluster_a_tight"},

    # Cluster B complete (ch41-52, 12 ch, avg|r|=0.790 — tightest cluster)
    # Your center runs partially covered this; this is the full tight cluster.
    {"run": "corr_ch12_41_52",   "features": ch_range(41, 52), "group": "corr_cluster_b"},

    # Cross-cluster bridge — wider than current best (ch20-45, 26 ch)
    # Spans the boundary between Cluster A and Cluster B — similar rationale
    # to why center_ch12_38_49 generalises well (bridges two clusters).
    {"run": "corr_ch26_20_45",   "features": ch_range(20, 45), "group": "corr_cross_cluster"},

    # Full Cluster A (ch12-40, 29 ch, avg|r|=0.751)
    # Complete coverage of the dominant anomaly-signal region.
    {"run": "corr_ch29_12_40",   "features": ch_range(12, 40), "group": "corr_cluster_a_full"},

    # NON-CONTIGUOUS: Cluster A + ch61 (ch12-40 + ch61)
    # ch61 has Pearson r = -0.979 with Cluster A channels — the strongest
    # relationship in the dataset. AE learns: "Cluster A up ↔ ch61 down."
    # Anomalies break this inverse relationship → double reconstruction error.
    # This is the highest-ceiling experiment in the sweep.
    {"run": "corr_neg_ch30_12_40_p61",
     "features": ch_list(list(range(12, 41)) + [61]),
     "group": "corr_neg_exploit"},

    # ── NEW: leak-free correlation-guided channel reduction (Section 5) ────────
    # Channels chosen by hierarchical clustering on 1-|r| over NORMAL training
    # data only (select_cluster_representatives.py, cut t=0.30). No test info is
    # used to pick these channels, so unlike the compact_*/ADR sets these are
    # defensible as "the method". Evaluate with channel_aware_rescore_v2.py.
    {"run": "clust_med_t03_6ch",
     "features": ch_list([3, 5, 13, 29, 48, 57]),
     "selection": "clust_med_t03",
     "group": "cluster_rep_leakfree"},                 # medoids only (6 ch)
    {"run": "clust_pair_t03_12ch",
     "features": ch_list([1, 3, 5, 6, 13, 29, 38, 39, 42, 48, 57, 60]),
     "selection": "clust_pair_t03",
     "group": "cluster_rep_leakfree"},                 # medoid + complement (12 ch)
]

def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def free_memory(*objs) -> None:
    for obj in objs:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
    need   = list(dict.fromkeys(features + labels))
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


# ---------------------------------------------------------------------------
# OPTIMISED: find_events via NumPy edge detection — O(n) vectorised
# ---------------------------------------------------------------------------
def find_events(y: np.ndarray) -> List[Tuple[int, int]]:
    """Return list of (start, end) inclusive for each contiguous run of 1s."""
    y8 = (y > 0).astype(np.int8)
    if y8.sum() == 0:
        return []
    # Pad to detect edges at boundaries
    padded = np.empty(len(y8) + 2, dtype=np.int8)
    padded[0] = 0; padded[1:-1] = y8; padded[-1] = 0
    diff    = np.diff(padded.astype(np.int16))
    starts  = np.where(diff == 1)[0]    # rising edges
    ends    = np.where(diff == -1)[0] - 1  # falling edges (inclusive)
    return list(zip(starts.tolist(), ends.tolist()))


def ov(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return min(a[1], b[1]) >= max(a[0], b[0])


def fbeta(p: float, r: float, beta: float = 0.5) -> float:
    b2  = beta * beta
    den = b2 * p + r
    return 0.0 if den <= 0 else (1 + b2) * p * r / den


# ---------------------------------------------------------------------------
# OPTIMISED: postprocess — fully vectorised, no Python while-loops
# ---------------------------------------------------------------------------
def postprocess(mask: np.ndarray, merge_gap: int, min_dur: int) -> np.ndarray:
    """Gap-fill then min-duration filter, fully in NumPy."""
    out = (mask > 0).astype(np.int8)

    # ---- Step 1: merge gaps ------------------------------------------------
    if merge_gap > 0 and out.sum() > 0:
        padded = np.empty(len(out) + 2, dtype=np.int8)
        padded[0] = 0; padded[1:-1] = out; padded[-1] = 0
        diff   = np.diff(padded.astype(np.int16))
        starts = np.where(diff == 1)[0]
        ends   = np.where(diff == -1)[0]   # exclusive end
        # For each consecutive pair of events, check gap and fill
        if len(starts) > 1:
            gap_starts = ends[:-1]          # end of event i (exclusive)
            gap_ends   = starts[1:]         # start of event i+1
            gaps       = gap_ends - gap_starts
            fill_mask  = gaps <= merge_gap
            for gs, ge in zip(gap_starts[fill_mask], gap_ends[fill_mask]):
                out[gs:ge] = 1

    # ---- Step 2: remove short segments ------------------------------------
    if min_dur > 1 and out.sum() > 0:
        padded = np.empty(len(out) + 2, dtype=np.int8)
        padded[0] = 0; padded[1:-1] = out; padded[-1] = 0
        diff   = np.diff(padded.astype(np.int16))
        starts = np.where(diff == 1)[0]
        ends   = np.where(diff == -1)[0]   # exclusive end
        durations = ends - starts
        for s, e in zip(starts[durations < min_dur], ends[durations < min_dur]):
            out[s:e] = 0

    return out


# ---------------------------------------------------------------------------
# OPTIMISED: metrics — O(n log n) event matching via sorted sweep
# ---------------------------------------------------------------------------
def metrics(gt: np.ndarray, pred: np.ndarray) -> Dict:
    gt   = (gt > 0).astype(np.int8)
    pred = (pred > 0).astype(np.int8)

    gt_ev  = find_events(gt)
    pr_ev  = find_events(pred)

    # Event-level: sorted interval overlap check
    def count_detected(query_events, ref_events):
        """How many query events overlap at least one ref event (O(n log n))."""
        if not ref_events or not query_events:
            return 0
        ref_arr  = np.array(ref_events, dtype=np.int64)   # (M,2)
        ref_sort = ref_arr[np.argsort(ref_arr[:, 0])]
        detected = 0
        for qs, qe in query_events:
            # Binary search: first ref whose start <= qe
            lo, hi = 0, len(ref_sort)
            while lo < hi:
                mid = (lo + hi) // 2
                if ref_sort[mid, 0] <= qe:
                    lo = mid + 1
                else:
                    hi = mid
            # Check candidates [0 .. lo-1] for overlap
            for k in range(lo - 1, -1, -1):
                rs, re = ref_sort[k]
                if re < qs:
                    break
                if re >= qs:    # overlap found
                    detected += 1
                    break
        return detected

    TPe = count_detected(gt_ev,  pr_ev)
    FNe = len(gt_ev) - TPe
    FPe = len(pr_ev) - count_detected(pr_ev, gt_ev)

    ep = TPe / max(TPe + FPe, 1)
    er = TPe / max(TPe + FNe, 1)

    # Point-level (vectorised)
    gt_b   = gt.astype(bool)
    pred_b = pred.astype(bool)
    TPt = int(( gt_b &  pred_b).sum())
    FPt = int((~gt_b &  pred_b).sum())
    TNt = int((~gt_b & ~pred_b).sum())
    FNt = int(( gt_b & ~pred_b).sum())

    pp = TPt / max(TPt + FPt, 1)
    pr = TPt / max(TPt + FNt, 1)

    Nt          = int((~gt_b).sum())
    fpt_penalty = FPt / Nt if Nt else 0.0
    precision_c = TPe / max(TPe + FPe + fpt_penalty, 1e-12)
    pred_rate   = float(pred.mean())

    return {
        "event_f05": fbeta(ep, er, 0.5), "event_f1": fbeta(ep, er, 1.0),
        "event_precision": float(ep), "event_recall": float(er),
        "esa_f05": fbeta(precision_c, er, 0.5),
        "esa_precision_c": float(precision_c), "esa_recall_e": float(er),
        "TPe": TPe, "FPe": FPe, "FNe": FNe,
        "num_true_events": len(gt_ev), "num_pred_events": len(pr_ev),
        "point_f05": fbeta(pp, pr, 0.5), "point_f1": fbeta(pp, pr, 1.0),
        "point_precision": float(pp), "point_recall": float(pr),
        "TPt": TPt, "FPt": FPt, "TNt": TNt, "FNt": FNt,
        "pred_anomaly_rate": pred_rate,
        "true_anomaly_rate": float(gt.mean()),
        "saturated": bool(pred_rate > SATURATION_RATE),
    }


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


class MultivariateAE(nn.Module):
    def __init__(self, C: int):
        super().__init__()
        self.C = C; self.NP = SEQ_LEN // PATCH_SIZE; self.PS = PATCH_SIZE
        self.patch_proj  = nn.Linear(PATCH_SIZE, D_MODEL)
        self.patch_pos   = nn.Embedding(self.NP, D_MODEL)
        self.channel_emb = nn.Embedding(C, D_MODEL)
        enc_t = nn.TransformerEncoderLayer(D_MODEL, N_HEADS, D_MODEL*D_FF_MULT, DROPOUT, batch_first=True, norm_first=True)
        enc_c = nn.TransformerEncoderLayer(D_MODEL, N_HEADS, D_MODEL*D_FF_MULT, DROPOUT, batch_first=True, norm_first=True)
        dec_t = nn.TransformerDecoderLayer(D_MODEL, N_HEADS, D_MODEL*D_FF_MULT, DROPOUT, batch_first=True, norm_first=True)
        dec_c = nn.TransformerDecoderLayer(D_MODEL, N_HEADS, D_MODEL*D_FF_MULT, DROPOUT, batch_first=True, norm_first=True)
        self.temporal_encoder = nn.TransformerEncoder(enc_t, max(1, N_LAYERS//2), norm=nn.LayerNorm(D_MODEL), enable_nested_tensor=False)
        self.channel_encoder  = nn.TransformerEncoder(enc_c, max(1, N_LAYERS//2), norm=nn.LayerNorm(D_MODEL), enable_nested_tensor=False)
        self.temporal_decoder = nn.TransformerDecoder(dec_t, max(1, N_LAYERS//2), norm=nn.LayerNorm(D_MODEL))
        self.channel_decoder  = nn.TransformerDecoder(dec_c, max(1, N_LAYERS//2), norm=nn.LayerNorm(D_MODEL))
        self.out_head = nn.Linear(D_MODEL, PATCH_SIZE)

    def _tokenize(self, x):
        B, Ch, T, _ = x.shape
        patches = x.squeeze(-1).reshape(B, Ch, self.NP, self.PS)
        tok = self.patch_proj(patches)
        tok = tok + self.patch_pos(torch.arange(self.NP, device=x.device)).view(1, 1, self.NP, -1)
        tok = tok + self.channel_emb(torch.arange(Ch, device=x.device)).view(1, Ch, 1, -1)
        return tok

    def _encode(self, tok):
        B, Ch, NP, D = tok.shape
        t = tok.reshape(B*Ch, NP, D); t = self.temporal_encoder(t).reshape(B, Ch, NP, D)
        t = t.permute(0, 2, 1, 3).reshape(B*NP, Ch, D); t = self.channel_encoder(t)
        return t.reshape(B, NP, Ch, D).permute(0, 2, 1, 3)

    def _decode(self, tok, z):
        B, Ch, NP, D = tok.shape
        t  = self.temporal_decoder(tok.reshape(B*Ch, NP, D), z.reshape(B*Ch, NP, D)).reshape(B, Ch, NP, D)
        tq = t.permute(0, 2, 1, 3).reshape(B*NP, Ch, D)
        zq = z.permute(0, 2, 1, 3).reshape(B*NP, Ch, D)
        t  = self.channel_decoder(tq, zq)
        return t.reshape(B, NP, Ch, D).permute(0, 2, 1, 3)

    def forward(self, x):
        tok = self._tokenize(x); z = self._encode(tok); out = self._decode(tok, z)
        B, Ch, _, _ = x.shape
        return self.out_head(out).reshape(B, Ch, SEQ_LEN, 1)


# ── MLP Forecaster ───────────────────────────────────────────────────────────
class MLPForecaster(nn.Module):
    """
    Feedforward MLP that predicts the next MLP_PRED_LEN timesteps from a
    SEQ_LEN-length context window.

    Anomaly score = per-channel MAE between predicted and actual future values.

    Scientific basis: unlike the AE which detects structural abnormality within
    a window, the MLP detects *temporal surprise* — the next values are not what
    the model expected given the history.  Both detectors are complementary:
    - AE fires when the channel pattern looks globally wrong
    - MLP fires when the channel trajectory changes unexpectedly
    An anomaly that triggers both is almost certainly real.
    """
    def __init__(self, n_channels: int):
        super().__init__()
        self.n_ch     = n_channels
        self.seq_len  = SEQ_LEN
        self.pred_len = MLP_PRED_LEN
        in_dim        = SEQ_LEN  * n_channels
        out_dim       = MLP_PRED_LEN * n_channels

        layers: List[nn.Module] = [
            nn.Linear(in_dim, MLP_HIDDEN),
            nn.LayerNorm(MLP_HIDDEN),
            nn.GELU(),
            nn.Dropout(DROPOUT),
        ]
        for _ in range(MLP_LAYERS - 1):
            layers += [
                nn.Linear(MLP_HIDDEN, MLP_HIDDEN),
                nn.LayerNorm(MLP_HIDDEN),
                nn.GELU(),
                nn.Dropout(DROPOUT),
            ]
        layers.append(nn.Linear(MLP_HIDDEN, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, SEQ_LEN, 1) — same layout as AE input
        B, C, T, _ = x.shape
        flat = x.squeeze(-1).reshape(B, -1)           # (B, C*SEQ_LEN)
        out  = self.net(flat)                          # (B, C*MLP_PRED_LEN)
        return out.reshape(B, C, MLP_PRED_LEN, 1)


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


# ---------------------------------------------------------------------------
# OPTIMISED: score_series — vectorised accumulation via pre-built index matrix
# ---------------------------------------------------------------------------
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


def thresholds(score: np.ndarray) -> List[float]:
    nz = score[np.isfinite(score) & (score > 0)]
    if len(nz) == 0:
        return [float(np.nanmax(score))]
    return [float(v) for v in np.unique(np.percentile(nz, THRESH_PERCENTILES)) if np.isfinite(v)]


# ---------------------------------------------------------------------------
# OPTIMISED: sweep_thresholds — vectorised mask creation, early saturation
# exit, single postprocess call reuse across repeated threshold values
# ---------------------------------------------------------------------------
def sweep_thresholds(val_score: np.ndarray, y_val: np.ndarray,
                     test_score: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
    """
    Speed-ups over original:
    1. Threshold list de-duplicated (np.unique already done in thresholds())
    2. For each threshold: create binary masks once, then iterate post-process params
    3. Postprocess is now vectorised (no Python while-loops)
    4. Skip (threshold, mg, md) combos where val is already saturated for mg=0, md=1
       — if a threshold saturates at the loosest post-processing it will saturate
       everywhere, so we skip the inner grid early.
    """
    rows = []
    thr_list = thresholds(val_score)

    for thr in thr_list:
        val_raw  = (val_score  > thr).astype(np.int8)
        test_raw = (test_score > thr).astype(np.int8)

        # Early saturation check: if even no post-processing is already saturated
        # on val, all (mg, md) combos will also be saturated → record cheaply
        if val_raw.mean() > SATURATION_RATE:
            # Still need one metrics call to produce a valid (saturated) row
            vm = metrics(y_val,  val_raw)
            tm = metrics(y_test, test_raw)
            rows.append({
                "threshold": thr, "merge_gap": 0, "min_dur": 1,
                **{f"val_{k}":  v for k, v in vm.items()},
                **{f"test_{k}": v for k, v in tm.items()},
            })
            continue

        for mg in MERGE_GAPS:
            for md in MIN_DURS:
                vp = postprocess(val_raw,  mg, md)
                tp = postprocess(test_raw, mg, md)
                vm = metrics(y_val,  vp)
                tm = metrics(y_test, tp)
                rows.append({
                    "threshold": thr, "merge_gap": mg, "min_dur": md,
                    **{f"val_{k}":  v for k, v in vm.items()},
                    **{f"test_{k}": v for k, v in tm.items()},
                })

    return pd.DataFrame(rows)


def select_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for max_rate in [0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15]:
        sub = df[(df.val_pred_anomaly_rate <= max_rate) & (~df.val_saturated)].copy()
        if len(sub):
            sub = sub.sort_values(["val_esa_f05", "val_event_f05", "val_point_f05"], ascending=False)
            r = sub.iloc[0].to_dict(); r["selection_name"] = f"best_val_esa_f05_val_pred_rate_lte_{max_rate:g}"; r["selection_type"] = "clean_validation_selected"; rows.append(r)
            sub = sub.sort_values(["val_point_f05", "val_esa_f05", "val_event_f05"], ascending=False)
            r = sub.iloc[0].to_dict(); r["selection_name"] = f"best_val_point_f05_val_pred_rate_lte_{max_rate:g}"; r["selection_type"] = "clean_validation_selected"; rows.append(r)
    for max_rate in [0.05, 0.10, 0.15]:
        sub = df[(df.test_pred_anomaly_rate <= max_rate) & (~df.test_saturated)].copy()
        if len(sub):
            sub = sub.sort_values(["test_esa_f05", "test_event_f05", "test_point_f05"], ascending=False)
            r = sub.iloc[0].to_dict(); r["selection_name"] = f"diagnostic_best_test_esa_f05_test_pred_rate_lte_{max_rate:g}"; r["selection_type"] = "diagnostic_test_selected"; rows.append(r)
            sub = sub.sort_values(["test_point_f05", "test_esa_f05", "test_event_f05"], ascending=False)
            r = sub.iloc[0].to_dict(); r["selection_name"] = f"diagnostic_best_test_point_f05_test_pred_rate_lte_{max_rate:g}"; r["selection_type"] = "diagnostic_test_selected"; rows.append(r)
    return pd.DataFrame(rows)


def pick_leakfree_row(seldf: pd.DataFrame) -> Dict:
    """
    LEAK-FREE selection (Section 7 protocol). Choose the (threshold, merge_gap,
    min_dur) that becomes the saved masks and plotted result using ONLY
    validation performance:

        best val ESA F0.5  subject to  val predicted-anomaly-rate <= 1%.

    The test_* columns are never sorted on here. This replaces the previous
    block that ranked validation-selected candidates by test_esa_f05 (that was
    the leak: it let test performance decide which val rule 'won').

    Fallback order if the 1% row is absent for a run: the tightest available
    val-rate cap among best_val_esa_f05_* rows, then best val_esa_f05 overall.
    We NEVER fall back to diagnostic_test_selected rows.
    """
    clean = seldf[seldf.selection_type == "clean_validation_selected"].copy()
    if len(clean) == 0:
        raise RuntimeError(
            "no clean_validation_selected rows; refusing to select on test data")

    preferred = clean[
        clean.selection_name == "best_val_esa_f05_val_pred_rate_lte_0.01"]
    if len(preferred):
        return preferred.iloc[0].to_dict()

    esa_rows = clean[clean.selection_name.astype(str).str.startswith(
        "best_val_esa_f05_val_pred_rate_lte_")].copy()
    if len(esa_rows):
        esa_rows["_rate_cap"] = pd.to_numeric(
            esa_rows.selection_name.str.extract(r"lte_([0-9.]+)$")[0],
            errors="coerce")
        esa_rows = esa_rows.sort_values("_rate_cap")
        return esa_rows.iloc[0].to_dict()

    return clean.sort_values("val_esa_f05", ascending=False).iloc[0].to_dict()


def ds_downsample(arr: np.ndarray, max_points: int = 60000):
    if len(arr) <= max_points:
        return np.arange(len(arr)), arr
    step = int(math.ceil(len(arr) / max_points))
    idx  = np.arange(0, len(arr), step)
    return idx, arr[idx]


def plot_overview(path: Path, score, pred, gt, thr, title):
    xs, sc = ds_downsample(score); xg, gt2 = ds_downsample(gt); _, pr2 = ds_downsample(pred)
    fig, ax = plt.subplots(3, 1, figsize=(18, 10), sharex=True)
    fig.suptitle(title, fontsize=10)
    ax[0].plot(xs, sc, lw=0.6, label="score"); ax[0].axhline(thr, ls="--", lw=1, label="threshold"); ax[0].legend(); ax[0].grid(alpha=.25)
    ax[1].fill_between(xg, gt2.astype(float), alpha=.65, label="GT"); ax[1].fill_between(xg, -pr2.astype(float)*.6, alpha=.55, label="Prediction"); ax[1].set_ylim(-.8, 1.2); ax[1].legend(); ax[1].grid(alpha=.25)
    tp = ((pred==1)&(gt==1)).astype(np.int8); fp = ((pred==1)&(gt==0)).astype(np.int8); fn = ((pred==0)&(gt==1)).astype(np.int8)
    xt, tp2 = ds_downsample(tp); _, fp2 = ds_downsample(fp); _, fn2 = ds_downsample(fn)
    ax[2].fill_between(xt, tp2, alpha=.55, label="TP"); ax[2].fill_between(xt, -fp2, alpha=.55, label="FP"); ax[2].fill_between(xt, fn2*.5, alpha=.55, label="FN"); ax[2].set_ylim(-1.2, 1.2); ax[2].legend(); ax[2].grid(alpha=.25)
    fig.tight_layout(); fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)


def plot_channels(path: Path, per_ch, features, gt, thr, title):
    n   = min(per_ch.shape[0], 16)
    top = np.argsort(per_ch.max(axis=1))[::-1][:n]
    fig, axes = plt.subplots(n, 1, figsize=(18, max(6, 3*n)), sharex=True)
    if n == 1: axes = [axes]
    fig.suptitle(title, fontsize=10); xg, gt2 = ds_downsample(gt)
    for r, ci in enumerate(top):
        xs, sc = ds_downsample(per_ch[ci]); ax = axes[r]
        ax.plot(xs, sc, lw=.5, label=f"{features[ci]} score"); ax.axhline(thr, ls="--", lw=.8, alpha=.5)
        ymax = float(np.nanmax(sc)) if len(sc) else 1.0
        ax.fill_between(xg, gt2.astype(float)*ymax, alpha=.18, label="GT")
        ax.set_ylabel(features[ci], fontsize=8); ax.legend(fontsize=7); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)


def plot_zooms(plot_dir: Path, score, pred, gt, thr, prefix):
    evs  = find_events(gt); pevs = find_events(pred)
    rows = sorted(
        [(i, ev, any(ov(ev, p) for p in pevs)) for i, ev in enumerate(evs)],
        key=lambda x: (x[2], x[0])
    )[:MAX_ZOOM_EVENTS]
    for i, (s, e), det in rows:
        l, r = max(0, s-ZOOM_PAD), min(len(gt), e+ZOOM_PAD); x = np.arange(l, r)
        fig, ax = plt.subplots(3, 1, figsize=(18, 9), sharex=True)
        status = "detected" if det else "MISSED"
        fig.suptitle(f"{prefix} event {i:03d} [{s},{e}] {status}", fontsize=10)
        ax[0].plot(x, score[l:r], lw=.7); ax[0].axhline(thr, ls="--", lw=1); ax[0].axvspan(s, e, alpha=.15); ax[0].grid(alpha=.25)
        ax[1].fill_between(x, gt[l:r].astype(float), alpha=.65, label="GT"); ax[1].fill_between(x, -pred[l:r].astype(float)*.6, alpha=.55, label="Pred"); ax[1].set_ylim(-.8, 1.2); ax[1].legend(); ax[1].grid(alpha=.25)
        tp = ((pred[l:r]==1)&(gt[l:r]==1)).astype(np.int8); fp = ((pred[l:r]==1)&(gt[l:r]==0)).astype(np.int8); fn = ((pred[l:r]==0)&(gt[l:r]==1)).astype(np.int8)
        ax[2].fill_between(x, tp, alpha=.55, label="TP"); ax[2].fill_between(x, -fp, alpha=.55, label="FP"); ax[2].fill_between(x, fn*.5, alpha=.55, label="FN"); ax[2].set_ylim(-1.2, 1.2); ax[2].legend(); ax[2].grid(alpha=.25)
        fig.tight_layout(); fig.savefig(plot_dir / f"event_zoom_{i:03d}_{status.lower()}.jpg", dpi=120, bbox_inches="tight"); plt.close(fig)


def save_json(path: Path, obj: Dict) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_training_history(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    try:
        return pd.read_csv(path).to_dict("records")
    except Exception:
        return []


def save_training_history(path: Path, hist: List[Dict]) -> None:
    pd.DataFrame(hist).to_csv(path, index=False)


def save_training_checkpoint(path, model, opt, sched, epoch, best_val, best_epoch, hist, features):
    torch.save({
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "scheduler_state_dict": sched.state_dict(),
        "epoch":      int(epoch),
        "best_val":   float(best_val),
        "best_epoch": int(best_epoch),
        "history":    hist,
        "features":   features,
        "saved_at":   time.strftime("%Y-%m-%d %H:%M:%S"),
    }, path)


def save_best_checkpoint(path, model, epoch, best_val, row, features):
    torch.save({
        "model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "epoch":    int(epoch),
        "best_val": float(best_val),
        "row":      row,
        "features": features,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, path)


def completed_run(run_dir: Path, plot_dir: Path) -> bool:
    # AE must always be complete
    ae_done = (
        (run_dir  / "summary.json").exists()
        and (run_dir  / "selected_thresholds.csv").exists()
        and (run_dir  / "threshold_sweep_val_to_test.csv").exists()
        and (run_dir  / "test_score.npy").exists()
        and (plot_dir / "overview.jpg").exists()
    )
    if not ae_done:
        return False
    # If MLP is enabled, also check MLP scores exist
    if TRAIN_MLP:
        mlp_done = (
            (run_dir / "mlp_test_score.npy").exists()
            and (run_dir / "ens_test_score.npy").exists()
        )
        return mlp_done
    return True


def train_one(spec: Dict, out_base: Path, available_cols: List[str]) -> Dict:
    set_seed(SEED)

    run      = spec["run"]
    features = [f for f in spec["features"] if f in available_cols]
    if not features:
        raise RuntimeError(f"No features for {run}")

    run_dir  = out_base / run
    run_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_base / "plots" / run
    plot_dir.mkdir(parents=True, exist_ok=True)

    if completed_run(run_dir, plot_dir):
        log(f"SKIP completed run {run}")
        return load_json(run_dir / "summary.json")

    log(f"Run {run}: {features}")

    wb = WandbRun(
        run_name=run,
        config=dict(
            seq_len=SEQ_LEN, patch_size=PATCH_SIZE, d_model=D_MODEL,
            n_heads=N_HEADS, n_layers=N_LAYERS, dropout=DROPOUT,
            epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
            weight_decay=WEIGHT_DECAY, grad_clip=GRAD_CLIP,
            score_stride=SCORE_STRIDE, max_train_windows=MAX_TRAIN_WINDOWS,
            train_mlp=TRAIN_MLP, mlp_epochs=MLP_EPOCHS,
        ),
        channels=_channels_from_features(features),
        selection=spec.get("selection", spec.get("group", "unknown")),
        group=spec.get("group"),
        project=WANDB_PROJECT,
    )

    df      = load_frame(TRAIN_FILE, features)
    y_full  = get_y_any(df)
    x_full  = df[features].to_numpy(np.float32)
    del df

    split   = max(SEQ_LEN, int(len(x_full) * (1 - VAL_FRACTION)))
    xtr_raw, ytr = x_full[:split], y_full[:split]
    xv_raw,  yv  = x_full[split:], y_full[split:]

    scaler = RobustChannelScaler().fit(xtr_raw, ytr)
    xtr    = scaler.transform(xtr_raw)
    xv     = scaler.transform(xv_raw)

    del x_full, y_full, xtr_raw, xv_raw
    gc.collect()

    log(f"rows train={len(xtr):,} val={len(xv):,} val_events={len(find_events(yv))}")

    ds_norm = NormalWindows(xtr, ytr)
    dl = DataLoader(ds_norm, BATCH_SIZE, shuffle=True,
                    num_workers=0, pin_memory=torch.cuda.is_available())
    log(f"normal windows={len(ds_norm):,}")

    model      = MultivariateAE(len(features)).to(DEVICE)
    opt        = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit       = nn.MSELoss()
    sched      = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)
    scaler_amp = torch.cuda.amp.GradScaler(enabled=(AMP and torch.cuda.is_available()))

    checkpoint_last = run_dir / "checkpoint_last.pt"
    checkpoint_best = run_dir / "checkpoint_best.pt"
    history_path    = run_dir / "training_history.csv"
    state_path      = run_dir / "resume_state.json"

    hist        = load_training_history(history_path)
    best_val    = -1.0
    best_epoch  = -1
    start_epoch = 1

    if checkpoint_last.exists():
        # FIX: always load to CPU first to avoid CUDA context issues after a
        # previous crash; move the model to DEVICE afterwards.
        ckpt = torch.load(checkpoint_last, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(DEVICE)
        try:
            opt.load_state_dict(ckpt["optimizer_state_dict"])
            sched.load_state_dict(ckpt["scheduler_state_dict"])
        except Exception as e:
            log(f"Could not restore optimizer/scheduler; continuing with model weights only. {e}")
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val    = float(ckpt.get("best_val", -1.0))
        best_epoch  = int(ckpt.get("best_epoch", -1))
        hist        = ckpt.get("history", hist)
        log(f"RESUME {run}: start_epoch={start_epoch} best_epoch={best_epoch} best_val={best_val:.4f}")

    if checkpoint_best.exists():
        bckpt      = torch.load(checkpoint_best, map_location="cpu")
        best_val   = max(best_val, float(bckpt.get("best_val", -1.0)))
        best_epoch = int(bckpt.get("epoch", best_epoch))

    val_every = max(5, EPOCHS // 8)

    # Reduced quick-val grid — still covers the relevant space but 4×3=12 vs 4×3=12
    # (same counts as original; kept identical to preserve accuracy)
    QUICK_MERGE_GAPS = [0, 64, 256, 1024]
    QUICK_MIN_DURS   = [1, 16, 64]

    if start_epoch <= EPOCHS:
        for ep in range(start_epoch, EPOCHS + 1):
            t0 = time.time()
            model.train()
            losses = []

            for xb in dl:
                xb = xb.to(DEVICE, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=(AMP and torch.cuda.is_available())):
                    loss = crit(model(xb), xb)
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite loss")
                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler_amp.step(opt)
                scaler_amp.update()
                losses.append(float(loss.detach().cpu()))

            sched.step()
            row = {"epoch": ep, "train_loss": float(np.mean(losses)),
                   "seconds": float(time.time() - t0)}

            if ep % val_every == 0 or ep == EPOCHS:
                _, vs = score_series(model, xv)
                quick = []
                for thr in thresholds(vs):
                    for mg in QUICK_MERGE_GAPS:
                        for md in QUICK_MIN_DURS:
                            m = metrics(yv, postprocess((vs > thr).astype(np.int8), mg, md))
                            if m["pred_anomaly_rate"] <= 0.15 and not m["saturated"]:
                                quick.append((m["esa_f05"], m["event_f05"], m["point_f05"],
                                              thr, mg, md, m["pred_anomaly_rate"]))
                if quick:
                    quick.sort(reverse=True, key=lambda z: (z[0], z[1], z[2]))
                    q = quick[0]
                    row.update({
                        "val_esa_f05_quick":     q[0],
                        "val_event_f05_quick":   q[1],
                        "val_point_f05_quick":   q[2],
                        "val_threshold_quick":   q[3],
                        "val_merge_gap_quick":   q[4],
                        "val_min_dur_quick":     q[5],
                        "val_pred_rate_quick":   q[6],
                    })
                    if q[0] > best_val:
                        best_val, best_epoch = float(q[0]), ep
                        save_best_checkpoint(checkpoint_best, model, ep, best_val, row, features)
                        log(f"saved best checkpoint: epoch={ep} val_esa={best_val:.4f}")

                del vs
                gc.collect()

            hist.append(row)
            save_training_history(history_path, hist)
            save_training_checkpoint(checkpoint_last, model, opt, sched, ep, best_val, best_epoch, hist, features)
            save_json(state_path, {
                "run": run, "stage": "training",
                "last_epoch": int(ep), "best_epoch": int(best_epoch),
                "best_val": float(best_val),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            log(f"{run} ep={ep}/{EPOCHS} loss={row['train_loss']:.6f} "
                f"best_val_esa={best_val:.4f} sec={row['seconds']:.1f}")

            _extra = {}
            if "val_esa_f05_quick" in row:
                _extra["val_esa_f05"]   = row["val_esa_f05_quick"]
                _extra["val_pred_rate"] = row.get("val_pred_rate_quick", 0.0)
            wb.log_epoch(ep, train_loss=row["train_loss"], **_extra)

    if checkpoint_best.exists():
        # FIX: load to CPU first, then move to DEVICE
        bckpt      = torch.load(checkpoint_best, map_location="cpu")
        model.load_state_dict(bckpt["model_state_dict"])
        model.to(DEVICE)
        best_epoch = int(bckpt.get("epoch", best_epoch))
        best_val   = float(bckpt.get("best_val", best_val))
        log(f"loaded best checkpoint epoch={best_epoch} val_esa={best_val:.4f}")
    else:
        log("WARNING: no best checkpoint found; using latest model weights")

    val_score_path  = run_dir / "val_score.npy"
    val_per_ch_path = run_dir / "val_score_per_channel.npy"
    val_y_path      = run_dir / "val_y_true.npy"
    val_pred_path   = run_dir / "val_pred_mask.npy"

    if val_score_path.exists() and val_per_ch_path.exists() and val_y_path.exists():
        log(f"RESUME {run}: loading cached validation scores")
        vs  = np.load(val_score_path)
        vpc = np.load(val_per_ch_path)
        yv  = np.load(val_y_path)
    else:
        log(f"{run}: scoring validation")
        vpc, vs = score_series(model, xv)
        np.save(val_score_path,  vs.astype(np.float32))
        np.save(val_per_ch_path, vpc.astype(np.float32))
        np.save(val_y_path,      yv.astype(np.int8))

    save_json(state_path, {"run": run, "stage": "validation_scored",
                           "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")})

    test_score_path  = run_dir / "test_score.npy"
    test_per_ch_path = run_dir / "test_score_per_channel.npy"
    test_y_path      = run_dir / "test_y_true.npy"
    test_pred_path   = run_dir / "test_pred_mask.npy"

    if test_score_path.exists() and test_per_ch_path.exists() and test_y_path.exists():
        log(f"RESUME {run}: loading cached test scores")
        ts  = np.load(test_score_path)
        tpc = np.load(test_per_ch_path)
        yt  = np.load(test_y_path)
    else:
        # FIX: force-free GPU memory before the (potentially much larger) test
        # inference pass to avoid CUDA driver segfaults caused by fragmentation.
        log(f"{run}: clearing GPU memory before test scoring")
        free_memory()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        # Reload best model weights cleanly onto the device
        if checkpoint_best.exists():
            _bckpt = torch.load(checkpoint_best, map_location="cpu")
            model.load_state_dict(_bckpt["model_state_dict"])
            del _bckpt
        model.to(DEVICE)
        model.eval()
        log(f"{run}: loading and scoring test")
        dfte    = load_frame(TEST_FILE, features)
        yt      = get_y_any(dfte)
        xt_raw  = dfte[features].to_numpy(np.float32)
        del dfte
        xt = scaler.transform(xt_raw)
        del xt_raw
        gc.collect()
        tpc, ts = score_series(model, xt)
        np.save(test_score_path,  ts.astype(np.float32))
        np.save(test_per_ch_path, tpc.astype(np.float32))
        np.save(test_y_path,      yt.astype(np.int8))

    save_json(state_path, {"run": run, "stage": "test_scored",
                           "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")})

    # ── MLP Forecaster training + scoring ────────────────────────────────────
    mlp_val_score_path  = run_dir / "mlp_val_score.npy"
    mlp_test_score_path = run_dir / "mlp_test_score.npy"
    ens_val_score_path  = run_dir / "ens_val_score.npy"
    ens_test_score_path = run_dir / "ens_test_score.npy"

    if TRAIN_MLP:
        mlp_ckpt_last = run_dir / "mlp_checkpoint_last.pt"
        mlp_ckpt_best = run_dir / "mlp_checkpoint_best.pt"

        mlp_model = MLPForecaster(len(features)).to(DEVICE)
        mlp_opt   = torch.optim.AdamW(mlp_model.parameters(), lr=MLP_LR,
                                       weight_decay=WEIGHT_DECAY)
        mlp_crit  = nn.MSELoss()
        mlp_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            mlp_opt, T_max=MLP_EPOCHS, eta_min=1e-6)

        mlp_start = 1
        mlp_best_val  = -1.0
        mlp_best_epoch = -1

        if mlp_ckpt_last.exists():
            mc = torch.load(mlp_ckpt_last, map_location="cpu")
            mlp_model.load_state_dict(mc["model_state_dict"])
            mlp_model.to(DEVICE)
            try:
                mlp_opt.load_state_dict(mc["optimizer_state_dict"])
                mlp_sched.load_state_dict(mc["scheduler_state_dict"])
            except Exception:
                pass
            mlp_start      = int(mc.get("epoch", 0)) + 1
            mlp_best_val   = float(mc.get("best_val", -1.0))
            mlp_best_epoch = int(mc.get("best_epoch", -1))
            log(f"RESUME MLP {run}: start_epoch={mlp_start} best_val={mlp_best_val:.4f}")

        # Build forecast windows from normal training data
        ds_fc  = ForecastWindows(xtr, stride=STRIDE, normal_only=True, y=ytr)
        dl_mlp = DataLoader(ds_fc, MLP_BATCH, shuffle=True,
                            num_workers=0, pin_memory=torch.cuda.is_available())
        log(f"{run}: MLP forecast windows={len(ds_fc):,}")

        mlp_val_every = max(5, MLP_EPOCHS // 6)

        if mlp_start <= MLP_EPOCHS:
            for ep in range(mlp_start, MLP_EPOCHS + 1):
                t0 = time.time()
                mlp_model.train()
                losses = []
                for ctx_b, tgt_b in dl_mlp:
                    ctx_b = ctx_b.to(DEVICE, non_blocking=True)
                    tgt_b = tgt_b.to(DEVICE, non_blocking=True)
                    mlp_opt.zero_grad(set_to_none=True)
                    pred  = mlp_model(ctx_b)
                    loss  = mlp_crit(pred, tgt_b)
                    if not torch.isfinite(loss):
                        raise RuntimeError("MLP non-finite loss")
                    loss.backward()
                    nn.utils.clip_grad_norm_(mlp_model.parameters(), GRAD_CLIP)
                    mlp_opt.step()
                    losses.append(float(loss.detach().cpu()))
                mlp_sched.step()

                if ep % mlp_val_every == 0 or ep == MLP_EPOCHS:
                    _, mlp_vs = score_series_mlp(mlp_model, xv)
                    quick_mlp = []
                    for thr in thresholds(mlp_vs):
                        for mg in [0, 64, 256]:
                            for md in [1, 16, 64]:
                                m = metrics(yv, postprocess(
                                    (mlp_vs > thr).astype(np.int8), mg, md))
                                if m["pred_anomaly_rate"] <= 0.15 and not m["saturated"]:
                                    quick_mlp.append((m["esa_f05"], thr))
                    if quick_mlp:
                        best_q = max(quick_mlp, key=lambda z: z[0])
                        if best_q[0] > mlp_best_val:
                            mlp_best_val   = best_q[0]
                            mlp_best_epoch = ep
                            torch.save({
                                "model_state_dict": {
                                    k: v.detach().cpu().clone()
                                    for k, v in mlp_model.state_dict().items()},
                                "epoch":     ep,
                                "best_val":  mlp_best_val,
                                "best_epoch": mlp_best_epoch,
                            }, mlp_ckpt_best)
                            log(f"MLP saved best: epoch={ep} val_esa={mlp_best_val:.4f}")
                    del mlp_vs; gc.collect()

                torch.save({
                    "model_state_dict":     mlp_model.state_dict(),
                    "optimizer_state_dict": mlp_opt.state_dict(),
                    "scheduler_state_dict": mlp_sched.state_dict(),
                    "epoch":      ep, "best_val":  mlp_best_val,
                    "best_epoch": mlp_best_epoch,
                }, mlp_ckpt_last)
                log(f"{run} MLP ep={ep}/{MLP_EPOCHS} loss={np.mean(losses):.6f} "
                    f"best_val_esa={mlp_best_val:.4f} sec={time.time()-t0:.1f}")

        # Load best MLP checkpoint
        if mlp_ckpt_best.exists():
            mc = torch.load(mlp_ckpt_best, map_location="cpu")
            mlp_model.load_state_dict(mc["model_state_dict"])
            mlp_model.to(DEVICE)
            log(f"loaded best MLP checkpoint epoch={mc['epoch']} val_esa={mc['best_val']:.4f}")

        # MLP val scoring
        if mlp_val_score_path.exists():
            log(f"RESUME {run}: loading cached MLP val scores")
            mlp_vs = np.load(mlp_val_score_path)
        else:
            log(f"{run}: MLP scoring validation")
            free_memory()
            if torch.cuda.is_available():
                torch.cuda.synchronize(); torch.cuda.empty_cache()
            mlp_vpc, mlp_vs = score_series_mlp(mlp_model, xv)
            np.save(mlp_val_score_path,              mlp_vs.astype(np.float32))
            np.save(run_dir / "mlp_val_score_per_channel.npy",
                    mlp_vpc.astype(np.float32))

        # MLP test scoring — reload test data if needed
        if mlp_test_score_path.exists():
            log(f"RESUME {run}: loading cached MLP test scores")
            mlp_ts = np.load(mlp_test_score_path)
        else:
            log(f"{run}: MLP clearing GPU + loading test for scoring")
            free_memory()
            if torch.cuda.is_available():
                torch.cuda.synchronize(); torch.cuda.empty_cache()
            if mlp_ckpt_best.exists():
                mc = torch.load(mlp_ckpt_best, map_location="cpu")
                mlp_model.load_state_dict(mc["model_state_dict"])
            mlp_model.to(DEVICE); mlp_model.eval()
            # Reload test data (may have been freed after AE scoring)
            dfte_mlp   = load_frame(TEST_FILE, features)
            yt_mlp     = get_y_any(dfte_mlp)
            xt_raw_mlp = dfte_mlp[features].to_numpy(np.float32)
            del dfte_mlp
            xt_mlp = scaler.transform(xt_raw_mlp)
            del xt_raw_mlp; gc.collect()
            mlp_tpc, mlp_ts = score_series_mlp(mlp_model, xt_mlp)
            np.save(mlp_test_score_path,             mlp_ts.astype(np.float32))
            np.save(run_dir / "mlp_test_score_per_channel.npy",
                    mlp_tpc.astype(np.float32))
            del xt_mlp, mlp_tpc; gc.collect()

        # Ensemble scores (OR-logic: max of normalised AE and MLP)
        if not ens_val_score_path.exists():
            log(f"{run}: computing ensemble val score")
            ens_vs = ensemble_scores(vs, mlp_vs)
            np.save(ens_val_score_path, ens_vs.astype(np.float32))

        if not ens_test_score_path.exists():
            log(f"{run}: computing ensemble test score")
            _ae_ts  = np.load(test_score_path)
            _mlp_ts = np.load(mlp_test_score_path)
            ens_ts  = ensemble_scores(_ae_ts, _mlp_ts)
            np.save(ens_test_score_path, ens_ts.astype(np.float32))
            del _ae_ts, _mlp_ts, ens_ts; gc.collect()

        free_memory(mlp_model, mlp_opt, mlp_sched, ds_fc, dl_mlp)
        log(f"{run}: MLP + ensemble done")

    # ── end MLP block ────────────────────────────────────────────────────────

    threshold_path = run_dir / "threshold_sweep_val_to_test.csv"
    if threshold_path.exists():
        log(f"RESUME {run}: loading cached threshold sweep")
        thdf = pd.read_csv(threshold_path)
    else:
        log(f"{run}: sweeping thresholds")
        thdf = sweep_thresholds(vs, yv, ts, yt)
        for col, val in [("run", run), ("group", spec.get("group", "")),
                         ("num_features", len(features)), ("features", ",".join(features))]:
            if col not in thdf.columns:
                thdf.insert(0, col, val)
        thdf.to_csv(threshold_path, index=False)

    selected_path = run_dir / "selected_thresholds.csv"
    if selected_path.exists():
        log(f"RESUME {run}: loading cached selected thresholds")
        seldf = pd.read_csv(selected_path)
    else:
        seldf = select_rows(thdf)
        for col, val in [("run", run), ("group", spec.get("group", "")),
                         ("num_features", len(features)), ("features", ",".join(features))]:
            if col not in seldf.columns:
                seldf.insert(0, col, val)
        seldf.to_csv(selected_path, index=False)

    if len(seldf) == 0:
        raise RuntimeError(f"No selected threshold rows for {run}")

    # LEAK FIX: the mask, stored threshold, and plotted result are now chosen by
    # validation only (best val ESA F0.5, val rate <= 1%). Previously this block
    # sorted validation-selected candidates by test_esa_f05 -- a test-set leak.
    prow = pick_leakfree_row(seldf)
    thr  = float(prow["threshold"])
    mg   = int(prow["merge_gap"])
    md   = int(prow["min_dur"])

    if test_pred_path.exists():
        pred = np.load(test_pred_path)
    else:
        pred = postprocess((ts > thr).astype(np.int8), mg, md)
        np.save(test_pred_path, pred.astype(np.int8))

    if val_pred_path.exists():
        vpred = np.load(val_pred_path)
    else:
        vpred = postprocess((vs > thr).astype(np.int8), mg, md)
        np.save(val_pred_path, vpred.astype(np.int8))

    torch.save({
        "model_state_dict": model.state_dict(),
        "features":    features,
        "scaler_median": scaler.median_,
        "scaler_iqr":    scaler.iqr_,
        "threshold":   thr,
        "merge_gap":   mg,
        "min_dur":     md,
        "selected_row": prow,
        "best_epoch":  best_epoch,
        "best_val":    best_val,
    }, run_dir / "model_checkpoint.pt")

    title = (
        f"{run} | {prow.get('selection_name','')} | "
        f"ESA={float(prow.get('test_esa_f05',0)):.4f} "
        f"event={float(prow.get('test_event_f05',0)):.4f} "
        f"point={float(prow.get('test_point_f05',0)):.4f} "
        f"pred_rate={float(prow.get('test_pred_anomaly_rate',0)):.4f}"
    )

    if not (plot_dir / "overview.jpg").exists():
        log(f"{run}: writing overview plot")
        plot_overview(plot_dir / "overview.jpg", ts, pred, yt, thr, title)

    if not (plot_dir / "per_channel_scores.jpg").exists():
        log(f"{run}: writing per-channel plot")
        plot_channels(plot_dir / "per_channel_scores.jpg", tpc, features, yt, thr, f"Per-channel scores | {run}")

    if not any(plot_dir.glob("event_zoom_*.jpg")):
        log(f"{run}: writing event zoom plots")
        plot_zooms(plot_dir, ts, pred, yt, thr, run)

    summary = {
        "mission": "Mission1", "run": run, "group": spec.get("group", ""),
        "features": features, "num_features": len(features),
        "num_train_rows": int(len(xtr)), "num_val_rows": int(len(yv)),
        "num_test_rows": int(len(yt)),
        "val_true_events":  len(find_events(yv)),
        "test_true_events": len(find_events(yt)),
        "best_epoch": int(best_epoch), "best_val_esa_quick": float(best_val),
        "history": hist, "plot_selection": prow,
        "outputs": {
            "selected_thresholds": str(selected_path.resolve()),
            "threshold_sweep":     str(threshold_path.resolve()),
            "overview_plot":       str((plot_dir / "overview.jpg").resolve()),
            "per_channel_plot":    str((plot_dir / "per_channel_scores.jpg").resolve()),
            "checkpoint_last":     str(checkpoint_last.resolve()),
            "checkpoint_best":     str(checkpoint_best.resolve()),
        },
    }

    save_json(run_dir / "summary.json", summary)
    save_json(state_path, {"run": run, "stage": "completed",
                           "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")})

    # Summary reflects the LEAK-FREE selection. test_esa_f05 is recorded as a
    # measured diagnostic ('_diag'), never as the quantity selection optimised.
    wb.log_summary({
        "selection_name":   prow.get("selection_name"),
        "n_channels":       len(features),
        "threshold":        thr, "merge_gap": mg, "min_dur": md,
        "val_esa_f05":      prow.get("val_esa_f05"),
        "val_pred_rate":    prow.get("val_pred_anomaly_rate"),
        "val_event_f05":    prow.get("val_event_f05"),
        "test_esa_f05_diag":   prow.get("test_esa_f05"),
        "test_event_f05_diag": prow.get("test_event_f05"),
        "best_epoch":       best_epoch,
        "mlp_best_val":     (mlp_best_val if TRAIN_MLP else None),
    })
    wb.finish()

    free_memory(model, opt, sched, xtr, xv, ytr, yv, yt)
    return summary


def combine(out_base: Path):
    sels, ths = [], []
    for sd in sorted(out_base.glob("*/selected_thresholds.csv")):
        sels.append(pd.read_csv(sd))
    for td in sorted(out_base.glob("*/threshold_sweep_val_to_test.csv")):
        ths.append(pd.read_csv(td))
    if sels:
        allsel = pd.concat(sels, ignore_index=True)
        allsel.to_csv(out_base / "selected_thresholds_all_runs.csv", index=False)
        excel = allsel.copy()
        for c in ["test_esa_f05","test_event_f05","test_point_f05","test_pred_anomaly_rate","val_esa_f05"]:
            excel[c] = pd.to_numeric(excel[c], errors="coerce")
        excel = excel.sort_values(
            ["selection_type","test_esa_f05","test_event_f05","test_point_f05"],
            ascending=[True,False,False,False])
        keep = ["selection_type","selection_name","run","group","num_features","features",
                "threshold","merge_gap","min_dur","val_esa_f05","val_event_f05",
                "val_event_precision","val_event_recall","val_point_f05","val_point_precision",
                "val_point_recall","val_pred_anomaly_rate","test_esa_f05","test_event_f05",
                "test_event_precision","test_event_recall","test_TPe","test_FPe","test_FNe",
                "test_num_true_events","test_num_pred_events","test_point_f05",
                "test_point_precision","test_point_recall","test_TPt","test_FPt","test_TNt",
                "test_FNt","test_pred_anomaly_rate","test_true_anomaly_rate"]
        excel[[c for c in keep if c in excel.columns]].to_csv(
            out_base / "excel_summary_rows.csv", index=False)
        sat = allsel.test_saturated
        if sat.dtype == object:
            sat = sat.astype(str).str.lower().isin(["true","1","yes"])
        practical = allsel[
            (pd.to_numeric(allsel.test_pred_anomaly_rate, errors="coerce") <= .10) & (~sat)
        ].copy()
        if len(practical):
            practical = practical.sort_values(
                ["test_esa_f05","test_event_f05","test_point_f05"], ascending=False)
            practical.to_csv(
                out_base / "diagnostic_ranking_test_pred_rate_lte_0.10.csv", index=False)
    if ths:
        pd.concat(ths, ignore_index=True).to_csv(
            out_base / "threshold_sweep_all_runs.csv", index=False)


def write_error(out_base: Path, run: str, e: Exception):
    p = out_base / "errors.csv"; exists = p.exists()
    with open(p, "a", encoding="utf-8") as f:
        if not exists:
            f.write("run,error_type,error,created_at\n")
        msg = str(e).replace("\n"," ").replace("\r"," ").replace(","," ;")
        f.write(f"{run},{type(e).__name__},{msg},{time.strftime('%Y-%m-%d %H:%M:%S')}\n")


def main():
    set_seed(SEED)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    latest_path = OUT_ROOT / "latest_run.txt"
    out_base    = None

    if latest_path.exists():
        try:
            candidate = Path(latest_path.read_text(encoding="utf-8").strip())
            if candidate.exists():
                out_base = candidate
        except Exception:
            out_base = None

    if out_base is None:
        out_base = OUT_ROOT / time.strftime("%Y%m%d_%H%M%S")
        out_base.mkdir(parents=True, exist_ok=True)
    else:
        out_base.mkdir(parents=True, exist_ok=True)

    latest_path.write_text(str(out_base.resolve()) + "\n", encoding="utf-8")
    (out_base / "plots").mkdir(exist_ok=True)

    available = read_columns(TRAIN_FILE)
    planned   = [{**s, "features": [f for f in s["features"] if f in available]}
                 for s in RUN_SPECS]
    summary   = {
        "run_id": out_base.name, "out_dir": str(out_base.resolve()),
        "mission": "Mission1", "method": "reconstruction_autoencoder_normal_windows",
        "train_file": str(TRAIN_FILE), "test_file": str(TEST_FILE),
        "runs_planned": planned,
        "config": {
            "seq_len": SEQ_LEN, "patch_size": PATCH_SIZE, "stride": STRIDE,
            "score_stride": SCORE_STRIDE, "d_model": D_MODEL, "n_heads": N_HEADS,
            "n_layers": N_LAYERS, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
            "merge_gaps": MERGE_GAPS, "min_durs": MIN_DURS,
        },
        "runs": [], "errors": [],
    }

    log("="*100)
    log("Mission1 reconstruction AE + MLP Forecaster sweep  [OPTIMIZED v2]")
    log(f"device={DEVICE}")
    if torch.cuda.is_available():
        log(f"gpu={torch.cuda.get_device_name(0)}")
    log(f"output={out_base}")
    log(f"planned runs={len(planned)}")
    log(f"MLP forecaster: {'ENABLED' if TRAIN_MLP else 'DISABLED'}  "
        f"(epochs={MLP_EPOCHS}  hidden={MLP_HIDDEN}  layers={MLP_LAYERS})")
    log("resume mode: uses latest_run.txt and cached checkpoints/scores when present")

    for i, s in enumerate(planned, 1):
        if not s["features"]:
            continue
        try:
            log(f"Starting {i}/{len(planned)} {s['run']}")
            res = train_one(s, out_base, available)
            summary["runs"].append(res)
        except torch.cuda.OutOfMemoryError as e:
            log(f"CUDA OOM {s['run']}: {e}")
            write_error(out_base, s["run"], e)
            summary["errors"].append({"run": s["run"], "error_type": "CUDA_OutOfMemoryError", "error": str(e)})
            free_memory(); continue
        except Exception as e:
            log(f"ERROR {s['run']}: {type(e).__name__}: {e}")
            write_error(out_base, s["run"], e)
            summary["errors"].append({"run": s["run"], "error_type": type(e).__name__, "error": str(e)})
            free_memory(); continue
        combine(out_base)
        (out_base / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    combine(out_base)
    summary["num_successful_runs"] = len(summary["runs"])
    summary["num_failed_runs"]     = len(summary["errors"])
    summary["outputs"] = {
        "selected_thresholds_all_runs": str((out_base / "selected_thresholds_all_runs.csv").resolve()),
        "threshold_sweep_all_runs":     str((out_base / "threshold_sweep_all_runs.csv").resolve()),
        "excel_summary_rows":           str((out_base / "excel_summary_rows.csv").resolve()),
        "diagnostic_ranking_test_pred_rate_lte_0.10": str((out_base / "diagnostic_ranking_test_pred_rate_lte_0.10.csv").resolve()),
        "plots_dir":                    str((out_base / "plots").resolve()),
    }
    (out_base / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log("="*100)
    log(f"Done  successful={summary['num_successful_runs']}  failed={summary['num_failed_runs']}")
    log(f"saved to {out_base}")

    if (out_base / "excel_summary_rows.csv").exists():
        df   = pd.read_csv(out_base / "excel_summary_rows.csv")
        cols = ["selection_type","selection_name","run","test_esa_f05","test_event_f05",
                "test_point_f05","test_event_precision","test_event_recall","test_pred_anomaly_rate"]
        print(df[[c for c in cols if c in df.columns]].head(30).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
