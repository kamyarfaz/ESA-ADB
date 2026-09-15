"""Config components of the Mission 1 research pipeline.

Extracted without changing numerical behavior; see docs/thesis-research-context.md.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
from .paths import project_path


WANDB_PROJECT = "esa-adb-mission1"


TRAIN_FILE = project_path("data/preprocessed/multivariate/ESA-Mission1-semi-supervised/84_months.train.csv")


TEST_FILE  = project_path("data/preprocessed/multivariate/ESA-Mission1-semi-supervised/84_months.test.csv")


OUT_ROOT   = project_path("results_longrun/mission1_reconstruction_ae_sweep_optimized")


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


TRAIN_MLP    = True       # set False to skip MLP and run AE only


MLP_PRED_LEN = 32         # timesteps to predict ahead (= SCORE_STRIDE)


MLP_HIDDEN   = 256        # hidden layer width


MLP_LAYERS   = 4          # number of hidden layers


MLP_EPOCHS   = 30         # fewer epochs than AE (MLP trains faster)


MLP_LR       = 3e-4       # learning rate for MLP


MLP_BATCH    = 512        # larger batch fits because MLP is lighter than AE


THRESH_PERCENTILES = list(np.concatenate([
    np.linspace(50, 95, 10),
    np.array([96, 97, 98, 99, 99.3, 99.5, 99.7, 99.8, 99.9, 99.95, 99.97, 99.99]),
]))


MERGE_GAPS = [0, 16, 32, 64, 128, 256, 512, 1024]


MIN_DURS   = [1, 8, 16, 32, 64, 128]


SATURATION_RATE = 0.80


MAX_ZOOM_EVENTS = 24


ZOOM_PAD   = 4096


PA_LAMBDA = 1.0            # weight of the pseudo-anomaly separation term


PA_MARGIN = 10.0           # pseudo recon error must reach >= PA_MARGIN x mean normal error


PA_SUBWIN = (0.05, 0.33)   # corrupted sub-window length, as a fraction of SEQ_LEN


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

    # ── NEW: pseudo-anomaly training on the SAME channel sets as the best runs ──
    # Only the training method differs (pseudo-anomaly loss), so these are direct
    # A/B tests vs their normal-trained twins. Same architecture, same budget.
    {"run": "pa_center_ch12_38_49", "features": ch_range(38, 49),
     "pseudo": True, "group": "pseudo_anomaly"},        # twin of center_ch12_38_49 (0.8011)
    {"run": "pa_center_ch08_40_47", "features": ch_range(40, 47),
     "pseudo": True, "group": "pseudo_anomaly"},        # twin of center_ch08_40_47 (0.7865)
    {"run": "pa_clust_med_t03_6ch", "features": ch_list([3, 5, 13, 29, 48, 57]),
     "pseudo": True, "selection": "clust_med_t03", "group": "pseudo_anomaly"},  # twin of clust_med (0.6736)
]
