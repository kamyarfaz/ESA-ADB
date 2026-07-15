#!/usr/bin/env python3
"""
channel_correlation_analysis.py
================================
Computes pairwise Pearson correlation between all Mission-1 channels
using NORMAL timesteps only (where y=0), then:

  1. Plots a full correlation heatmap
  2. Clusters channels via hierarchical linkage
  3. Prints correlation-based channel groups
  4. Suggests new RUN_SPECS based on those groups
  5. Shows which existing run subsets are well vs poorly correlated

Run:
    python channel_correlation_analysis.py
    python channel_correlation_analysis.py --max_rows 500000  # faster, uses a sample
    python channel_correlation_analysis.py --min_corr 0.7     # tighter cluster threshold
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform

# ── config ─────────────────────────────────────────────────────────────────────
TRAIN_FILE  = Path("data/preprocessed/multivariate/ESA-Mission1-semi-supervised/84_months.train.csv")
LABELS_FILE = Path("data/ESA-Mission1/labels.csv")
OUT_DIR     = Path("results_longrun/channel_correlation")
N_CHANNELS  = 65          # Mission-1 has channels 1–65 (adjust if different)
SEED        = 42


# ── helpers ────────────────────────────────────────────────────────────────────

def load_normal_data(max_rows: int) -> pd.DataFrame:
    """Load training CSV, keep only normal rows (is_anomaly == 0)."""
    print(f"Loading training data from {TRAIN_FILE} ...")
    df = pd.read_csv(TRAIN_FILE, nrows=max_rows if max_rows else None)
    print(f"  total rows: {len(df):,}")

    # detect label column
    label_col = None
    for c in ["is_anomaly", "label", "anomaly", "y"]:
        if c in df.columns:
            label_col = c
            break

    if label_col:
        before = len(df)
        df = df[df[label_col] == 0].reset_index(drop=True)
        print(f"  normal rows after filtering '{label_col}': {len(df):,}  "
              f"(removed {before - len(df):,} anomalous)")
    else:
        print("  WARNING: no label column found — using all rows as normal")

    return df


def get_channel_cols(df: pd.DataFrame) -> list:
    """Return sorted channel_* columns present in the dataframe."""
    cols = sorted(
        [c for c in df.columns if c.startswith("channel_")],
        key=lambda x: int(x.split("_")[1])
    )
    print(f"  channel columns found: {len(cols)}  ({cols[0]} … {cols[-1]})")
    return cols


def compute_correlation(df: pd.DataFrame, cols: list, max_rows: int) -> pd.DataFrame:
    """Compute Pearson correlation matrix on a random sample if needed."""
    data = df[cols]
    if max_rows and len(data) > max_rows:
        rng  = np.random.default_rng(SEED)
        idx  = rng.choice(len(data), size=max_rows, replace=False)
        data = data.iloc[idx]
        print(f"  sampling {max_rows:,} rows for correlation")
    print(f"  computing {len(cols)}×{len(cols)} correlation matrix ...")
    corr = data.corr(method="pearson")
    return corr


# ── plots ───────────────────────────────────────────────────────────────────────

def plot_heatmap(corr: pd.DataFrame, out_path: Path) -> None:
    n    = len(corr)
    size = max(12, n * 0.22)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))

    # colour map: red=negative, white=zero, blue=positive
    cmap = plt.cm.RdBu_r
    im   = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Pearson r")

    labels = [c.replace("channel_", "ch") for c in corr.columns]
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=7)
    ax.set_title("Channel Pearson correlation — normal data only", fontsize=13, pad=12)

    # grid lines every 5 channels
    for k in range(0, n, 5):
        ax.axhline(k - 0.5, color="white", lw=0.4, alpha=0.5)
        ax.axvline(k - 0.5, color="white", lw=0.4, alpha=0.5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved heatmap: {out_path}")


def plot_clustered_heatmap(corr: pd.DataFrame, labels_order: list,
                           cluster_ids: np.ndarray, out_path: Path) -> None:
    n    = len(corr)
    size = max(12, n * 0.22)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))

    reordered = corr.loc[labels_order, labels_order]
    cmap = plt.cm.RdBu_r
    im   = ax.imshow(reordered.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Pearson r")

    short = [c.replace("channel_", "ch") for c in labels_order]
    ax.set_xticks(range(n)); ax.set_xticklabels(short, rotation=90, fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(short, fontsize=7)
    ax.set_title("Channel correlation — clustered order", fontsize=13, pad=12)

    # draw cluster boundaries
    prev  = 0
    seen  = {}
    order_ids = [cluster_ids[list(corr.columns).index(c)] for c in labels_order]
    for i, cid in enumerate(order_ids):
        if cid not in seen:
            if i > 0:
                ax.axhline(i - 0.5, color="lime", lw=1.2)
                ax.axvline(i - 0.5, color="lime", lw=1.2)
            seen[cid] = True

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved clustered heatmap: {out_path}")


def plot_dendrogram(corr: pd.DataFrame, linkage_mat, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(20, 5))
    labels  = [c.replace("channel_", "ch") for c in corr.columns]
    dendrogram(linkage_mat, labels=labels, ax=ax,
               leaf_rotation=90, leaf_font_size=8,
               color_threshold=0.4 * linkage_mat[-1, 2])
    ax.set_title("Channel hierarchical clustering dendrogram", fontsize=12)
    ax.set_ylabel("Distance (1 − |r|)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved dendrogram: {out_path}")


# ── cluster analysis ────────────────────────────────────────────────────────────

def cluster_channels(corr: pd.DataFrame, min_corr: float):
    """
    Hierarchical clustering on distance = 1 - |r|.
    min_corr: channels with |r| >= min_corr are in the same cluster.
    """
    dist   = 1.0 - corr.abs().values
    np.fill_diagonal(dist, 0.0)
    dist   = np.clip(dist, 0, None)   # floating point safety
    cond   = squareform(dist, checks=False)
    Z      = linkage(cond, method="average")
    thresh = 1.0 - min_corr
    ids    = fcluster(Z, t=thresh, criterion="distance")
    return Z, ids


def summarise_clusters(corr: pd.DataFrame, cluster_ids: np.ndarray,
                       min_corr: float) -> dict:
    cols   = list(corr.columns)
    groups = {}
    for col, cid in zip(cols, cluster_ids):
        groups.setdefault(cid, []).append(col)

    print(f"\n{'='*70}")
    print(f"CHANNEL CLUSTERS  (min |r| = {min_corr})")
    print(f"{'='*70}")

    cluster_info = {}
    for cid in sorted(groups, key=lambda k: len(groups[k]), reverse=True):
        members  = sorted(groups[cid], key=lambda x: int(x.split("_")[1]))
        nums     = [int(c.split("_")[1]) for c in members]
        sub      = corr.loc[members, members]
        avg_corr = (sub.values.sum() - len(members)) / max(len(members) * (len(members) - 1), 1)
        is_contig = (max(nums) - min(nums) == len(nums) - 1)

        print(f"\n  Cluster {cid:2d}  ({len(members)} channels)  "
              f"avg |r|={avg_corr:.3f}  "
              f"{'contiguous' if is_contig else 'non-contiguous'}")
        print(f"    channels : {', '.join(members)}")
        if is_contig:
            print(f"    range    : channel_{min(nums)} – channel_{max(nums)}")

        cluster_info[cid] = {
            "members":    members,
            "nums":       nums,
            "avg_corr":   avg_corr,
            "contiguous": is_contig,
        }

    return cluster_info


# ── existing run audit ──────────────────────────────────────────────────────────

EXISTING_RUNS = {
    "fed_ch06_41_46":    list(range(41, 47)),
    "center_ch08_40_47": list(range(40, 48)),
    "center_ch12_38_49": list(range(38, 50)),
    "block_ch08_01_08":  list(range(1,  9)),
    "block_ch08_17_24":  list(range(17, 25)),
    "block_ch08_33_40":  list(range(33, 41)),
    "fine_ch08_03_10":   list(range(3,  11)),
    "fine_ch08_05_12":   list(range(5,  13)),
    "fine_ch08_13_20":   list(range(13, 21)),
    "fine_ch08_15_22":   list(range(15, 23)),
    "fine_ch08_19_26":   list(range(19, 27)),
    "fine_ch08_21_28":   list(range(21, 29)),
    "fine_ch08_29_36":   list(range(29, 37)),
    "fine_ch08_31_38":   list(range(31, 39)),
    "fine_ch08_35_42":   list(range(35, 43)),
    "fine_ch08_37_44":   list(range(37, 45)),
    "combo_ch16_01_16":  list(range(1,  17)),
    "combo_ch16_17_32":  list(range(17, 33)),
    "combo_ch16_29_44":  list(range(29, 45)),
    "combo_ch24_17_40":  list(range(17, 41)),
}


def audit_existing_runs(corr: pd.DataFrame) -> None:
    print(f"\n{'='*70}")
    print("EXISTING RUN SUBSET — INTERNAL CORRELATION AUDIT")
    print(f"{'='*70}")
    print(f"  {'run':30s}  {'n_ch':4s}  {'avg|r|':7s}  {'min|r|':7s}  {'verdict':10s}")
    print(f"  {'-'*30}  {'-'*4}  {'-'*7}  {'-'*7}  {'-'*10}")

    for run, nums in EXISTING_RUNS.items():
        cols = [f"channel_{i}" for i in nums if f"channel_{i}" in corr.columns]
        if len(cols) < 2:
            continue
        sub     = corr.loc[cols, cols].abs()
        n       = len(cols)
        vals    = sub.values
        off_diag = vals[~np.eye(n, dtype=bool)]
        avg_r   = off_diag.mean()
        min_r   = off_diag.min()

        if avg_r >= 0.7:
            verdict = "tight"
        elif avg_r >= 0.5:
            verdict = "moderate"
        elif avg_r >= 0.3:
            verdict = "loose"
        else:
            verdict = "weak"

        print(f"  {run:30s}  {n:4d}  {avg_r:7.3f}  {min_r:7.3f}  {verdict}")


# ── new run suggestions ─────────────────────────────────────────────────────────

def suggest_new_runs(cluster_info: dict, corr: pd.DataFrame,
                     min_cluster_size: int = 4) -> None:
    print(f"\n{'='*70}")
    print("SUGGESTED NEW RUN SPECS  (based on correlation clusters)")
    print(f"{'='*70}")
    print("\nRUN_SPECS = [")

    suggestions = []
    for cid, info in sorted(cluster_info.items(),
                             key=lambda x: x[1]["avg_corr"], reverse=True):
        members = info["members"]
        nums    = info["nums"]
        if len(members) < min_cluster_size:
            continue

        # contiguous ranges → direct run
        if info["contiguous"]:
            mn, mx = min(nums), max(nums)
            n      = len(nums)
            name   = f"corr_ch{n:02d}_{mn:02d}_{mx:02d}"
            group  = f"corr_cluster_{cid}"
            print(f'    {{"run": "{name}", '
                  f'"features": ch_range({mn}, {mx}), '
                  f'"group": "{group}"}},  '
                  f'# avg|r|={info["avg_corr"]:.2f}  n={n}')
            suggestions.append((name, mn, mx, info["avg_corr"]))

        else:
            # non-contiguous cluster → suggest the bounding range as a run
            mn, mx = min(nums), max(nums)
            n_span = mx - mn + 1
            n      = len(nums)
            name   = f"corr_span_ch{n_span:02d}_{mn:02d}_{mx:02d}"
            group  = f"corr_cluster_{cid}_span"
            print(f'    {{"run": "{name}", '
                  f'"features": ch_range({mn}, {mx}), '
                  f'"group": "{group}"}},  '
                  f'# avg|r|={info["avg_corr"]:.2f}  n_cluster={n}  '
                  f'non-contiguous span')
            suggestions.append((name, mn, mx, info["avg_corr"]))

    print("]")

    # top pairs — highest correlated pairs not yet in a run
    print(f"\n{'='*70}")
    print("TOP 15 HIGHEST CORRELATED CHANNEL PAIRS")
    print(f"{'='*70}")
    print(f"  {'channel A':12s}  {'channel B':12s}  {'r':7s}")
    print(f"  {'-'*12}  {'-'*12}  {'-'*7}")
    cols   = list(corr.columns)
    pairs  = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append((abs(corr.iloc[i, j]), cols[i], cols[j]))
    pairs.sort(reverse=True)
    for r, ca, cb in pairs[:15]:
        print(f"  {ca:12s}  {cb:12s}  {r:7.4f}")


# ── stats summary ───────────────────────────────────────────────────────────────

def print_stats(corr: pd.DataFrame) -> None:
    vals = corr.abs().values
    np.fill_diagonal(vals, np.nan)
    flat = vals.flatten()
    flat = flat[~np.isnan(flat)]
    print(f"\n{'='*70}")
    print("GLOBAL CORRELATION STATS (|r|, off-diagonal)")
    print(f"{'='*70}")
    print(f"  mean   : {np.mean(flat):.4f}")
    print(f"  median : {np.median(flat):.4f}")
    print(f"  max    : {np.max(flat):.4f}")
    print(f"  min    : {np.min(flat):.4f}")
    print(f"  pairs |r|>=0.9 : {(flat>=0.9).sum()}")
    print(f"  pairs |r|>=0.7 : {(flat>=0.7).sum()}")
    print(f"  pairs |r|>=0.5 : {(flat>=0.5).sum()}")
    print(f"  pairs |r|<0.1  : {(flat<0.1).sum()}")


# ── main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Channel correlation analysis for Mission-1"
    )
    parser.add_argument("--max_rows",    type=int,   default=1_000_000,
                        help="Max rows to load for correlation (default 1M, 0=all)")
    parser.add_argument("--min_corr",    type=float, default=0.6,
                        help="Min |r| to consider channels in same cluster (default 0.6)")
    parser.add_argument("--sample_rows", type=int,   default=500_000,
                        help="Rows sampled for correlation matrix (default 500k)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── load data ──────────────────────────────────────────────────────────────
    df   = load_normal_data(args.max_rows)
    cols = get_channel_cols(df)

    # ── correlation ────────────────────────────────────────────────────────────
    corr = compute_correlation(df, cols, args.sample_rows)
    del df

    # save raw matrix
    csv_path = OUT_DIR / "channel_correlation_matrix.csv"
    corr.to_csv(csv_path)
    print(f"  saved matrix: {csv_path}")

    print_stats(corr)

    # ── clustering ─────────────────────────────────────────────────────────────
    print(f"\nClustering channels (min |r| = {args.min_corr}) ...")
    Z, cluster_ids = cluster_channels(corr, args.min_corr)

    # reorder columns by dendrogram leaf order
    from scipy.cluster.hierarchy import leaves_list
    order        = leaves_list(Z)
    ordered_cols = [cols[i] for i in order]

    # ── plots ──────────────────────────────────────────────────────────────────
    print("\nGenerating plots ...")
    plot_heatmap(corr, OUT_DIR / "heatmap_raw.jpg")
    plot_clustered_heatmap(corr, ordered_cols, cluster_ids,
                           OUT_DIR / "heatmap_clustered.jpg")
    plot_dendrogram(corr, Z, OUT_DIR / "dendrogram.jpg")

    # ── analysis ───────────────────────────────────────────────────────────────
    cluster_info = summarise_clusters(corr, cluster_ids, args.min_corr)
    audit_existing_runs(corr)
    suggest_new_runs(cluster_info, corr)

    print(f"\n{'='*70}")
    print(f"All outputs saved to: {OUT_DIR}/")
    print(f"  heatmap_raw.jpg        — full correlation heatmap")
    print(f"  heatmap_clustered.jpg  — reordered by cluster")
    print(f"  dendrogram.jpg         — hierarchical tree")
    print(f"  channel_correlation_matrix.csv — raw matrix")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
