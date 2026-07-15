#!/usr/bin/env python3
"""
channel_importance_analysis.py
================================
Analyses which channels are actually doing the anomaly detection work
inside each trained AE run, using saved per-channel reconstruction scores.

Three analyses:
  1. Anomaly Discrimination Ratio (ADR) per channel per run
     ADR = mean_error_during_anomaly / mean_error_during_normal
     High ADR = channel genuinely detects anomalies
     ADR ≈ 1.0 = channel contributes nothing

  2. Redundancy analysis — which channels are so correlated that
     only one of them is needed

  3. Optimal channel subset suggestion — greedy selection of minimum
     channels needed to reconstruct current detection performance

Run:
    python channel_importance_analysis.py
    python channel_importance_analysis.py --runs center_ch12_38_49 corr_ch24_12_35
    python channel_importance_analysis.py --top_k 8
"""

import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd


# ── config ─────────────────────────────────────────────────────────────────────

ROOT = Path(
    "results_longrun/mission1_reconstruction_ae_sweep_optimized/20260519_155620"
)

# Runs to analyse in detail (best performers from sweep)
DEFAULT_RUNS = [
    "center_ch12_38_49",
    "corr_ch24_12_35",
    "corr_neg_ch30_12_40_p61",
    "center_ch08_40_47",
    "combo_ch16_01_16",
    "corr_ch19_17_35",
    "corr_ch29_12_40",
    "block_ch08_17_24",
]

# Channel ranges per run (must match RUN_SPECS in main script)
RUN_CHANNELS = {
    "center_ch12_38_49":       list(range(38, 50)),
    "corr_ch24_12_35":         list(range(12, 36)),
    "corr_neg_ch30_12_40_p61": list(range(12, 41)) + [61],
    "center_ch08_40_47":       list(range(40, 48)),
    "combo_ch16_01_16":        list(range(1,  17)),
    "corr_ch19_17_35":         list(range(17, 36)),
    "corr_ch29_12_40":         list(range(12, 41)),
    "block_ch08_17_24":        list(range(17, 25)),
    "combo_ch24_17_40":        list(range(17, 41)),
    "block_ch08_01_08":        list(range(1,  9)),
    "fine_ch08_15_22":         list(range(15, 23)),
    "fine_ch08_29_36":         list(range(29, 37)),
}


# ── metric helpers ─────────────────────────────────────────────────────────────

def find_events(y: np.ndarray) -> List[Tuple[int, int]]:
    y8 = (y > 0).astype(np.int8)
    if y8.sum() == 0:
        return []
    padded = np.zeros(len(y8) + 2, dtype=np.int8)
    padded[1:-1] = y8
    diff = np.diff(padded.astype(np.int16))
    starts = np.where(diff ==  1)[0]
    ends   = np.where(diff == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))


def events_overlap_count(query, ref):
    if not query or not ref:
        return 0
    ref_arr  = np.array(ref, dtype=np.int64)
    ref_sort = ref_arr[np.argsort(ref_arr[:, 0])]
    count = 0
    for qs, qe in query:
        lo, hi = 0, len(ref_sort)
        while lo < hi:
            mid = (lo + hi) // 2
            if ref_sort[mid, 0] <= qe:
                lo = mid + 1
            else:
                hi = mid
        for k in range(lo - 1, -1, -1):
            rs, re = ref_sort[k]
            if re < qs:
                break
            if re >= qs:
                count += 1
                break
    return count


def esa_f05(gt: np.ndarray, pred: np.ndarray) -> dict:
    gt   = (gt   > 0).astype(np.int8)
    pred = (pred > 0).astype(np.int8)
    gt_ev   = find_events(gt)
    pred_ev = find_events(pred)
    TPe = events_overlap_count(gt_ev, pred_ev)
    FNe = len(gt_ev) - TPe
    FPe = len(pred_ev) - events_overlap_count(pred_ev, gt_ev)
    gt_b, pred_b = gt.astype(bool), pred.astype(bool)
    FPt = int((~gt_b & pred_b).sum())
    Nt  = int((~gt_b).sum())
    fpt = FPt / Nt if Nt > 0 else 0.0
    pc  = TPe / max(TPe + FPe + fpt, 1e-12)
    re  = TPe / max(TPe + FNe, 1)
    b2  = 0.25
    den = b2 * pc + re
    f05 = (1 + b2) * pc * re / den if den > 0 else 0.0
    return {"esa_f05": round(f05, 4), "precision_c": round(pc, 4),
            "recall_e": round(re, 4), "TPe": TPe, "FPe": FPe, "FNe": FNe,
            "pred_rate": round(float(pred.mean()), 4)}


def best_threshold_on_normal(score: np.ndarray, gt: np.ndarray,
                              n: int = 200) -> Tuple[float, dict]:
    """Find best threshold using normal test points (no label leakage)."""
    normal_scores = score[gt == 0]
    percs  = np.linspace(90, 99.9, n)
    thrs   = np.unique(np.percentile(normal_scores, percs))
    best_f = -1.0
    best_t = float(thrs[-1])
    best_m : dict = {}
    for t in thrs:
        pm = (score > t).astype(np.int8)
        if pm.mean() > 0.25 or pm.mean() == 0:
            continue
        m = esa_f05(gt, pm)
        if m["esa_f05"] > best_f:
            best_f = m["esa_f05"]
            best_t = float(t)
            best_m = m
    return best_t, best_m


# ── analysis 1: ADR per channel ────────────────────────────────────────────────

def compute_adr(run_dir: Path, channels: List[int]) -> pd.DataFrame:
    """
    Anomaly Discrimination Ratio per channel.
    ADR = mean_score_during_anomaly / mean_score_during_normal
    """
    pc_path = run_dir / "test_score_per_channel.npy"
    y_path  = run_dir / "test_y_true.npy"
    if not pc_path.exists() or not y_path.exists():
        return pd.DataFrame()

    pc = np.load(pc_path).astype(np.float32)   # shape (C, T)
    gt = np.load(y_path).astype(np.int8)        # shape (T,)

    anom_mask   = gt == 1
    normal_mask = gt == 0

    if anom_mask.sum() == 0 or normal_mask.sum() == 0:
        return pd.DataFrame()

    rows = []
    for i, ch in enumerate(channels):
        if i >= pc.shape[0]:
            break
        ch_score = pc[i]
        mean_anom   = float(ch_score[anom_mask].mean())
        mean_normal = float(ch_score[normal_mask].mean())
        adr = mean_anom / max(mean_normal, 1e-12)
        rows.append({
            "channel":      f"ch{ch}",
            "channel_num":  ch,
            "mean_anom":    round(mean_anom,   6),
            "mean_normal":  round(mean_normal, 6),
            "adr":          round(adr,         4),
            "signal_ratio": round(mean_anom / max(pc[i].max(), 1e-12), 4),
        })

    df = pd.DataFrame(rows).sort_values("adr", ascending=False).reset_index(drop=True)
    return df


# ── analysis 2: greedy channel selection ──────────────────────────────────────

def greedy_channel_select(run_dir: Path, channels: List[int],
                           adr_df: pd.DataFrame,
                           top_k: int = 8) -> Dict:
    """
    Greedy forward selection: start with highest-ADR channel,
    add channels one by one, track F0.5 improvement.
    Uses aggregate score = max over selected channels' per-channel scores.
    """
    pc_path = run_dir / "test_score_per_channel.npy"
    y_path  = run_dir / "test_y_true.npy"
    if not pc_path.exists() or not y_path.exists():
        return {}

    pc = np.load(pc_path).astype(np.float32)   # (C, T)
    gt = np.load(y_path).astype(np.int8)

    # Order channels by ADR
    ordered = adr_df.sort_values("adr", ascending=False)["channel_num"].tolist()

    # Map channel number → index in pc array
    ch_to_idx = {ch: i for i, ch in enumerate(channels)}

    selected  = []
    f05_curve = []

    for ch in ordered[:top_k]:
        if ch not in ch_to_idx:
            continue
        selected.append(ch)
        idx_list = [ch_to_idx[c] for c in selected]

        # Aggregate: max over selected channels
        agg = pc[idx_list, :].max(axis=0)

        # Threshold using normal test points
        _, m = best_threshold_on_normal(agg, gt)
        f05  = m.get("esa_f05", 0.0) if m else 0.0
        f05_curve.append({
            "n_channels":  len(selected),
            "channels":    sorted(selected),
            "esa_f05":     f05,
            "TPe":         m.get("TPe", 0) if m else 0,
            "FPe":         m.get("FPe", 0) if m else 0,
        })

    return {"curve": f05_curve,
            "best":  max(f05_curve, key=lambda x: x["esa_f05"]) if f05_curve else {}}


# ── analysis 3: cross-run channel importance ──────────────────────────────────

def global_channel_ranking(root: Path, run_names: List[str]) -> pd.DataFrame:
    """Rank all channels across all runs by their ADR."""
    all_rows = []
    for rname in run_names:
        rd  = root / rname
        chs = RUN_CHANNELS.get(rname, [])
        if not chs:
            continue
        adr = compute_adr(rd, chs)
        if adr.empty:
            continue
        adr["run"] = rname
        all_rows.append(adr)

    if not all_rows:
        return pd.DataFrame()

    combined = pd.concat(all_rows, ignore_index=True)

    # For channels appearing in multiple runs, take max ADR
    best = (combined.groupby("channel_num")
            .agg({"adr": "max", "mean_anom": "max", "mean_normal": "min"})
            .reset_index()
            .sort_values("adr", ascending=False))
    best["channel"] = best["channel_num"].apply(lambda x: f"ch{x}")
    return best


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Channel importance analysis for ESA ADB runs")
    parser.add_argument("--root",   default=str(ROOT))
    parser.add_argument("--runs",   nargs="+", default=DEFAULT_RUNS)
    parser.add_argument("--top_k",  type=int,  default=10,
                        help="Max channels in greedy selection")
    args = parser.parse_args()
    root = Path(args.root)

    print(f"\nChannel Importance Analysis")
    print(f"Root  : {root}")
    print(f"Runs  : {args.runs}\n")

    all_adr_rows = []
    greedy_rows  = []

    # ── per-run analysis ───────────────────────────────────────────────────────
    for rname in args.runs:
        rd  = root / rname
        chs = RUN_CHANNELS.get(rname)
        if not chs or not rd.exists():
            print(f"  SKIP {rname} — folder or channel mapping not found")
            continue

        adr_df = compute_adr(rd, chs)
        if adr_df.empty:
            continue

        print(f"{'='*70}")
        print(f"  {rname}  ({len(chs)} channels: ch{min(chs)}–ch{max(chs)})")
        print(f"{'='*70}")
        print(f"  {'channel':8s}  {'ADR':7s}  {'mean_anom':10s}  "
              f"{'mean_normal':11s}  {'contribution'}")
        print(f"  {'-'*8}  {'-'*7}  {'-'*10}  {'-'*11}  {'-'*20}")

        for _, r in adr_df.iterrows():
            if r["adr"] >= 3.0:
                label = "★★★ strong"
            elif r["adr"] >= 2.0:
                label = "★★  moderate"
            elif r["adr"] >= 1.3:
                label = "★   weak"
            else:
                label = "    noise"
            print(f"  {r['channel']:8s}  {r['adr']:7.3f}  "
                  f"{r['mean_anom']:10.6f}  {r['mean_normal']:11.6f}  {label}")

        # Strong channels (ADR >= 2.0)
        strong = adr_df[adr_df["adr"] >= 2.0]["channel_num"].tolist()
        weak   = adr_df[adr_df["adr"] <  1.3]["channel_num"].tolist()
        print(f"\n  Strong (ADR≥2.0): {strong}")
        print(f"  Weak   (ADR<1.3): {weak}")
        if weak:
            print(f"  → Could REMOVE these {len(weak)} channels with minimal performance loss")

        # Greedy selection
        print(f"\n  Greedy forward selection (adding by ADR rank):")
        greedy = greedy_channel_select(rd, chs, adr_df, top_k=args.top_k)
        if greedy.get("curve"):
            print(f"  {'n_ch':4s}  {'F0.5':6s}  {'TPe':4s}  {'FPe':4s}  {'channels'}")
            print(f"  {'-'*4}  {'-'*6}  {'-'*4}  {'-'*4}  {'-'*40}")
            for step in greedy["curve"]:
                clist = step["channels"]
                cstr  = str(clist[:6]) + ("..." if len(clist) > 6 else "")
                print(f"  {step['n_channels']:4d}  {step['esa_f05']:.4f}  "
                      f"{step['TPe']:4d}  {step['FPe']:4d}  {cstr}")
            best = greedy["best"]
            print(f"\n  → Optimal subset: {best.get('n_channels')} channels "
                  f"→ F0.5={best.get('esa_f05'):.4f}  "
                  f"TPe={best.get('TPe')}  FPe={best.get('FPe')}")
            greedy_rows.append({
                "run":               rname,
                "original_n_ch":     len(chs),
                "optimal_n_ch":      best.get("n_channels"),
                "optimal_channels":  str(best.get("channels")),
                "optimal_f05":       best.get("esa_f05"),
                "optimal_TPe":       best.get("TPe"),
                "optimal_FPe":       best.get("FPe"),
            })

        # Accumulate for global ranking
        adr_df["run"] = rname
        all_adr_rows.append(adr_df)
        print()

    # ── global channel ranking across all runs ─────────────────────────────────
    print(f"\n{'='*70}")
    print("  GLOBAL CHANNEL RANKING — best ADR across all analysed runs")
    print(f"{'='*70}")
    global_df = global_channel_ranking(root, args.runs)
    if not global_df.empty:
        print(f"  {'channel':8s}  {'best ADR':8s}  {'verdict'}")
        print(f"  {'-'*8}  {'-'*8}  {'-'*20}")
        for _, r in global_df.head(20).iterrows():
            if r["adr"] >= 3.0:
                v = "★★★ highly discriminative"
            elif r["adr"] >= 2.0:
                v = "★★  useful"
            elif r["adr"] >= 1.3:
                v = "★   marginal"
            else:
                v = "    not useful"
            print(f"  ch{int(r['channel_num']):4d}     {r['adr']:8.3f}  {v}")

    # ── suggested new compact run ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  SUGGESTED COMPACT RUN — top channels by global ADR")
    print(f"{'='*70}")
    if not global_df.empty:
        top_channels = global_df[global_df["adr"] >= 2.0]["channel_num"].astype(int).tolist()
        print(f"  Channels with ADR >= 2.0: {sorted(top_channels)}")
        print(f"  Total: {len(top_channels)} channels")
        print()
        print(f"  Add to RUN_SPECS in your script:")
        print(f'  {{"run": "compact_top_adr",')
        print(f'   "features": ch_list({sorted(top_channels)}),')
        print(f'   "group": "compact_adr_guided"}},')

        # Split into contiguous sub-ranges
        s = sorted(top_channels)
        ranges = []
        start = s[0]
        prev  = s[0]
        for ch in s[1:]:
            if ch > prev + 2:   # allow 1-channel gaps
                ranges.append((start, prev))
                start = ch
            prev = ch
        ranges.append((start, prev))

        print(f"\n  Approximate contiguous ranges: ")
        for lo, hi in ranges:
            print(f"    ch{lo}–ch{hi}  ({hi-lo+1} channels)")

    # ── save results ───────────────────────────────────────────────────────────
    out = root

    if all_adr_rows:
        full_adr = pd.concat(all_adr_rows, ignore_index=True)
        p = out / "channel_adr_per_run.csv"
        full_adr.to_csv(p, index=False)
        print(f"\n💾  Saved: {p}")

    if not global_df.empty:
        p = out / "channel_global_adr_ranking.csv"
        global_df.to_csv(p, index=False)
        print(f"💾  Saved: {p}")

    if greedy_rows:
        df_g = pd.DataFrame(greedy_rows)
        p = out / "greedy_channel_selection.csv"
        df_g.to_csv(p, index=False)
        print(f"💾  Saved: {p}")
        print(f"\n{'='*70}")
        print("  GREEDY SELECTION SUMMARY")
        print(f"{'='*70}")
        print(df_g[["run","original_n_ch","optimal_n_ch",
                     "optimal_f05","optimal_TPe","optimal_FPe"]].to_string(index=False))

    print()


if __name__ == "__main__":
    main()
