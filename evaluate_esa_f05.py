#!/usr/bin/env python3
"""
evaluate_esa_f05.py
====================
Computes the official ESA corrected event-wise F0.5 score for every completed
run found under results_longrun/.

Formula (from ESA-ADB paper, Sehili & Zhang correction):
    Precision_ecorr = TPe / (TPe + FPe + FPt/Nt)
    Recall_e        = TPe / (TPe + FNe)
    F0.5            = (1 + 0.5²) * Precision_ecorr * Recall_e
                      / (0.5² * Precision_ecorr + Recall_e)

Where:
    TPe  = true positive events  (GT events that overlap ≥1 pred segment)
    FPe  = false positive events (pred segments with no overlap with any GT event)
    FNe  = false negative events (GT events with no overlap with any pred segment)
    FPt  = total timesteps falsely flagged as anomalous
    Nt   = total nominal (non-anomalous) timesteps in the test set

Inputs per run (loaded from run folder):
    test_pred_mask.npy  — binary prediction array (int8 / bool), shape (T,)
    test_y_true.npy     — binary ground truth array (int8 / bool), shape (T,)

Usage:
    python evaluate_esa_f05.py
    python evaluate_esa_f05.py --results_root results_longrun/mission1_reconstruction_ae_sweep_optimized/20260519_155620
    python evaluate_esa_f05.py --top 10
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core metric helpers  (identical logic to mission1_reconstruction_ae_sweep_optimized.py)
# ---------------------------------------------------------------------------

def find_events(y: np.ndarray) -> List[Tuple[int, int]]:
    """Return list of (start, end) inclusive for each contiguous run of 1s."""
    y8 = (y > 0).astype(np.int8)
    if y8.sum() == 0:
        return []
    padded      = np.empty(len(y8) + 2, dtype=np.int8)
    padded[0]   = 0
    padded[1:-1] = y8
    padded[-1]  = 0
    diff   = np.diff(padded.astype(np.int16))
    starts = np.where(diff ==  1)[0]          # rising edges
    ends   = np.where(diff == -1)[0] - 1      # falling edges (inclusive)
    return list(zip(starts.tolist(), ends.tolist()))


def count_detected(query_events: List[Tuple[int,int]],
                   ref_events:   List[Tuple[int,int]]) -> int:
    """How many query events overlap ≥1 ref event — O(n log n) sweep."""
    if not ref_events or not query_events:
        return 0
    ref_arr  = np.array(ref_events, dtype=np.int64)
    ref_sort = ref_arr[np.argsort(ref_arr[:, 0])]
    detected = 0
    for qs, qe in query_events:
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
                detected += 1
                break
    return detected


def fbeta(p: float, r: float, beta: float = 0.5) -> float:
    b2  = beta * beta
    den = b2 * p + r
    return 0.0 if den <= 0 else (1 + b2) * p * r / den


def esa_corrected_f05(gt: np.ndarray, pred: np.ndarray) -> Dict:
    """
    Compute the full ESA corrected event-wise F0.5 and supporting metrics.

    Parameters
    ----------
    gt   : binary array, ground truth  (0=normal, 1=anomaly)
    pred : binary array, prediction    (0=normal, 1=anomaly)

    Returns
    -------
    dict with all metric fields
    """
    gt   = (gt   > 0).astype(np.int8)
    pred = (pred > 0).astype(np.int8)

    gt_ev = find_events(gt)
    pr_ev = find_events(pred)

    # ---- Event-level counts ------------------------------------------------
    TPe = count_detected(gt_ev, pr_ev)    # GT events caught by ≥1 pred
    FNe = len(gt_ev) - TPe                # GT events missed entirely
    FPe = count_detected(pr_ev, gt_ev)    # pred events that overlap GT  (for FP we need complement)
    FPe = len(pr_ev) - FPe               # pred events with no GT overlap

    # Simple event precision / recall (uncorrected)
    ep_simple = TPe / max(TPe + FPe, 1)
    er        = TPe / max(TPe + FNe, 1)

    # ---- Point-level counts ------------------------------------------------
    gt_b   = gt.astype(bool)
    pred_b = pred.astype(bool)
    TPt = int(( gt_b &  pred_b).sum())
    FPt = int((~gt_b &  pred_b).sum())   # false positive timesteps
    TNt = int((~gt_b & ~pred_b).sum())
    FNt = int(( gt_b & ~pred_b).sum())

    pp = TPt / max(TPt + FPt, 1)   # point precision
    pr = TPt / max(TPt + FNt, 1)   # point recall

    # ---- ESA corrected precision -------------------------------------------
    # Penalises false alarms proportionally to their duration relative to
    # the total nominal signal length (Sehili & Zhang correction)
    Nt          = int((~gt_b).sum())
    fpt_penalty = FPt / Nt if Nt > 0 else 0.0
    precision_c = TPe / max(TPe + FPe + fpt_penalty, 1e-12)

    # ---- F-scores ----------------------------------------------------------
    esa_f05     = fbeta(precision_c, er,        beta=0.5)
    event_f05   = fbeta(ep_simple,   er,        beta=0.5)
    event_f1    = fbeta(ep_simple,   er,        beta=1.0)
    point_f05   = fbeta(pp,          pr,        beta=0.5)
    point_f1    = fbeta(pp,          pr,        beta=1.0)

    pred_rate   = float(pred.mean())
    true_rate   = float(gt.mean())

    return {
        # Primary metric
        "esa_f05":              round(esa_f05,     6),
        "esa_precision_corr":   round(precision_c, 6),
        "esa_recall_e":         round(er,          6),
        # Event metrics (simple, uncorrected)
        "event_f05":            round(event_f05,   6),
        "event_f1":             round(event_f1,    6),
        "event_precision":      round(ep_simple,   6),
        "event_recall":         round(er,          6),
        "TPe": TPe, "FPe": FPe, "FNe": FNe,
        "num_gt_events":        len(gt_ev),
        "num_pred_events":      len(pr_ev),
        # Point metrics
        "point_f05":            round(point_f05, 6),
        "point_f1":             round(point_f1,  6),
        "point_precision":      round(pp,        6),
        "point_recall":         round(pr,        6),
        "TPt": TPt, "FPt": FPt, "TNt": TNt, "FNt": FNt,
        # Rates
        "pred_anomaly_rate":    round(pred_rate, 6),
        "true_anomaly_rate":    round(true_rate, 6),
        # Correction details
        "Nt": Nt, "FPt_penalty": round(fpt_penalty, 6),
    }


# ---------------------------------------------------------------------------
# Run discovery and evaluation
# ---------------------------------------------------------------------------

def find_completed_runs(root: Path) -> List[Path]:
    """Return sorted list of run folders that have both required .npy files."""
    completed = []
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        if (run_dir / "test_pred_mask.npy").exists() and \
           (run_dir / "test_y_true.npy").exists():
            completed.append(run_dir)
    return completed


def load_run_info(run_dir: Path) -> Dict:
    """Load optional summary.json for extra info (best epoch, val ESA etc.)."""
    info = {}
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        try:
            with open(summary_path) as f:
                s = json.load(f)
            # Pull out the best clean-val ESA threshold result if present
            best = s.get("best_clean_val_esa", {})
            info["val_esa_f05"]   = best.get("val_esa_f05",   None)
            info["best_epoch"]    = s.get("best_epoch",        None)
            info["best_val_esa"]  = s.get("best_val_esa",      None)
            info["threshold"]     = best.get("threshold",      None)
            info["merge_gap"]     = best.get("merge_gap",      None)
            info["min_dur"]       = best.get("min_dur",        None)
        except Exception:
            pass
    return info


def evaluate_all_runs(root: Path) -> pd.DataFrame:
    run_dirs = find_completed_runs(root)
    if not run_dirs:
        print(f"[ERROR] No completed runs found under {root}")
        print("        Expected: test_pred_mask.npy + test_y_true.npy in each run folder")
        return pd.DataFrame()

    print(f"Found {len(run_dirs)} completed run(s) under {root}\n")
    rows = []

    for run_dir in run_dirs:
        run_name = run_dir.name
        print(f"  Evaluating: {run_name} ... ", end="", flush=True)

        try:
            pred = np.load(run_dir / "test_pred_mask.npy")
            gt   = np.load(run_dir / "test_y_true.npy")
        except Exception as e:
            print(f"SKIP ({e})")
            continue

        m    = esa_corrected_f05(gt, pred)
        info = load_run_info(run_dir)

        row = {"run": run_name}
        row.update(info)
        row.update(m)
        rows.append(row)

        print(f"ESA F0.5={m['esa_f05']:.4f}  "
              f"Prec_c={m['esa_precision_corr']:.4f}  "
              f"Rec_e={m['esa_recall_e']:.4f}  "
              f"TPe={m['TPe']}  FPe={m['FPe']}  FNe={m['FNe']}")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values("esa_f05", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    return df


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------

DISPLAY_COLS = [
    "rank", "run",
    "esa_f05", "esa_precision_corr", "esa_recall_e",
    "event_f05", "event_precision", "event_recall",
    "point_f05",
    "TPe", "FPe", "FNe", "num_gt_events",
    "FPt", "pred_anomaly_rate",
    "val_esa_f05", "best_epoch",
]


def print_table(df: pd.DataFrame, top: int = None) -> None:
    if df.empty:
        return
    show = df.head(top) if top else df
    # Select only columns that exist
    cols = [c for c in DISPLAY_COLS if c in show.columns]
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width",       220)
    pd.set_option("display.float_format", "{:.4f}".format)
    print("\n" + "=" * 120)
    print("ESA CORRECTED EVENT-WISE F0.5 — RANKED RESULTS")
    print("=" * 120)
    print(show[cols].to_string(index=False))
    print("=" * 120)

    # Key observations
    best = df.iloc[0]
    print(f"\n🏆  Best run : {best['run']}")
    print(f"    ESA F0.5         : {best['esa_f05']:.4f}")
    print(f"    Precision (corr) : {best['esa_precision_corr']:.4f}")
    print(f"    Recall (event)   : {best['esa_recall_e']:.4f}")
    print(f"    TPe / FPe / FNe  : {int(best['TPe'])} / {int(best['FPe'])} / {int(best['FNe'])}")
    print(f"    GT events total  : {int(best['num_gt_events'])}")
    print(f"    FPt (false alarm timesteps): {int(best['FPt'])}")

    # Val → test gap
    if "val_esa_f05" in df.columns:
        df2 = df.copy()
        df2["val_test_gap"] = df2["val_esa_f05"].astype(float) - df2["esa_f05"].astype(float)
        best_gen = df2.nsmallest(1, "val_test_gap").iloc[0]
        print(f"\n📐  Best generalisation (smallest val→test gap): {best_gen['run']}  "
              f"gap={best_gen['val_test_gap']:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ESA corrected F0.5 for all completed sweep runs"
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default="results_longrun/mission1_reconstruction_ae_sweep_optimized/20260519_155620",
        help="Path to the timestamped results folder containing run subdirs",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Only display top-N runs (default: all)",
    )
    parser.add_argument(
        "--save_csv",
        type=str,
        default="esa_evaluation_results.csv",
        help="Output CSV filename (default: esa_evaluation_results.csv)",
    )
    args = parser.parse_args()

    root = Path(args.results_root)
    if not root.exists():
        print(f"[ERROR] results_root not found: {root}")
        return

    print(f"\nESA Corrected Event-wise F0.5 Evaluator")
    print(f"Results root : {root}")
    print(f"Formula      : Precision_ecorr = TPe / (TPe + FPe + FPt/Nt)")
    print(f"             : Recall_e        = TPe / (TPe + FNe)")
    print(f"             : F0.5            = 1.25 * P_c * R_e / (0.25*P_c + R_e)\n")

    df = evaluate_all_runs(root)
    if df.empty:
        return

    print_table(df, top=args.top)

    # Save full results
    out_path = root / args.save_csv
    df.to_csv(out_path, index=False)
    print(f"\n💾  Full results saved to: {out_path}")

    # Also print a compact summary sorted by group
    if "run" in df.columns:
        print("\n--- Summary by run group ---")
        df["group"] = df["run"].str.extract(r"^([a-z]+)_")
        for grp, gdf in df.groupby("group", sort=True):
            best_in_grp = gdf.iloc[0]
            print(f"  {grp:12s}  best={best_in_grp['run']:30s}  "
                  f"ESA_F05={best_in_grp['esa_f05']:.4f}  "
                  f"TPe={int(best_in_grp['TPe'])}  FPe={int(best_in_grp['FPe'])}")


if __name__ == "__main__":
    main()
