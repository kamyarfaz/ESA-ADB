#!/usr/bin/env python3
"""
esa_score.py
============
Computes the official ESA corrected event-wise F0.5 for every completed run.
Evaluates val and test sets separately, one row per run.

Official formula (ESA-ADB paper / Kaggle page):
    Precision_ecorr = TPe / (TPe + FPe + FPt/Nt)
    Recall_e        = TPe / (TPe + FNe)
    F0.5            = (1 + 0.5²) × Precision_ecorr × Recall_e
                      ─────────────────────────────────────────
                       0.5² × Precision_ecorr + Recall_e

    TPe = GT events detected by ≥1 predicted segment
    FPe = predicted segments that overlap no GT event
    FNe = GT events missed entirely
    FPt = total false-positive timesteps
    Nt  = total nominal (non-anomalous) timesteps

Usage:
    python esa_score.py
    python esa_score.py --root results_longrun/mission1_reconstruction_ae_sweep_optimized/20260519_155620
    python esa_score.py --split val      # only validation
    python esa_score.py --split test     # only test
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


# ── metric core ────────────────────────────────────────────────────────────────

def find_events(y: np.ndarray) -> List[Tuple[int, int]]:
    """Contiguous runs of 1s → list of (start, end) inclusive."""
    y8 = (y > 0).astype(np.int8)
    if y8.sum() == 0:
        return []
    padded = np.zeros(len(y8) + 2, dtype=np.int8)
    padded[1:-1] = y8
    diff = np.diff(padded.astype(np.int16))
    starts = np.where(diff ==  1)[0]
    ends   = np.where(diff == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))


def events_overlap_count(query: List[Tuple[int,int]],
                         ref:   List[Tuple[int,int]]) -> int:
    """Count query events that overlap ≥1 ref event (O(n log n))."""
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
    """
    Compute ESA corrected event-wise F0.5.
    gt, pred: 1-D binary arrays (0=normal, 1=anomaly).
    """
    gt   = (gt   > 0).astype(np.int8)
    pred = (pred > 0).astype(np.int8)

    gt_events   = find_events(gt)
    pred_events = find_events(pred)

    TPe = events_overlap_count(gt_events,   pred_events)   # GT events caught
    FNe = len(gt_events) - TPe                              # GT events missed
    FPe = len(pred_events) - events_overlap_count(pred_events, gt_events)  # spurious pred events

    gt_bool   = gt.astype(bool)
    pred_bool = pred.astype(bool)
    FPt = int((~gt_bool & pred_bool).sum())   # false-positive timesteps
    Nt  = int((~gt_bool).sum())               # total nominal timesteps

    # corrected precision
    fpt_penalty = FPt / Nt if Nt > 0 else 0.0
    prec_c = TPe / max(TPe + FPe + fpt_penalty, 1e-12)

    # recall
    rec_e = TPe / max(TPe + FNe, 1)

    # F0.5
    b2    = 0.25                               # 0.5²
    denom = b2 * prec_c + rec_e
    f05   = (1 + b2) * prec_c * rec_e / denom if denom > 0 else 0.0

    return {
        "esa_f05":      round(f05,    4),
        "precision_c":  round(prec_c, 4),
        "recall_e":     round(rec_e,  4),
        "TPe":  TPe,
        "FPe":  FPe,
        "FNe":  FNe,
        "FPt":  FPt,
        "Nt":   Nt,
        "gt_events":   len(gt_events),
        "pred_events": len(pred_events),
        "pred_rate":   round(float(pred.mean()), 4),
    }


# ── run discovery ──────────────────────────────────────────────────────────────

def score_split(run_dir: Path, split: str):
    """Load pred_mask + y_true for a split and return metric dict, or None."""
    pred_path = run_dir / f"{split}_pred_mask.npy"
    gt_path   = run_dir / f"{split}_y_true.npy"
    if not pred_path.exists() or not gt_path.exists():
        return None
    pred = np.load(pred_path)
    gt   = np.load(gt_path)
    return esa_f05(gt, pred)


def evaluate(root: Path, splits: List[str]) -> pd.DataFrame:
    run_dirs = sorted(
        d for d in root.iterdir()
        if d.is_dir() and not d.name.startswith("plots")
    )
    if not run_dirs:
        print(f"No run folders found under {root}")
        return pd.DataFrame()

    rows = []
    for run_dir in run_dirs:
        # check at least one split exists
        has_any = any(
            (run_dir / f"{s}_pred_mask.npy").exists() for s in splits
        )
        if not has_any:
            continue

        row = {"run": run_dir.name}
        for split in splits:
            m = score_split(run_dir, split)
            if m is None:
                for k in ["esa_f05","precision_c","recall_e",
                           "TPe","FPe","FNe","pred_rate"]:
                    row[f"{split}_{k}"] = None
            else:
                for k, v in m.items():
                    row[f"{split}_{k}"] = v
        rows.append(row)

    return pd.DataFrame(rows)


# ── display ────────────────────────────────────────────────────────────────────

def print_results(df: pd.DataFrame, splits: List[str]) -> None:
    if df.empty:
        print("No results to display.")
        return

    # sort by first available esa_f05 column
    sort_col = f"{splits[0]}_esa_f05"
    if sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width",        200)
    pd.set_option("display.float_format", "{:.4f}".format)

    for split in splits:
        cols = ["rank", "run"] + [
            f"{split}_{k}" for k in
            ["esa_f05", "precision_c", "recall_e",
             "TPe", "FPe", "FNe", "gt_events", "pred_rate"]
            if f"{split}_{k}" in df.columns
        ]
        sub = df[cols].dropna(subset=[f"{split}_esa_f05"])

        print(f"\n{'='*100}")
        print(f"  {split.upper()} SET — ESA Corrected Event-wise F0.5  "
              f"(sorted by {split} ESA F0.5 descending)")
        print(f"{'='*100}")
        print(sub.to_string(index=False))

        best = sub[sub[f"{split}_esa_f05"] == sub[f"{split}_esa_f05"].max()].iloc[0]
        print(f"\n  🏆  Best: {best['run']}  "
              f"F0.5={best[f'{split}_esa_f05']:.4f}  "
              f"Prec_c={best[f'{split}_precision_c']:.4f}  "
              f"Rec_e={best[f'{split}_recall_e']:.4f}  "
              f"TPe={int(best[f'{split}_TPe'])}  "
              f"FPe={int(best[f'{split}_FPe'])}  "
              f"FNe={int(best[f'{split}_FNe'])}")

    # side-by-side val vs test if both present
    if "val" in splits and "test" in splits:
        if "val_esa_f05" in df.columns and "test_esa_f05" in df.columns:
            df2 = df[["rank","run","val_esa_f05","test_esa_f05"]].copy()
            df2["gap (val-test)"] = (
                df2["val_esa_f05"].astype(float) -
                df2["test_esa_f05"].astype(float)
            ).round(4)
            print(f"\n{'='*100}")
            print("  VAL vs TEST COMPARISON")
            print(f"{'='*100}")
            print(df2.dropna().to_string(index=False))


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ESA corrected event-wise F0.5 — per-run evaluation"
    )
    parser.add_argument(
        "--root",
        default="results_longrun/mission1_reconstruction_ae_sweep_optimized/20260519_155620",
        help="Timestamped results folder containing run subdirs",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test", "both"],
        default="both",
        help="Which split to evaluate (default: both)",
    )
    parser.add_argument(
        "--save",
        default="esa_scores.csv",
        help="Output CSV filename (saved inside --root)",
    )
    args = parser.parse_args()

    root   = Path(args.root)
    splits = ["val", "test"] if args.split == "both" else [args.split]

    print(f"\nESA Corrected Event-wise F0.5 Evaluator")
    print(f"Root   : {root}")
    print(f"Splits : {splits}")
    print(f"Formula: Precision_ecorr = TPe / (TPe + FPe + FPt/Nt)")
    print(f"         Recall_e        = TPe / (TPe + FNe)")
    print(f"         F0.5            = 1.25 × P_c × R_e / (0.25×P_c + R_e)")

    df = evaluate(root, splits)
    print_results(df, splits)

    out = root / args.save
    df.to_csv(out, index=False)
    print(f"\n💾  Saved: {out}\n")


if __name__ == "__main__":
    main()
