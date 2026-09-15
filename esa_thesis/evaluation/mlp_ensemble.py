#!/usr/bin/env python3
"""
evaluate_mlp_ensemble.py
=========================
Evaluates MLP and ensemble scores for all completed runs, and tests
cross-run OR/AND ensemble combinations of the best AE models.

Three things this script does:
  1. Threshold-sweeps mlp_val_score → finds best threshold → evaluates on test
  2. Threshold-sweeps ens_val_score → finds best threshold → evaluates on test
  3. Cross-run OR/AND ensembles using saved test_pred_mask.npy files

Run:
    python -m esa_thesis evaluate-ensemble
    python -m esa_thesis evaluate-ensemble --root results_longrun/mission1_reconstruction_ae_sweep_optimized/20260519_155620
"""

from esa_thesis.paths import project_path, default_run_dir

import argparse
from itertools import combinations
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


# ── metric core (same as esa_score.py) ────────────────────────────────────────

def find_events(y: np.ndarray) -> List[Tuple[int,int]]:
    y8 = (y > 0).astype(np.int8)
    if y8.sum() == 0:
        return []
    padded = np.zeros(len(y8) + 2, dtype=np.int8)
    padded[1:-1] = y8
    diff   = np.diff(padded.astype(np.int16))
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
    TPe = events_overlap_count(gt_ev,   pred_ev)
    FNe = len(gt_ev) - TPe
    FPe = len(pred_ev) - events_overlap_count(pred_ev, gt_ev)
    gt_b, pred_b = gt.astype(bool), pred.astype(bool)
    FPt = int((~gt_b &  pred_b).sum())
    Nt  = int((~gt_b).sum())
    fpt_penalty = FPt / Nt if Nt > 0 else 0.0
    prec_c = TPe / max(TPe + FPe + fpt_penalty, 1e-12)
    rec_e  = TPe / max(TPe + FNe, 1)
    b2     = 0.25
    denom  = b2 * prec_c + rec_e
    f05    = (1 + b2) * prec_c * rec_e / denom if denom > 0 else 0.0
    return {
        "esa_f05":     round(f05,    4),
        "precision_c": round(prec_c, 4),
        "recall_e":    round(rec_e,  4),
        "TPe": TPe, "FPe": FPe, "FNe": FNe,
        "gt_events": len(gt_ev),
        "pred_rate": round(float(pred.mean()), 4),
    }


# ── threshold sweep ────────────────────────────────────────────────────────────

def best_threshold(score: np.ndarray, gt: np.ndarray,
                   n_thresholds: int = 200,
                   max_pred_rate: float = 0.20) -> Tuple[float, float, np.ndarray]:
    """
    Sweep n_thresholds percentile-based thresholds on score+gt,
    return (best_threshold, best_f05, best_pred_mask).
    Only considers thresholds where pred_rate <= max_pred_rate.
    """
    percs  = np.linspace(80, 99.9, n_thresholds)
    thrs   = np.percentile(score, percs)
    thrs   = np.unique(thrs)

    best_f  = -1.0
    best_t  = float(thrs[-1])
    best_pm = (score > best_t).astype(np.int8)

    for t in thrs:
        pm = (score > t).astype(np.int8)
        if pm.mean() > max_pred_rate:
            continue
        if pm.mean() == 0:
            continue
        m = esa_f05(gt, pm)
        if m["esa_f05"] > best_f:
            best_f  = m["esa_f05"]
            best_t  = float(t)
            best_pm = pm
    return best_t, best_f, best_pm


# ── per-run evaluation ─────────────────────────────────────────────────────────

def evaluate_score_type(root: Path, score_type: str) -> pd.DataFrame:
    """
    score_type: 'mlp' or 'ens'
    Loads {score_type}_val_score.npy → sweeps thresholds → evaluates on test.
    """
    rows = []
    run_dirs = sorted(d for d in root.iterdir()
                      if d.is_dir() and not d.name.startswith("plots"))

    print(f"\n{'='*70}")
    print(f"  {score_type.upper()} SCORES — threshold sweep on val → evaluate on test")
    print(f"{'='*70}")

    for run_dir in run_dirs:
        val_score_path  = run_dir / f"{score_type}_val_score.npy"
        test_score_path = run_dir / f"{score_type}_test_score.npy"
        val_y_path      = run_dir / "val_y_true.npy"
        test_y_path     = run_dir / "test_y_true.npy"

        if not all(p.exists() for p in
                   [val_score_path, test_score_path, val_y_path, test_y_path]):
            continue

        val_score  = np.load(val_score_path).astype(np.float32)
        test_score = np.load(test_score_path).astype(np.float32)
        val_y      = np.load(val_y_path)
        test_y     = np.load(test_y_path)

        # find best threshold using validation set
        best_t, best_val_f, _ = best_threshold(val_score, val_y)

        # apply that threshold to test
        test_pm = (test_score > best_t).astype(np.int8)
        test_m  = esa_f05(test_y, test_pm)

        row = {"run": run_dir.name, "score_type": score_type,
               "best_threshold": round(best_t, 6),
               "val_esa_f05":  round(best_val_f, 4)}
        row.update({f"test_{k}": v for k, v in test_m.items()})
        rows.append(row)

        print(f"  {run_dir.name:35s} val={best_val_f:.4f}  "
              f"test={test_m['esa_f05']:.4f}  "
              f"TPe={test_m['TPe']}  FPe={test_m['FPe']}")

    if not rows:
        print(f"  No {score_type} score files found.")
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("test_esa_f05", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    return df


# ── cross-run OR / AND ensemble ────────────────────────────────────────────────

def cross_run_ensemble(root: Path, run_names: List[str],
                       logic: str = "OR") -> dict:
    """
    Combine test_pred_mask.npy from multiple runs using OR or AND logic.
    Returns metric dict.
    """
    masks = []
    for name in run_names:
        p = root / name / "test_pred_mask.npy"
        if p.exists():
            masks.append(np.load(p).astype(np.int8))

    if not masks:
        return {}

    # load ground truth from first available run
    gt = np.load(root / run_names[0] / "test_y_true.npy")

    if logic == "OR":
        combined = (np.stack(masks, axis=0).max(axis=0)).astype(np.int8)
    else:  # AND
        combined = (np.stack(masks, axis=0).min(axis=0)).astype(np.int8)

    m = esa_f05(gt, combined)
    m["run_names"]  = " | ".join(run_names)
    m["logic"]      = logic
    m["n_models"]   = len(masks)
    return m


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=
        str(default_run_dir()))
    args   = parser.parse_args()
    root   = Path(args.root)

    print(f"\nESA MLP + Ensemble Evaluator")
    print(f"Root: {root}")

    # ── 1. MLP per-run evaluation ─────────────────────────────────────────────
    df_mlp = evaluate_score_type(root, "mlp")

    # ── 2. Ensemble (AE+MLP) per-run evaluation ───────────────────────────────
    df_ens = evaluate_score_type(root, "ens")

    # ── 3. Cross-run OR/AND ensembles ─────────────────────────────────────────
    # Best runs identified from AE sweep (by test F0.5)
    top_runs = [
        "center_ch12_38_49",       # AE test 0.7548
        "corr_ch24_12_35",         # AE test 0.7357
        "corr_neg_ch30_12_40_p61", # AE test 0.7191
        "center_ch08_40_47",       # AE test 0.6965
        "combo_ch16_01_16",        # AE test 0.6875
        "corr_ch19_17_35",         # AE test 0.6793
        "corr_ch29_12_40",         # AE test 0.6793
        "block_ch08_17_24",        # AE test 0.6572
    ]

    print(f"\n{'='*70}")
    print("  CROSS-RUN ENSEMBLE — OR logic (flag if ANY model fires)")
    print(f"{'='*70}")
    print(f"  {'combination':55s}  {'F0.5':6s}  {'TPe':4s}  {'FPe':4s}  {'pred%':6s}")
    print(f"  {'-'*55}  {'-'*6}  {'-'*4}  {'-'*4}  {'-'*6}")

    or_rows = []
    # Try all combinations of 2, 3, and 4 top runs
    for n in [2, 3, 4, 5]:
        for combo in combinations(top_runs[:6], n):
            m = cross_run_ensemble(root, list(combo), logic="OR")
            if not m:
                continue
            label = " + ".join(c.replace("center_","c").replace("corr_","cr").replace("combo_","co")
                               .replace("block_","b").replace("fine_","f") for c in combo)
            print(f"  {label:55s}  {m['esa_f05']:.4f}  {m['TPe']:4d}  {m['FPe']:4d}  {m['pred_rate']:.3f}")
            or_rows.append({"combination": " | ".join(combo), "logic":"OR",
                            "n_models": n, **{k:v for k,v in m.items()
                            if k not in ["run_names","logic","n_models"]}})

    print(f"\n{'='*70}")
    print("  CROSS-RUN ENSEMBLE — AND logic (flag only if ALL models agree)")
    print(f"{'='*70}")
    print(f"  {'combination':55s}  {'F0.5':6s}  {'TPe':4s}  {'FPe':4s}  {'pred%':6s}")
    print(f"  {'-'*55}  {'-'*6}  {'-'*4}  {'-'*4}  {'-'*6}")

    and_rows = []
    for combo in combinations(top_runs[:5], 2):
        m = cross_run_ensemble(root, list(combo), logic="AND")
        if not m:
            continue
        label = " & ".join(c.replace("center_","c").replace("corr_","cr").replace("combo_","co")
                           .replace("block_","b") for c in combo)
        print(f"  {label:55s}  {m['esa_f05']:.4f}  {m['TPe']:4d}  {m['FPe']:4d}  {m['pred_rate']:.3f}")
        and_rows.append({"combination": " & ".join(combo), "logic":"AND",
                         "n_models": 2, **{k:v for k,v in m.items()
                         if k not in ["run_names","logic","n_models"]}})

    # ── 4. Save results ───────────────────────────────────────────────────────
    out = root

    if not df_mlp.empty:
        p = out / "mlp_evaluation_results.csv"
        df_mlp.to_csv(p, index=False)
        print(f"\nSaved: {p}")

    if not df_ens.empty:
        p = out / "ensemble_per_run_results.csv"
        df_ens.to_csv(p, index=False)
        print(f"Saved: {p}")

    if or_rows or and_rows:
        df_cross = pd.DataFrame(or_rows + and_rows).sort_values(
            "esa_f05", ascending=False).reset_index(drop=True)
        p = out / "cross_run_ensemble_results.csv"
        df_cross.to_csv(p, index=False)
        print(f"Saved: {p}")

    # ── 5. Final summary ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  FINAL SUMMARY — best result from each approach")
    print(f"{'='*70}")
    print(f"  {'approach':40s}  {'test F0.5':9s}  {'TPe':4s}  {'FPe':4s}")
    print(f"  {'-'*40}  {'-'*9}  {'-'*4}  {'-'*4}")
    print(f"  {'AE best (center_ch12_38_49)':40s}  {'0.7548':9s}  {'34':4s}  {'0':4s}")

    if not df_mlp.empty:
        b = df_mlp.iloc[0]
        print(f"  {'MLP best (' + b['run'] + ')':40s}  "
              f"{b['test_esa_f05']:.4f}     {int(b['test_TPe']):4d}  {int(b['test_FPe']):4d}")

    if not df_ens.empty:
        b = df_ens.iloc[0]
        print(f"  {'Ensemble best (' + b['run'] + ')':40s}  "
              f"{b['test_esa_f05']:.4f}     {int(b['test_TPe']):4d}  {int(b['test_FPe']):4d}")

    if or_rows:
        best_or = max(or_rows, key=lambda x: x["esa_f05"])
        label   = best_or["combination"].replace("center_ch12_38_49","c12")
        print(f"  {'OR ensemble best':40s}  {best_or['esa_f05']:.4f}     "
              f"{best_or['TPe']:4d}  {best_or['FPe']:4d}")

    if and_rows:
        best_and = max(and_rows, key=lambda x: x["esa_f05"])
        print(f"  {'AND ensemble best':40s}  {best_and['esa_f05']:.4f}     "
              f"{best_and['TPe']:4d}  {best_and['FPe']:4d}")

    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
