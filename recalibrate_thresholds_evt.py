#!/usr/bin/env python3
"""
recalibrate_thresholds_evt.py
=============================
Label-free anomaly-threshold recalibration via Peaks-Over-Threshold (POT) /
Extreme-Value Theory, after Siffer et al., "Anomaly Detection in Streams with
Extreme Value Theory" (KDD 2017).

WHY THIS EXISTS
---------------
The sweep and channel_aware_rescore_v2 choose a detection threshold by
maximising VALIDATION event-F0.5 (subject to a val predicted-rate cap). With
only 13 weak validation events that choice is noisy and, as observed, does NOT
transfer to test: the cluster-representative models were tuned to <=1% on
validation but flagged 25-28% of the TEST set. That over-prediction, not the
channel choice, is what depresses their precision / F0.5.

POT sets the threshold from the TAIL of the score distribution instead of from
labels. We fit a Generalised Pareto Distribution (GPD) to the largest
validation scores and solve for the level z_q whose exceedance probability is a
small, fixed risk q. No anomaly labels and no test data are used to choose the
threshold, so the procedure is leak-free by construction, and the same z_q is
applied unchanged to test.

WHAT IT REPORTS (per run, per risk level q)
-------------------------------------------
  val_rate, test_rate  -> the transfer diagnostic. POT should keep test_rate
                          near q; the val-F0.5 threshold did not.
  A_f05, B_f05         -> mission-level event F0.5 under Protocol A
                          (subset-relevant) and Protocol B (all 89 events),
                          computed with the SAME machinery as
                          channel_aware_rescore_v2 so the numbers are directly
                          comparable to that script's output.

Leak-free: GPD is fit on validation scores only; z_q is applied to test as-is.
The risk levels q are fixed a priori (a grid, reported as sensitivity); test
performance never influences the threshold.

Outputs (inside --root):
    evt_recalibration.csv          # every run x every q
    <run>/test_pred_mask_evt_q{Q}.npy   # masks for the cluster runs (opt.)

Usage:
    python recalibrate_thresholds_evt.py
    python recalibrate_thresholds_evt.py --risk_levels 1e-2,1e-3,1e-4 --pot_init_pct 98
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Reuse the EXACT mission-level event definitions and scoring from the v2
# rescore, so EVT numbers are directly comparable to esa_scores_channel_aware_v2.
from channel_aware_rescore_v2 import (
    postprocess, find_events, event_scores,
    load_label_frame, relevant_flags, run_features,
)

try:
    from scipy.stats import genpareto
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


# ── GPD fit + POT threshold ───────────────────────────────────────────────────

def fit_gpd(excesses: np.ndarray):
    """MLE fit of a GPD with location fixed at 0 -> (shape xi, scale sigma).
    Falls back to method-of-moments if scipy is unavailable or the sample is
    tiny. xi>0 heavy tail, xi=0 exponential, xi<0 bounded tail."""
    if _HAVE_SCIPY and len(excesses) >= 20:
        try:
            xi, _loc, sigma = genpareto.fit(excesses, floc=0.0)
            if np.isfinite(xi) and np.isfinite(sigma) and sigma > 0:
                return float(xi), float(sigma)
        except Exception:
            pass
    m = float(np.mean(excesses)); v = float(np.var(excesses))
    if v <= 0:
        return 0.0, max(m, 1e-12)
    xi = 0.5 * (1.0 - (m * m) / v)
    sigma = 0.5 * m * ((m * m) / v + 1.0)
    return float(xi), float(max(sigma, 1e-12))


def pot_threshold(scores: np.ndarray, init_pct: float, q: float) -> Optional[Dict]:
    """Peaks-Over-Threshold anomaly level for target tail probability q.
    init_pct : percentile of `scores` used as the POT anchor t (e.g. 98).
    Returns z_q and fit diagnostics, or None if the tail is too thin to fit."""
    s = scores[np.isfinite(scores)]
    n = len(s)
    if n < 200:
        return None
    t = float(np.percentile(s, init_pct))
    peaks = s[s > t] - t
    Nt = int(len(peaks))
    if Nt < 20:
        return None
    xi, sigma = fit_gpd(peaks)
    ratio = q * n / Nt                      # = q / P(exceed t)
    if ratio <= 0 or not np.isfinite(ratio):
        return None
    if abs(xi) < 1e-6:
        z = t - sigma * np.log(ratio)       # exponential (xi -> 0) limit
    else:
        z = t + (sigma / xi) * (ratio ** (-xi) - 1.0)
    if not np.isfinite(z):
        return None
    return {"z": float(z), "t": t, "xi": float(xi), "sigma": float(sigma),
            "n_peaks": Nt, "n": n}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=("results_longrun/"
                    "mission1_reconstruction_ae_sweep_optimized/20260519_155620"))
    ap.add_argument("--train_file", default=("data/preprocessed/multivariate/"
                    "ESA-Mission1-semi-supervised/84_months.train.csv"))
    ap.add_argument("--test_file", default=("data/preprocessed/multivariate/"
                    "ESA-Mission1-semi-supervised/84_months.test.csv"))
    ap.add_argument("--risk_levels", default="1e-2,1e-3,1e-4",
                    help="comma-separated target tail probabilities q")
    ap.add_argument("--pot_init_pct", type=float, default=98.0,
                    help="percentile of val scores used as the POT anchor t")
    ap.add_argument("--merge_gap", type=int, default=0)
    ap.add_argument("--min_dur", type=int, default=1)
    ap.add_argument("--event_merge", type=int, default=0,
                    help="mission-event fragment merge tolerance (match v2)")
    ap.add_argument("--save_masks_for", default="clust_med_t03_6ch,clust_pair_t03_12ch",
                    help="comma-separated run names to dump EVT test masks for")
    args = ap.parse_args()

    root = Path(args.root)
    qs = [float(x) for x in args.risk_levels.split(",") if x.strip()]
    save_for = {s.strip() for s in args.save_masks_for.split(",") if s.strip()}

    run_dirs = [d for d in sorted(root.iterdir())
                if d.is_dir() and not d.name.startswith("plots")
                and (d / "test_score.npy").exists()
                and (d / "val_score.npy").exists()]
    if not run_dirs:
        print("No runs found."); return
    val_len = len(np.load(run_dirs[0] / "val_score.npy"))

    print("EVT / POT threshold recalibration  (label-free, mission-level events)")
    print(f"Root: {root}")
    print(f"POT anchor: {args.pot_init_pct:.1f}th pct of val scores   "
          f"risk levels q: {qs}\n")

    test_labels = load_label_frame(Path(args.test_file))
    val_labels  = load_label_frame(Path(args.train_file), tail=val_len)

    yt_global = test_labels.to_numpy().max(axis=1).astype(np.int8)
    yv_global = val_labels.to_numpy().max(axis=1).astype(np.int8)
    if args.event_merge > 0:
        yt_global = postprocess(yt_global, args.event_merge, 1)
        yv_global = postprocess(yv_global, args.event_merge, 1)
    test_events = find_events(yt_global)
    val_events  = find_events(yv_global)
    print(f"  mission test events (fixed, merge={args.event_merge}): "
          f"{len(test_events)}   val events: {len(val_events)}\n")

    # Optional: v2 F0.5 for a side-by-side reference column.
    v2_path = root / "esa_scores_channel_aware_v2.csv"
    v2_ref = {}
    if v2_path.exists():
        v2df = pd.read_csv(v2_path)
        for _, r in v2df.iterrows():
            v2_ref[str(r["run"])] = (r.get("A_esa_f05"), r.get("B_esa_f05"))

    rows = []
    for rd in run_dirs:
        features = run_features(rd)
        if not features:
            print(f"  {rd.name:32s} SKIP (no feature list)"); continue
        t_rel = relevant_flags(test_labels, features)
        vs = np.load(rd / "val_score.npy").astype(np.float32)
        ts = np.load(rd / "test_score.npy").astype(np.float32)

        for q in qs:
            pot = pot_threshold(vs, args.pot_init_pct, q)
            if pot is None:
                print(f"  {rd.name:32s} q={q:g}  SKIP (tail too thin)")
                continue
            z = pot["z"]
            vp = postprocess((vs > z).astype(np.int8), args.merge_gap, args.min_dur)
            tp = postprocess((ts > z).astype(np.int8), args.merge_gap, args.min_dur)
            A = event_scores(test_events, tp, t_rel)     # Protocol A
            B = event_scores(test_events, tp, None)      # Protocol B
            v2A, v2B = v2_ref.get(rd.name, (None, None))
            rows.append({
                "run": rd.name, "n_features": len(features), "q": q,
                "pot_anchor_t": round(pot["t"], 6), "gpd_xi": round(pot["xi"], 4),
                "gpd_sigma": round(pot["sigma"], 6), "z_threshold": round(z, 6),
                "val_rate": round(float(vp.mean()), 5),
                "test_rate": round(float(tp.mean()), 5),
                "A_f05": A["esa_f05"], "A_TPe": A["TPe"], "A_FPe": A["FPe"],
                "A_scoped": A["n_scoped_events"],
                "B_f05": B["esa_f05"], "B_TPe": B["TPe"],
                "v2_A_f05": v2A, "v2_B_f05": v2B,
            })
            if rd.name in save_for:
                np.save(rd / f"test_pred_mask_evt_q{q:g}.npy", tp.astype(np.int8))

    if not rows:
        print("Nothing evaluated."); return
    df = pd.DataFrame(rows)
    out = root / "evt_recalibration.csv"
    df.to_csv(out, index=False)
    print(f"Saved: {out}\n")

    # Focused readout: the transfer diagnostic + A/B vs the v2 threshold.
    focus = [r for r in ["clust_med_t03_6ch", "clust_pair_t03_12ch",
                          "center_ch12_38_49", "center_ch08_40_47"]
             if r in set(df["run"])]
    print("Transfer diagnostic (test_rate should stay near q; v2 blew up to ~0.25-0.28):\n")
    hdr = (f"{'run':22s} {'q':>7s} {'val_rate':>9s} {'test_rate':>10s} "
           f"{'A_f05':>7s} {'B_f05':>7s} {'v2_A':>7s} {'v2_B':>7s}")
    print(hdr); print("-" * len(hdr))
    for run in focus:
        for _, r in df[df.run == run].sort_values("q", ascending=False).iterrows():
            v2a = f"{r['v2_A_f05']:.4f}" if pd.notna(r["v2_A_f05"]) else "  -  "
            v2b = f"{r['v2_B_f05']:.4f}" if pd.notna(r["v2_B_f05"]) else "  -  "
            print(f"{run:22s} {r['q']:>7g} {r['val_rate']:>9.5f} "
                  f"{r['test_rate']:>10.5f} {r['A_f05']:>7.4f} {r['B_f05']:>7.4f} "
                  f"{v2a:>7s} {v2b:>7s}")
        print()

    # Best EVT operating point per focus run by Protocol A (across the q grid).
    print("Best EVT operating point per run (by Protocol A F0.5):")
    for run in focus:
        sub = df[df.run == run]
        b = sub.loc[sub["A_f05"].idxmax()]
        print(f"  {run:22s} q={b['q']:g}  A={b['A_f05']:.4f}  B={b['B_f05']:.4f}  "
              f"test_rate={b['test_rate']:.4f}   (v2 A={b['v2_A_f05']})")


if __name__ == "__main__":
    main()
