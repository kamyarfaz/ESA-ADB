#!/usr/bin/env python3
"""
recalibrate_thresholds_dspot.py
===============================
Drift-aware anomaly thresholding (DSPOT), after Siffer et al. 2017.

WHY (from the EVT diagnosis)
----------------------------
Static EVT-on-validation failed for two reasons:
  1. DEGENERATE VAL TAIL. The AE reconstructs validation almost perfectly, so
     ~98% of val scores are ~0 and the GPD fit is pathological (xi ~ 5-8). No
     usable tail exists on validation.
  2. VAL->TEST LEVEL SHIFT (concept drift). Test scores live at a higher, drifting
     level than validation, so any val-anchored threshold either floods or vanishes.

DSPOT addresses both. For each series it:
  * de-trends the score against a slow, robust ROLLING-MEDIAN baseline (trailing /
    causal), giving a residual r_t = score_t - baseline_t that is stationary even
    as the underlying level drifts. Short anomalies barely move a long trailing
    median (and the median is robust to them), so they stand out as positive
    residual spikes.
  * fits a GPD (POT) to the tail of the residual on an INITIAL CALIBRATION WINDOW
    of the stream itself (not on the degenerate val set), and solves for the level
    z_q whose residual-exceedance probability equals a fixed, a-priori risk q.
  * flags r_t > z_q over the full stream.

LEAK-FREE / HONEST SCOPE
------------------------
No anomaly LABELS are used to set the threshold. The tail is calibrated on the
initial portion of the test SCORES, which is the standard streaming-SPOT setup
and is unsupervised (this is a deployment-realistic, online assumption: the
detector calibrates on early stream and detects on the rest). Risk levels q are
fixed a priori and reported as a grid; test performance never selects q. Scoring
reuses the exact mission-level event machinery from channel_aware_rescore_v2, so
numbers are directly comparable to that script and to esa_scores_channel_aware_v2.

Outputs (inside --root):
    dspot_recalibration.csv
    <run>/test_pred_mask_dspot_q{Q}.npy   (for --save_masks_for runs)

Usage:
    python -m esa_thesis dspot
    python -m esa_thesis dspot --window 20000 --init_frac 0.2 --risk_levels 1e-2,1e-3,1e-4
"""

from esa_thesis.paths import project_path, default_run_dir

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from esa_thesis.evaluation.subset import (
    postprocess, find_events, event_scores,
    load_label_frame, relevant_flags, run_features,
)

try:
    from scipy.stats import genpareto
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


# ── GPD fit ───────────────────────────────────────────────────────────────────

def fit_gpd(excesses: np.ndarray):
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


def pot_level(calib: np.ndarray, init_pct: float, q: float) -> Optional[Dict]:
    """POT anomaly level for target exceedance q, fit on a calibration array."""
    s = calib[np.isfinite(calib)]
    n = len(s)
    if n < 200:
        return None
    t = float(np.percentile(s, init_pct))
    peaks = s[s > t] - t
    Nt = int(len(peaks))
    if Nt < 20:
        return None
    xi, sigma = fit_gpd(peaks)
    ratio = q * n / Nt
    if ratio <= 0 or not np.isfinite(ratio):
        return None
    if abs(xi) < 1e-6:
        z = t - sigma * np.log(ratio)
    else:
        z = t + (sigma / xi) * (ratio ** (-xi) - 1.0)
    if not np.isfinite(z):
        return None
    return {"z": float(z), "t": t, "xi": float(xi), "sigma": float(sigma), "n_peaks": Nt}


# ── drift baseline + DSPOT detection ──────────────────────────────────────────

def rolling_baseline(score: np.ndarray, window: int) -> np.ndarray:
    """Trailing (causal) rolling median: a slow, robust estimate of the drifting
    'normal' level. Trailing => no future leakage. Median => robust to the short
    anomaly spikes we want to keep in the residual."""
    s = pd.Series(score.astype(np.float64))
    base = s.rolling(window=window, min_periods=max(window // 10, 50)).median()
    return base.to_numpy()


def dspot_detect(score: np.ndarray, window: int, n_init: int,
                 init_pct: float, q: float) -> Optional[Dict]:
    base = rolling_baseline(score, window)
    resid = score - base
    resid[~np.isfinite(resid)] = 0.0
    # positive residuals only matter (score above local normal)
    n_init = int(min(max(n_init, 5 * window), len(resid)))
    calib = resid[:n_init]
    calib = calib[calib > 0]
    pot = pot_level(calib, init_pct, q)
    if pot is None:
        return None
    z = pot["z"]
    pred = (resid > z).astype(np.int8)
    return {"pred": pred, "z": z, "resid": resid, "baseline": base, **pot}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=(str(default_run_dir())))
    ap.add_argument("--train_file", default=(str(project_path('data/preprocessed/multivariate/ESA-Mission1-semi-supervised/84_months.train.csv'))))
    ap.add_argument("--test_file", default=(str(project_path('data/preprocessed/multivariate/ESA-Mission1-semi-supervised/84_months.test.csv'))))
    ap.add_argument("--risk_levels", default="1e-2,1e-3,1e-4")
    ap.add_argument("--window", type=int, default=20000,
                    help="trailing window (samples) for the drift baseline")
    ap.add_argument("--init_frac", type=float, default=0.2,
                    help="fraction of the stream used to calibrate the residual tail")
    ap.add_argument("--pot_init_pct", type=float, default=98.0)
    ap.add_argument("--merge_gap", type=int, default=0)
    ap.add_argument("--min_dur", type=int, default=1)
    ap.add_argument("--event_merge", type=int, default=0)
    ap.add_argument("--save_masks_for", default="clust_med_t03_6ch,clust_pair_t03_12ch")
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

    print("DSPOT drift-aware threshold recalibration  (label-free, mission-level events)")
    print(f"Root: {root}")
    print(f"drift window: {args.window}   calib frac: {args.init_frac}   "
          f"POT pct: {args.pot_init_pct}   q: {qs}\n")

    test_labels = load_label_frame(Path(args.test_file))
    yt_global = test_labels.to_numpy().max(axis=1).astype(np.int8)
    if args.event_merge > 0:
        yt_global = postprocess(yt_global, args.event_merge, 1)
    test_events = find_events(yt_global)
    print(f"  mission test events (fixed, merge={args.event_merge}): {len(test_events)}\n")

    v2_path = root / "esa_scores_channel_aware_v2.csv"
    v2_ref = {}
    if v2_path.exists():
        for _, r in pd.read_csv(v2_path).iterrows():
            v2_ref[str(r["run"])] = (r.get("A_esa_f05"), r.get("B_esa_f05"))

    rows = []
    for rd in run_dirs:
        features = run_features(rd)
        if not features:
            print(f"  {rd.name:32s} SKIP (no feature list)"); continue
        t_rel = relevant_flags(test_labels, features)
        ts = np.load(rd / "test_score.npy").astype(np.float32)
        n_init = int(args.init_frac * len(ts))

        for q in qs:
            det = dspot_detect(ts, args.window, n_init, args.pot_init_pct, q)
            if det is None:
                print(f"  {rd.name:32s} q={q:g}  SKIP (tail too thin)"); continue
            tp = postprocess(det["pred"], args.merge_gap, args.min_dur)
            A = event_scores(test_events, tp, t_rel)
            B = event_scores(test_events, tp, None)
            v2A, v2B = v2_ref.get(rd.name, (None, None))
            rows.append({
                "run": rd.name, "n_features": len(features), "q": q,
                "drift_window": args.window, "gpd_xi": round(det["xi"], 4),
                "resid_z": round(det["z"], 6),
                "test_rate": round(float(tp.mean()), 5),
                "A_f05": A["esa_f05"], "A_TPe": A["TPe"], "A_FPe": A["FPe"],
                "A_scoped": A["n_scoped_events"],
                "B_f05": B["esa_f05"], "B_TPe": B["TPe"],
                "v2_A_f05": v2A, "v2_B_f05": v2B,
            })
            if rd.name in save_for:
                np.save(rd / f"test_pred_mask_dspot_q{q:g}.npy", tp.astype(np.int8))

    if not rows:
        print("Nothing evaluated."); return
    df = pd.DataFrame(rows)
    out = root / "dspot_recalibration.csv"
    df.to_csv(out, index=False)
    print(f"Saved: {out}\n")

    focus = [r for r in ["clust_med_t03_6ch", "clust_pair_t03_12ch",
                          "center_ch12_38_49", "center_ch08_40_47"]
             if r in set(df["run"])]
    hdr = (f"{'run':22s} {'q':>7s} {'gpd_xi':>7s} {'test_rate':>10s} "
           f"{'A_f05':>7s} {'B_f05':>7s} {'v2_A':>7s} {'v2_B':>7s}")
    print("DSPOT vs v2 (drift-corrected; gpd_xi should now be sane, <~1):\n")
    print(hdr); print("-" * len(hdr))
    for run in focus:
        for _, r in df[df.run == run].sort_values("q", ascending=False).iterrows():
            v2a = f"{r['v2_A_f05']:.4f}" if pd.notna(r["v2_A_f05"]) else "  -  "
            v2b = f"{r['v2_B_f05']:.4f}" if pd.notna(r["v2_B_f05"]) else "  -  "
            print(f"{run:22s} {r['q']:>7g} {r['gpd_xi']:>7.3f} {r['test_rate']:>10.5f} "
                  f"{r['A_f05']:>7.4f} {r['B_f05']:>7.4f} {v2a:>7s} {v2b:>7s}")
        print()

    print("Best DSPOT operating point per run (by Protocol A F0.5, diagnostic only):")
    for run in focus:
        sub = df[df.run == run]
        b = sub.loc[sub["A_f05"].idxmax()]
        print(f"  {run:22s} q={b['q']:g}  A={b['A_f05']:.4f}  B={b['B_f05']:.4f}  "
              f"test_rate={b['test_rate']:.4f}   (v2 A={b['v2_A_f05']})")


if __name__ == "__main__":
    main()
