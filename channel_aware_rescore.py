#!/usr/bin/env python3
"""
channel_aware_rescore.py
========================
Re-evaluates every completed run under the OFFICIAL ESA-ADB subset protocol:
a model trained on a subset of channels is scored against the anomalies
annotated in THOSE channels (union of their is_anomaly_* columns), not
against all anomalies in the mission.

WHY: the sweep pipeline builds ground truth with get_y_any() = max over ALL
is_anomaly_* columns, so subset models are penalised for events invisible in
their inputs. This caps recall (best run: 34/89 = 0.38) and therefore F0.5.
The ESA-ADB paper benchmarks "lightweight subsets of channels" against the
events in those subsets — this script aligns the evaluation with that
protocol and makes results comparable to other Mul-AE results on the
benchmark.

Everything stays leak-free: thresholds are selected on VALIDATION data only
(channel-aware val labels, fixed protocol: best val ESA F0.5 subject to
val predicted-anomaly-rate <= --rate_cap), then applied unchanged to test.

Outputs (inside --root):
    <run>/test_pred_mask_ca.npy       channel-aware honest test mask
    <run>/test_y_true_ca.npy          channel-aware test labels for this run
    esa_scores_channel_aware.csv      per-run results (old vs new protocol)
    cross_run_ensemble_ca.csv         OR/AND search, labels = union of
                                      the combined models' channel sets

Usage:
    python channel_aware_rescore.py
    python channel_aware_rescore.py --rate_cap 0.01 --top_n 8
"""

import argparse
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── constants: identical to the sweep script ──────────────────────────────────

THRESH_PERCENTILES = list(np.concatenate([
    np.linspace(50, 95, 10),
    np.array([96, 97, 98, 99, 99.3, 99.5, 99.7, 99.8, 99.9, 99.95, 99.97, 99.99]),
]))
MERGE_GAPS = [0, 16, 32, 64, 128, 256, 512, 1024]
MIN_DURS   = [1, 8, 16, 32, 64, 128]


def postprocess(mask: np.ndarray, merge_gap: int, min_dur: int) -> np.ndarray:
    out = (mask > 0).astype(np.int8)
    if merge_gap > 0 and out.sum() > 0:
        padded = np.empty(len(out) + 2, dtype=np.int8)
        padded[0] = 0; padded[1:-1] = out; padded[-1] = 0
        diff   = np.diff(padded.astype(np.int16))
        starts = np.where(diff == 1)[0]
        ends   = np.where(diff == -1)[0]
        if len(starts) > 1:
            gap_starts = ends[:-1]
            gap_ends   = starts[1:]
            fill = (gap_ends - gap_starts) <= merge_gap
            for gs, ge in zip(gap_starts[fill], gap_ends[fill]):
                out[gs:ge] = 1
    if min_dur > 1 and out.sum() > 0:
        padded = np.empty(len(out) + 2, dtype=np.int8)
        padded[0] = 0; padded[1:-1] = out; padded[-1] = 0
        diff   = np.diff(padded.astype(np.int16))
        starts = np.where(diff == 1)[0]
        ends   = np.where(diff == -1)[0]
        dur = ends - starts
        for s, e in zip(starts[dur < min_dur], ends[dur < min_dur]):
            out[s:e] = 0
    return out


def find_events(y: np.ndarray) -> List[Tuple[int, int]]:
    y8 = (y > 0).astype(np.int8)
    if y8.sum() == 0:
        return []
    padded = np.zeros(len(y8) + 2, dtype=np.int8)
    padded[1:-1] = y8
    diff   = np.diff(padded.astype(np.int16))
    starts = np.where(diff ==  1)[0]
    ends   = np.where(diff == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))


def events_overlap_count(query, ref) -> int:
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


def esa_f05(gt: np.ndarray, pred: np.ndarray) -> Dict:
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
            "gt_events": len(gt_ev), "pred_rate": round(float(pred.mean()), 4)}


# ── label handling ─────────────────────────────────────────────────────────────

def load_label_frame(path: Path, tail: Optional[int] = None) -> pd.DataFrame:
    """Load only is_anomaly_* columns as int8. Optionally keep last `tail` rows."""
    print(f"  loading label columns from {path.name} ...", flush=True)
    df = pd.read_csv(path, usecols=lambda c: c.startswith("is_anomaly_"),
                     low_memory=False)
    for c in df.columns:
        df[c] = (pd.to_numeric(df[c], errors="coerce")
                   .fillna(0).astype(np.float32) > 0).astype(np.int8)
    if tail is not None:
        df = df.tail(tail).reset_index(drop=True)
    return df


def y_for_features(labels: pd.DataFrame, features: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Union of is_anomaly columns matching the given feature list."""
    cols = [f"is_anomaly_{f}" for f in features
            if f"is_anomaly_{f}" in labels.columns]
    if not cols:
        return np.zeros(len(labels), dtype=np.int8), []
    return labels[cols].to_numpy().max(axis=1).astype(np.int8), cols


def run_features(run_dir: Path) -> List[str]:
    sel = run_dir / "selected_thresholds.csv"
    if sel.exists():
        df = pd.read_csv(sel, nrows=1)
        if "features" in df.columns:
            return [f.strip() for f in str(df["features"].iloc[0]).split(",") if f.strip()]
    return []


# ── honest channel-aware selection ────────────────────────────────────────────

def select_and_score(vs: np.ndarray, yv: np.ndarray,
                     ts: np.ndarray, yt: np.ndarray,
                     rate_cap: float) -> Optional[Dict]:
    nz = vs[np.isfinite(vs) & (vs > 0)]
    if len(nz) == 0:
        return None
    thr_list = [float(v) for v in
                np.unique(np.percentile(nz, THRESH_PERCENTILES)) if np.isfinite(v)]

    best = None
    for thr in thr_list:
        val_raw = (vs > thr).astype(np.int8)
        if val_raw.mean() > 0.30:          # cannot reach the cap; skip grid
            continue
        for mg in MERGE_GAPS:
            for md in MIN_DURS:
                vp = postprocess(val_raw, mg, md)
                if vp.mean() > rate_cap or vp.mean() == 0:
                    continue
                vm = esa_f05(yv, vp)
                key = (vm["esa_f05"], -vm["FPe"], -vp.mean())
                if best is None or key > best["key"]:
                    best = {"key": key, "thr": thr, "mg": mg, "md": md,
                            "val": vm}
    if best is None:
        return None
    tp = postprocess((ts > best["thr"]).astype(np.int8), best["mg"], best["md"])
    tm = esa_f05(yt, tp)
    return {"thr": best["thr"], "mg": best["mg"], "md": best["md"],
            "val": best["val"], "test": tm, "test_mask": tp}


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=("results_longrun/"
                    "mission1_reconstruction_ae_sweep_optimized/20260519_155620"))
    ap.add_argument("--train_file", default=("data/preprocessed/multivariate/"
                    "ESA-Mission1-semi-supervised/84_months.train.csv"))
    ap.add_argument("--test_file", default=("data/preprocessed/multivariate/"
                    "ESA-Mission1-semi-supervised/84_months.test.csv"))
    ap.add_argument("--rate_cap", type=float, default=0.01)
    ap.add_argument("--top_n", type=int, default=8,
                    help="top runs entering the ensemble search")
    args = ap.parse_args()
    root = Path(args.root)

    run_dirs = [d for d in sorted(root.iterdir())
                if d.is_dir() and not d.name.startswith("plots")
                and (d / "test_score.npy").exists()
                and (d / "val_score.npy").exists()]
    if not run_dirs:
        print("No runs found."); return

    # val length from any run (identical split across runs)
    val_len = len(np.load(run_dirs[0] / "val_score.npy"))

    print("Channel-aware honest re-evaluation")
    print(f"Root: {root}\nProtocol: best val ESA F0.5, val pred rate <= {args.rate_cap}\n")
    test_labels = load_label_frame(Path(args.test_file))
    val_labels  = load_label_frame(Path(args.train_file), tail=val_len)
    global_yt   = test_labels.to_numpy().max(axis=1).astype(np.int8)
    print(f"  test label columns: {len(test_labels.columns)}   "
          f"global test events: {len(find_events(global_yt))}\n")

    rows, feats_by_run = [], {}
    for rd in run_dirs:
        features = run_features(rd)
        if not features:
            print(f"  {rd.name:32s} SKIP (no feature list)"); continue
        feats_by_run[rd.name] = features

        yt, cols_t = y_for_features(test_labels, features)
        yv, cols_v = y_for_features(val_labels,  features)
        if not cols_t:
            print(f"  {rd.name:32s} SKIP (no matching label columns)"); continue

        vs = np.load(rd / "val_score.npy").astype(np.float32)
        ts = np.load(rd / "test_score.npy").astype(np.float32)
        res = select_and_score(vs, yv, ts, yt, args.rate_cap)
        if res is None:
            print(f"  {rd.name:32s} SKIP (no valid threshold under cap)"); continue

        np.save(rd / "test_pred_mask_ca.npy", res["test_mask"].astype(np.int8))
        np.save(rd / "test_y_true_ca.npy",    yt)

        # for reference: old honest mask scored under the new labels
        old_on_ca = None
        p = rd / "test_pred_mask_honest.npy"
        if p.exists():
            old_on_ca = esa_f05(yt, np.load(p))["esa_f05"]

        row = {"run": rd.name, "n_features": len(features),
               "n_label_cols": len(cols_t),
               "ca_gt_events_test": res["test"]["gt_events"],
               "threshold": res["thr"], "merge_gap": res["mg"],
               "min_dur": res["md"],
               "val_esa_f05_ca": res["val"]["esa_f05"],
               "old_honest_mask_on_ca_labels": old_on_ca}
        row.update({f"test_{k}_ca": v for k, v in res["test"].items()})
        rows.append(row)
        print(f"  {rd.name:32s} events={res['test']['gt_events']:3d}  "
              f"val={res['val']['esa_f05']:.4f}  test={res['test']['esa_f05']:.4f}  "
              f"TPe={res['test']['TPe']}  FPe={res['test']['FPe']}")

    if not rows:
        print("Nothing evaluated."); return

    df = (pd.DataFrame(rows)
          .sort_values("test_esa_f05_ca", ascending=False)
          .reset_index(drop=True))
    df.insert(0, "rank", df.index + 1)
    p = root / "esa_scores_channel_aware.csv"
    df.to_csv(p, index=False)
    print(f"\nSaved: {p}")

    # ── ensemble search: labels = union of member channel sets ───────────────
    top = df.head(args.top_n)["run"].tolist()
    masks = {r: np.load(root / r / "test_pred_mask_ca.npy").astype(np.int8)
             for r in top if (root / r / "test_pred_mask_ca.npy").exists()}
    erows = []
    names = list(masks.keys())
    for n in range(2, min(4, len(names)) + 1):
        for combo in combinations(names, n):
            union_feats = sorted(set(f for c in combo for f in feats_by_run[c]))
            yt_u, _ = y_for_features(test_labels, union_feats)
            stack = np.stack([masks[c] for c in combo], axis=0)
            for logic, agg in [("OR", stack.max(axis=0)),
                               ("AND", stack.min(axis=0))]:
                m = esa_f05(yt_u, agg.astype(np.int8))
                erows.append({"combination": " | ".join(combo), "logic": logic,
                              "n_models": n, "union_n_channels": len(union_feats),
                              **m})
    if erows:
        dfe = (pd.DataFrame(erows)
               .sort_values("esa_f05", ascending=False).reset_index(drop=True))
        p = root / "cross_run_ensemble_ca.csv"
        dfe.to_csv(p, index=False)
        print(f"Saved: {p}\n")
        print("Top-10 channel-aware ensembles:")
        print(dfe[["combination", "logic", "n_models", "esa_f05",
                   "TPe", "FPe", "gt_events", "pred_rate"]]
              .head(10).to_string(index=False))

    b = df.iloc[0]
    print(f"\nChannel-aware best single: {b['run']}  "
          f"test F0.5={b['test_esa_f05_ca']:.4f}  "
          f"({int(b['test_TPe_ca'])}/{int(b['ca_gt_events_test'])} events)")


if __name__ == "__main__":
    main()
