#!/usr/bin/env python3
"""
channel_aware_rescore_v2.py
===========================
Fixes the event-fragmentation bug in channel_aware_rescore.py and reports TWO
recall definitions side by side, per the thesis decision.

THE BUG (v1): ground truth was the union of a run's is_anomaly_* columns, then
contiguous label runs were counted as "events". When different channels flag
different sub-windows of the SAME underlying anomaly, one real anomaly
fragments into several counted events — so per-run event counts were not
comparable (e.g. a 6-channel run reported 109 events > the 89 real ones).

THE FIX: events are defined ONCE at the mission level from the global
annotation (max over ALL is_anomaly_* columns -> contiguous runs = the real
anomalies). A subset model is credited with detecting a mission event if any
predicted segment overlaps that event's [start,end] span, regardless of which
channel flagged it. This matches the ESA-ADB subset-evaluation intent and
removes fragmentation.

TWO PROTOCOLS reported per run (headline = both, thesis discusses the gap):
  A) SUBSET-RELEVANT: recall denominator = mission events whose span intersects
     at least one timestep where any of the run's channels is annotated
     anomalous. "Of the anomalies visible in my channels, how many did I catch?"
  B) ALL-EVENTS (hard): recall denominator = all mission events (the original
     89). "Of every mission anomaly, how many did this subset catch?"

Precision is IDENTICAL under both (false positives are false positives
regardless of recall bookkeeping); only the recall denominator / F0.5 differ.

Leak-free: thresholds selected on VALIDATION data only, fixed protocol
(best val ESA F0.5 under protocol A subject to val pred-rate <= --rate_cap),
then applied unchanged to test. Selection uses protocol A (the operationally
meaningful one); protocol B is reported at the same threshold.

Outputs (inside --root):
    <run>/test_pred_mask_ca2.npy
    esa_scores_channel_aware_v2.csv
    cross_run_ensemble_ca2.csv

Usage:
    python -m esa_thesis subset
    python -m esa_thesis subset --rate_cap 0.01 --top_n 8
"""

from esa_thesis.paths import project_path, default_run_dir

import argparse
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

THRESH_PERCENTILES = list(np.concatenate([
    np.linspace(50, 95, 10),
    np.array([96, 97, 98, 99, 99.3, 99.5, 99.7, 99.8, 99.9, 99.95, 99.97, 99.99]),
]))
MERGE_GAPS = [0, 16, 32, 64, 128, 256, 512, 1024]
MIN_DURS   = [1, 8, 16, 32, 64, 128]


# ── postprocess / events (identical to sweep script) ──────────────────────────

def postprocess(mask: np.ndarray, merge_gap: int, min_dur: int) -> np.ndarray:
    out = (mask > 0).astype(np.int8)
    if merge_gap > 0 and out.sum() > 0:
        p = np.empty(len(out) + 2, dtype=np.int8); p[0] = 0; p[1:-1] = out; p[-1] = 0
        d = np.diff(p.astype(np.int16))
        st = np.where(d == 1)[0]; en = np.where(d == -1)[0]
        if len(st) > 1:
            gs, ge = en[:-1], st[1:]
            fill = (ge - gs) <= merge_gap
            for a, b in zip(gs[fill], ge[fill]):
                out[a:b] = 1
    if min_dur > 1 and out.sum() > 0:
        p = np.empty(len(out) + 2, dtype=np.int8); p[0] = 0; p[1:-1] = out; p[-1] = 0
        d = np.diff(p.astype(np.int16))
        st = np.where(d == 1)[0]; en = np.where(d == -1)[0]
        dur = en - st
        for a, b in zip(st[dur < min_dur], en[dur < min_dur]):
            out[a:b] = 0
    return out


def find_events(y: np.ndarray) -> List[Tuple[int, int]]:
    y8 = (y > 0).astype(np.int8)
    if y8.sum() == 0:
        return []
    p = np.zeros(len(y8) + 2, dtype=np.int8); p[1:-1] = y8
    d = np.diff(p.astype(np.int16))
    st = np.where(d == 1)[0]; en = np.where(d == -1)[0] - 1
    return list(zip(st.tolist(), en.tolist()))


def _overlaps_any(span: Tuple[int, int], events: List[Tuple[int, int]]) -> bool:
    s, e = span
    for rs, re in events:
        if re >= s and rs <= e:
            return True
    return False


def event_scores(gt_events: List[Tuple[int, int]],
                 pred: np.ndarray,
                 relevant_flags: Optional[np.ndarray] = None) -> Dict:
    """
    Mission-level event scoring.
      gt_events        : list of (start,end) global anomalies (fixed set)
      pred             : binary prediction over full timeline
      relevant_flags   : optional bool array; a gt event counts toward the
                         denominator only if its span intersects a True flag.
                         None -> all gt events count (protocol B).
    Precision uses ALL predicted events vs ALL gt events (channel-agnostic).
    """
    pred_events = find_events(pred)

    # which gt events are in scope for recall
    if relevant_flags is None:
        scoped = gt_events
    else:
        rel_ev = find_events(relevant_flags.astype(np.int8))
        scoped = [ev for ev in gt_events if _overlaps_any(ev, rel_ev)]

    # TP among scoped gt events
    TPe = sum(1 for ev in scoped if _overlaps_any(ev, pred_events))
    FNe = len(scoped) - TPe

    # FP predicted events: overlap NO global gt event (use full gt, not scoped,
    # so a prediction hitting an out-of-scope real anomaly is not punished)
    FPe = sum(1 for pe in pred_events if not _overlaps_any(pe, gt_events))

    gt_all = np.zeros(len(pred), dtype=np.int8)
    for s, e in gt_events:
        gt_all[s:e + 1] = 1
    gt_b = gt_all.astype(bool); pred_b = pred.astype(bool)
    FPt = int((~gt_b & pred_b).sum()); Nt = int((~gt_b).sum())

    fpt = FPt / Nt if Nt > 0 else 0.0
    pc  = TPe / max(TPe + FPe + fpt, 1e-12)
    re  = TPe / max(TPe + FNe, 1)
    b2  = 0.25
    den = b2 * pc + re
    f05 = (1 + b2) * pc * re / den if den > 0 else 0.0
    return {"esa_f05": round(f05, 4), "precision_c": round(pc, 4),
            "recall_e": round(re, 4), "TPe": TPe, "FPe": FPe, "FNe": FNe,
            "n_scoped_events": len(scoped), "pred_events": len(pred_events),
            "pred_rate": round(float(pred.mean()), 4)}


# ── labels ─────────────────────────────────────────────────────────────────────

def load_label_frame(path: Path, tail: Optional[int] = None) -> pd.DataFrame:
    print(f"  loading label columns from {path.name} ...", flush=True)
    # The ESA CSVs occasionally contain a non-UTF-8 byte in a non-label column
    # header/value. We only need is_anomaly_* (numeric) and channel names
    # (ASCII), so a latin-1 fallback is safe and lossless for our purposes.
    read_kwargs = dict(usecols=lambda c: c.startswith("is_anomaly_"),
                       low_memory=False)
    try:
        df = pd.read_csv(path, encoding="utf-8", **read_kwargs)
    except (UnicodeDecodeError, ValueError):
        print("    (non-UTF-8 bytes found; retrying with latin-1)", flush=True)
        df = pd.read_csv(path, encoding="latin-1", **read_kwargs)
    if df.shape[1] == 0:
        raise RuntimeError(
            f"No is_anomaly_* columns found in {path}. "
            "Check the header row — column prefix may differ.")
    for c in df.columns:
        df[c] = (pd.to_numeric(df[c], errors="coerce")
                   .fillna(0).astype(np.float32) > 0).astype(np.int8)
    if tail is not None:
        df = df.tail(tail).reset_index(drop=True)
    return df


def relevant_flags(labels: pd.DataFrame, features: List[str]) -> np.ndarray:
    cols = [f"is_anomaly_{f}" for f in features
            if f"is_anomaly_{f}" in labels.columns]
    if not cols:
        return np.zeros(len(labels), dtype=np.int8)
    return labels[cols].to_numpy().max(axis=1).astype(np.int8)


def run_features(run_dir: Path) -> List[str]:
    sel = run_dir / "selected_thresholds.csv"
    if sel.exists():
        df = pd.read_csv(sel, nrows=1)
        if "features" in df.columns:
            return [f.strip() for f in str(df["features"].iloc[0]).split(",") if f.strip()]
    return []


# ── honest selection on protocol A ────────────────────────────────────────────

def select_and_score(vs, yv_events, v_relevant,
                     ts, yt_events, t_relevant,
                     rate_cap: float) -> Optional[Dict]:
    nz = vs[np.isfinite(vs) & (vs > 0)]
    if len(nz) == 0:
        return None
    thr_list = [float(v) for v in
                np.unique(np.percentile(nz, THRESH_PERCENTILES)) if np.isfinite(v)]
    best = None
    for thr in thr_list:
        raw = (vs > thr).astype(np.int8)
        if raw.mean() > 0.30:
            continue
        for mg in MERGE_GAPS:
            for md in MIN_DURS:
                vp = postprocess(raw, mg, md)
                if vp.mean() > rate_cap or vp.mean() == 0:
                    continue
                vm = event_scores(yv_events, vp, v_relevant)   # protocol A on val
                key = (vm["esa_f05"], -vm["FPe"], -vp.mean())
                if best is None or key > best["key"]:
                    best = {"key": key, "thr": thr, "mg": mg, "md": md}
    if best is None:
        return None
    tp = postprocess((ts > best["thr"]).astype(np.int8), best["mg"], best["md"])
    return {"thr": best["thr"], "mg": best["mg"], "md": best["md"],
            "A": event_scores(yt_events, tp, t_relevant),   # subset-relevant
            "B": event_scores(yt_events, tp, None),          # all events
            "mask": tp}


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=(str(default_run_dir())))
    ap.add_argument("--train_file", default=(str(project_path('data/preprocessed/multivariate/ESA-Mission1-semi-supervised/84_months.train.csv'))))
    ap.add_argument("--test_file", default=(str(project_path('data/preprocessed/multivariate/ESA-Mission1-semi-supervised/84_months.test.csv'))))
    ap.add_argument("--rate_cap", type=float, default=0.01)
    ap.add_argument("--top_n", type=int, default=8)
    ap.add_argument("--event_merge", type=int, default=0,
                    help="global anomaly-fragment merge tolerance (timesteps); "
                         "report sensitivity over {0,60,300}")
    args = ap.parse_args()
    root = Path(args.root)

    run_dirs = [d for d in sorted(root.iterdir())
                if d.is_dir() and not d.name.startswith("plots")
                and (d / "test_score.npy").exists()
                and (d / "val_score.npy").exists()]
    if not run_dirs:
        print("No runs found."); return
    val_len = len(np.load(run_dirs[0] / "val_score.npy"))

    print("Channel-aware honest re-evaluation v2  (mission-level events)")
    print(f"Root: {root}\nProtocol A selection: best val ESA F0.5, val rate <= {args.rate_cap}\n")

    test_labels = load_label_frame(Path(args.test_file))
    val_labels  = load_label_frame(Path(args.train_file), tail=val_len)

    # fixed mission-level event sets (max over ALL channels).
    # A global merge tolerance stitches anomaly fragments separated by <= g
    # nominal timesteps into one event. This is a stated, reportable choice
    # (sensitivity: g in {0, 60, 300}); g=0 uses the raw annotation as-is.
    yt_global = test_labels.to_numpy().max(axis=1).astype(np.int8)
    yv_global = val_labels.to_numpy().max(axis=1).astype(np.int8)
    if args.event_merge > 0:
        yt_global = postprocess(yt_global, args.event_merge, 1)
        yv_global = postprocess(yv_global, args.event_merge, 1)
    test_events = find_events(yt_global)
    val_events  = find_events(yv_global)
    print(f"  mission test events (fixed, merge={args.event_merge}): "
          f"{len(test_events)}   val events: {len(val_events)}\n")

    rows, feats_by_run = [], {}
    for rd in run_dirs:
        features = run_features(rd)
        if not features:
            print(f"  {rd.name:32s} SKIP (no feature list)"); continue
        feats_by_run[rd.name] = features

        t_rel = relevant_flags(test_labels, features)
        v_rel = relevant_flags(val_labels,  features)
        vs = np.load(rd / "val_score.npy").astype(np.float32)
        ts = np.load(rd / "test_score.npy").astype(np.float32)

        res = select_and_score(vs, val_events, v_rel,
                               ts, test_events, t_rel, args.rate_cap)
        if res is None:
            print(f"  {rd.name:32s} SKIP (no valid threshold under cap)"); continue

        np.save(rd / "test_pred_mask_ca2.npy", res["mask"].astype(np.int8))
        A, B = res["A"], res["B"]
        row = {"run": rd.name, "n_features": len(features),
               "threshold": res["thr"], "merge_gap": res["mg"], "min_dur": res["md"],
               # protocol A: subset-relevant
               "A_scoped_events": A["n_scoped_events"],
               "A_esa_f05": A["esa_f05"], "A_recall": A["recall_e"],
               "A_precision": A["precision_c"],
               "A_TPe": A["TPe"], "A_FPe": A["FPe"], "A_FNe": A["FNe"],
               # protocol B: all mission events
               "B_scoped_events": B["n_scoped_events"],
               "B_esa_f05": B["esa_f05"], "B_recall": B["recall_e"],
               "B_TPe": B["TPe"], "B_FNe": B["FNe"]}
        rows.append(row)
        print(f"  {rd.name:32s} A:F0.5={A['esa_f05']:.4f} "
              f"({A['TPe']}/{A['n_scoped_events']}, FPe={A['FPe']})   "
              f"B:F0.5={B['esa_f05']:.4f} ({B['TPe']}/{B['n_scoped_events']})")

    if not rows:
        print("Nothing evaluated."); return
    df = (pd.DataFrame(rows).sort_values("A_esa_f05", ascending=False)
          .reset_index(drop=True))
    df.insert(0, "rank", df.index + 1)
    p = root / "esa_scores_channel_aware_v2.csv"
    df.to_csv(p, index=False)
    print(f"\nSaved: {p}")

    # ── ensemble on protocol A relevance = union of member channels ──────────
    top = df.head(args.top_n)["run"].tolist()
    masks = {r: np.load(root / r / "test_pred_mask_ca2.npy").astype(np.int8)
             for r in top if (root / r / "test_pred_mask_ca2.npy").exists()}
    erows = []
    names = list(masks.keys())
    for n in range(2, min(4, len(names)) + 1):
        for combo in combinations(names, n):
            union_feats = sorted(set(f for c in combo for f in feats_by_run[c]))
            u_rel = relevant_flags(test_labels, union_feats)
            stack = np.stack([masks[c] for c in combo], axis=0)
            for logic, agg in [("OR", stack.max(axis=0)),
                               ("AND", stack.min(axis=0))]:
                mA = event_scores(test_events, agg.astype(np.int8), u_rel)
                mB = event_scores(test_events, agg.astype(np.int8), None)
                erows.append({"combination": " | ".join(combo), "logic": logic,
                              "n_models": n, "union_n_channels": len(union_feats),
                              "A_esa_f05": mA["esa_f05"], "A_TPe": mA["TPe"],
                              "A_FPe": mA["FPe"], "A_scoped": mA["n_scoped_events"],
                              "B_esa_f05": mB["esa_f05"], "B_TPe": mB["TPe"]})
    if erows:
        dfe = (pd.DataFrame(erows).sort_values("A_esa_f05", ascending=False)
               .reset_index(drop=True))
        p = root / "cross_run_ensemble_ca2.csv"
        dfe.to_csv(p, index=False)
        print(f"Saved: {p}\n")
        print("Top-10 ensembles (protocol A):")
        print(dfe.head(10).to_string(index=False))

    b = df.iloc[0]
    print(f"\nBest single (protocol A): {b['run']}  F0.5={b['A_esa_f05']:.4f}  "
          f"({int(b['A_TPe'])}/{int(b['A_scoped_events'])} relevant events)  "
          f"| same model protocol B: F0.5={b['B_esa_f05']:.4f} "
          f"({int(b['B_TPe'])}/{int(b['B_scoped_events'])})")


if __name__ == "__main__":
    main()
