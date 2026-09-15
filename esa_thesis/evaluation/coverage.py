#!/usr/bin/env python3
"""
greedy_coverage_ensemble.py
===========================
A PRINCIPLED replacement for the brute-force OR-ensemble search.

Motivation
----------
channel_aware_rescore_v2.py enumerates every OR/AND combination of the top runs
and reports the one with the best TEST F0.5. That has two problems: the choice of
combination peeks at test (a mild leak), and it gives no reason WHY a given triple
was chosen ("the search ranked it first" is not a defence).

This script instead builds the ensemble greedily, using VALIDATION only:

    1. Recompute each run's val + test mask from its cached score and the
       (threshold, merge_gap, min_dur) already selected in
       esa_scores_channel_aware_v2.csv  (no model, no retraining).
    2. Seed the ensemble with the single run of highest VALIDATION Protocol-A F0.5.
    3. Repeatedly add the candidate whose OR with the current set most improves
       VALIDATION Protocol-A F0.5 (union-of-channels relevance, matching v2).
    4. Stop when no candidate improves validation F0.5 (with a small patience).
    5. Report the resulting fixed ensemble on TEST (Protocol A and B).

Because every selection decision uses validation, the reported test number is
leak-free, and each member has an explicit reason for inclusion: it covered
validation anomalies the current set was missing. This is the defensible
"coverage-maximising ensemble" for the thesis.

Leak-free note: test masks are touched only to PRINT the final result; they never
influence which runs are chosen.

Usage:
    python -m esa_thesis coverage
    python -m esa_thesis coverage --objective A --max_members 5 --cand_floor 0.45
"""

from esa_thesis.paths import project_path, default_run_dir

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from esa_thesis.evaluation.subset import (
    postprocess, find_events, event_scores,
    relevant_flags, load_label_frame, run_features,
)


def mask_from(score: np.ndarray, thr: float, mg: int, md: int) -> np.ndarray:
    return postprocess((score > thr).astype(np.int8), int(mg), int(md))


def union_score(members: List[str], masks: Dict[str, np.ndarray],
                feats: Dict[str, List[str]], labels, events, objective: str):
    """OR the member masks; score under Protocol A (union-channel relevance) or B."""
    agg = np.stack([masks[m] for m in members], axis=0).max(axis=0).astype(np.int8)
    if objective == "A":
        union_feats = sorted(set(f for m in members for f in feats[m]))
        rel = relevant_flags(labels, union_feats)
        return event_scores(events, agg, rel), agg, len(union_feats)
    else:
        return event_scores(events, agg, None), agg, \
               len(sorted(set(f for m in members for f in feats[m])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=(str(default_run_dir())))
    ap.add_argument("--train_file", default=(str(project_path('data/preprocessed/multivariate/ESA-Mission1-semi-supervised/84_months.train.csv'))))
    ap.add_argument("--test_file", default=(str(project_path('data/preprocessed/multivariate/ESA-Mission1-semi-supervised/84_months.test.csv'))))
    ap.add_argument("--event_merge", type=int, default=0)
    ap.add_argument("--objective", choices=["A", "B"], default="A",
                    help="selection objective: A=subset-relevant (matches v2 headline), "
                         "B=all mission events (operational recall)")
    ap.add_argument("--max_members", type=int, default=6)
    ap.add_argument("--min_val_events", type=int, default=6,
                    help="drop candidates that scope fewer than this many VALIDATION "
                         "events — their val score is a fluke on too-few events "
                         "(this is what let fine_ch08_05_12 seed with a spurious 1.0)")
    ap.add_argument("--min_val_prec", type=float, default=0.80,
                    help="event-precision floor on VALIDATION; the ensemble may only "
                         "grow while union val precision stays above this")
    args = ap.parse_args()
    root = Path(args.root)

    scores_csv = root / "esa_scores_channel_aware_v2.csv"
    if not scores_csv.exists():
        print("Run channel_aware_rescore_v2.py first (need esa_scores_channel_aware_v2.csv).")
        return
    sel = pd.read_csv(scores_csv).set_index("run")

    run_dirs = [d for d in sorted(root.iterdir())
                if d.is_dir() and d.name in sel.index
                and (d / "val_score.npy").exists() and (d / "test_score.npy").exists()]
    val_len = len(np.load(run_dirs[0] / "val_score.npy"))

    test_labels = load_label_frame(Path(args.test_file))
    val_labels  = load_label_frame(Path(args.train_file), tail=val_len)
    yt = test_labels.to_numpy().max(axis=1).astype(np.int8)
    yv = val_labels.to_numpy().max(axis=1).astype(np.int8)
    if args.event_merge > 0:
        yt = postprocess(yt, args.event_merge, 1); yv = postprocess(yv, args.event_merge, 1)
    test_events, val_events = find_events(yt), find_events(yv)

    print("Greedy coverage ensemble  (leak-free: members chosen on VALIDATION)")
    print(f"Root: {root}")
    print(f"objective={args.objective}  test events={len(test_events)}  "
          f"val events={len(val_events)}\n")

    # recompute val & test masks; score each run individually on validation
    val_masks, test_masks, feats = {}, {}, {}
    solo = {}   # name -> dict(scoped, TPe, FPe, prec)
    for rd in run_dirs:
        name = rd.name
        f = run_features(rd)
        if not f:
            continue
        thr = float(sel.loc[name, "threshold"]); mg = int(sel.loc[name, "merge_gap"]); md = int(sel.loc[name, "min_dur"])
        vs = np.load(rd / "val_score.npy").astype(np.float32)
        ts = np.load(rd / "test_score.npy").astype(np.float32)
        vm = mask_from(vs, thr, mg, md); tm = mask_from(ts, thr, mg, md)
        s = event_scores(val_events, vm, relevant_flags(val_labels, f))
        prec = s["TPe"] / (s["TPe"] + s["FPe"]) if (s["TPe"] + s["FPe"]) else 0.0
        val_masks[name] = vm; test_masks[name] = tm; feats[name] = f
        solo[name] = {"scoped": s["n_scoped_events"], "TPe": s["TPe"], "FPe": s["FPe"], "prec": prec}

    # candidate filter: validation must be INFORMATIVE (enough scoped events) and clean
    cands = [r for r in val_masks
             if solo[r]["scoped"] >= args.min_val_events and solo[r]["prec"] >= args.min_val_prec]
    print(f"{len(cands)} of {len(val_masks)} runs pass val guards "
          f"(>= {args.min_val_events} scoped val events, val precision >= {args.min_val_prec}):")
    for r in sorted(cands, key=lambda r: solo[r]["TPe"], reverse=True):
        print(f"    {r:32s} val TPe={solo[r]['TPe']}/{solo[r]['scoped']}  "
              f"FPe={solo[r]['FPe']}  prec={solo[r]['prec']:.2f}")
    if not cands:
        print("\nNo run has an informative, clean validation signal — the 13-event "
              "validation set cannot support ensemble selection. Report the brute-force "
              "ensemble as an ORACLE upper bound instead."); return

    # greedy COVERAGE: maximise union validation TPe while union val precision stays
    # above the floor. Coverage does not saturate at 1.0 the way F0.5 does.
    def union_val(members):
        agg = np.stack([val_masks[m] for m in members], 0).max(0).astype(np.int8)
        uf = sorted(set(f for m in members for f in feats[m]))
        s = event_scores(val_events, agg, relevant_flags(val_labels, uf))
        p = s["TPe"] / (s["TPe"] + s["FPe"]) if (s["TPe"] + s["FPe"]) else 0.0
        return s["TPe"], p

    seed = max(cands, key=lambda r: (solo[r]["TPe"], solo[r]["prec"]))
    members = [seed]
    best_tpe, best_prec = union_val(members)
    print(f"\n  seed  {seed:32s} val coverage={best_tpe} events  prec={best_prec:.2f}")
    path = [(seed, best_tpe, best_prec)]
    while len(members) < args.max_members:
        best_add, best_sc = None, None
        for c in cands:
            if c in members:
                continue
            tpe, p = union_val(members + [c])
            if tpe > best_tpe and p >= args.min_val_prec:
                if best_sc is None or tpe > best_sc[0] or (tpe == best_sc[0] and p > best_sc[1]):
                    best_add, best_sc = c, (tpe, p)
        if best_add is None:
            print("  (no candidate adds validation coverage within the precision floor — stopping)")
            break
        members.append(best_add); best_tpe, best_prec = best_sc
        path.append((best_add, best_tpe, best_prec))
        print(f"  +add  {best_add:32s} val coverage={best_tpe} events  prec={best_prec:.2f}")

    # final report on TEST (touched only now)
    print("\nFinal ensemble (members chosen on validation):")
    for m in members:
        print(f"    {m}   [{','.join(feats[m])}]")
    tA, aggT, uch = union_score(members, test_masks, feats, test_labels, test_events, "A")
    tB, _, _      = union_score(members, test_masks, feats, test_labels, test_events, "B")
    vA, _, _      = union_score(members, val_masks,  feats, val_labels,  val_events,  "A")
    print(f"\n  union channels: {uch}")
    print(f"  VALIDATION  A_F0.5={vA['esa_f05']:.4f}")
    print(f"  TEST        A_F0.5={tA['esa_f05']:.4f}  (TPe={tA['TPe']}/{tA['n_scoped_events']}, FPe={tA['FPe']})")
    print(f"  TEST        B_F0.5={tB['esa_f05']:.4f}  (TPe={tB['TPe']}/89)")

    # baseline: best single model on test, for the delta
    best_single = max(cands, key=lambda r: sel.loc[r, "A_esa_f05"])
    bs = float(sel.loc[best_single, "A_esa_f05"])
    print(f"\n  vs best single model ({best_single}): test A_F0.5={bs:.4f}"
          f"   ->  ensemble delta = {tA['esa_f05']-bs:+.4f}")

    out = root / "greedy_ensemble.csv"
    pd.DataFrame([{"step": i, "added": m, "val_coverage_events": t, "val_precision": p}
                  for i, (m, t, p) in enumerate(path)]).to_csv(out, index=False)
    print(f"\nSaved greedy path: {out}")


if __name__ == "__main__":
    main()
