"""Metrics components of the Mission 1 research pipeline.

Uses corrected annotation-ID/duration evaluation; see docs/research/corrected-evaluation.md.
"""
from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
from .config import (SATURATION_RATE)


def find_events(y: np.ndarray) -> List[Tuple[int, int]]:
    """Return list of (start, end) inclusive for each contiguous run of 1s."""
    y8 = (y > 0).astype(np.int8)
    if y8.sum() == 0:
        return []
    # Pad to detect edges at boundaries
    padded = np.empty(len(y8) + 2, dtype=np.int8)
    padded[0] = 0; padded[1:-1] = y8; padded[-1] = 0
    diff    = np.diff(padded.astype(np.int16))
    starts  = np.where(diff == 1)[0]    # rising edges
    ends    = np.where(diff == -1)[0] - 1  # falling edges (inclusive)
    return list(zip(starts.tolist(), ends.tolist()))


def ov(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return min(a[1], b[1]) >= max(a[0], b[0])


def fbeta(p: float, r: float, beta: float = 0.5) -> float:
    b2  = beta * beta
    den = b2 * p + r
    return 0.0 if den <= 0 else (1 + b2) * p * r / den


def legacy_metrics(gt: np.ndarray, pred: np.ndarray) -> Dict:
    gt   = (gt > 0).astype(np.int8)
    pred = (pred > 0).astype(np.int8)

    gt_ev  = find_events(gt)
    pr_ev  = find_events(pred)

    # Event-level: sorted interval overlap check
    def count_detected(query_events, ref_events):
        """How many query events overlap at least one ref event (O(n log n))."""
        if not ref_events or not query_events:
            return 0
        ref_arr  = np.array(ref_events, dtype=np.int64)   # (M,2)
        ref_sort = ref_arr[np.argsort(ref_arr[:, 0])]
        detected = 0
        for qs, qe in query_events:
            # Binary search: first ref whose start <= qe
            lo, hi = 0, len(ref_sort)
            while lo < hi:
                mid = (lo + hi) // 2
                if ref_sort[mid, 0] <= qe:
                    lo = mid + 1
                else:
                    hi = mid
            # Check candidates [0 .. lo-1] for overlap
            for k in range(lo - 1, -1, -1):
                rs, re = ref_sort[k]
                if re < qs:
                    break
                if re >= qs:    # overlap found
                    detected += 1
                    break
        return detected

    TPe = count_detected(gt_ev,  pr_ev)
    FNe = len(gt_ev) - TPe
    FPe = len(pr_ev) - count_detected(pr_ev, gt_ev)

    ep = TPe / max(TPe + FPe, 1)
    er = TPe / max(TPe + FNe, 1)

    # Point-level (vectorised)
    gt_b   = gt.astype(bool)
    pred_b = pred.astype(bool)
    TPt = int(( gt_b &  pred_b).sum())
    FPt = int((~gt_b &  pred_b).sum())
    TNt = int((~gt_b & ~pred_b).sum())
    FNt = int(( gt_b & ~pred_b).sum())

    pp = TPt / max(TPt + FPt, 1)
    pr = TPt / max(TPt + FNt, 1)

    Nt          = int((~gt_b).sum())
    fpt_penalty = FPt / Nt if Nt else 0.0
    precision_c = TPe / max(TPe + FPe + fpt_penalty, 1e-12)
    pred_rate   = float(pred.mean())

    return {
        "event_f05": fbeta(ep, er, 0.5), "event_f1": fbeta(ep, er, 1.0),
        "event_precision": float(ep), "event_recall": float(er),
        "esa_f05": fbeta(precision_c, er, 0.5),
        "esa_precision_c": float(precision_c), "esa_recall_e": float(er),
        "TPe": TPe, "FPe": FPe, "FNe": FNe,
        "num_true_events": len(gt_ev), "num_pred_events": len(pr_ev),
        "point_f05": fbeta(pp, pr, 0.5), "point_f1": fbeta(pp, pr, 1.0),
        "point_precision": float(pp), "point_recall": float(pr),
        "TPt": TPt, "FPt": FPt, "TNt": TNt, "FNt": FNt,
        "pred_anomaly_rate": pred_rate,
        "true_anomaly_rate": float(gt.mean()),
        "saturated": bool(pred_rate > SATURATION_RATE),
    }


def metrics(gt: np.ndarray, pred: np.ndarray, *, evaluator) -> Dict:
    """Point diagnostics plus official ID/duration event scores.

    The annotation evaluator is mandatory: binary labels cannot reconstruct IDs.
    Legacy values are retained only under an explicit legacy_ prefix.
    """
    result = legacy_metrics(gt, pred)
    official = evaluator.score(pred)
    for key in ('esa_f05', 'esa_precision_c', 'esa_recall_e', 'event_f05',
                'event_f1', 'event_precision', 'event_recall', 'TPe', 'FPe',
                'FNe', 'num_true_events', 'num_pred_events'):
        result['legacy_' + key] = result.pop(key)
    precision, recall = official['EW_precision'], official['EW_recall']
    result.update(official)
    result.update(esa_f05=official['EW_F_0.50'], esa_precision_c=precision,
                  esa_recall_e=recall, event_precision=precision, event_recall=recall,
                  event_f05=official['EW_F_0.50'], event_f1=fbeta(precision, recall, 1.0))
    return result
