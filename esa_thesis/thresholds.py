"""Thresholds components of the Mission 1 research pipeline.

Uses corrected annotation-ID/duration evaluation; see docs/research/corrected-evaluation.md.
"""
from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from .config import (MERGE_GAPS, MIN_DURS, SATURATION_RATE, THRESH_PERCENTILES)
from .metrics import (metrics)


def postprocess(mask: np.ndarray, merge_gap: int, min_dur: int) -> np.ndarray:
    """Gap-fill then min-duration filter, fully in NumPy."""
    out = (mask > 0).astype(np.int8)

    # ---- Step 1: merge gaps ------------------------------------------------
    if merge_gap > 0 and out.sum() > 0:
        padded = np.empty(len(out) + 2, dtype=np.int8)
        padded[0] = 0; padded[1:-1] = out; padded[-1] = 0
        diff   = np.diff(padded.astype(np.int16))
        starts = np.where(diff == 1)[0]
        ends   = np.where(diff == -1)[0]   # exclusive end
        # For each consecutive pair of events, check gap and fill
        if len(starts) > 1:
            gap_starts = ends[:-1]          # end of event i (exclusive)
            gap_ends   = starts[1:]         # start of event i+1
            gaps       = gap_ends - gap_starts
            fill_mask  = gaps <= merge_gap
            for gs, ge in zip(gap_starts[fill_mask], gap_ends[fill_mask]):
                out[gs:ge] = 1

    # ---- Step 2: remove short segments ------------------------------------
    if min_dur > 1 and out.sum() > 0:
        padded = np.empty(len(out) + 2, dtype=np.int8)
        padded[0] = 0; padded[1:-1] = out; padded[-1] = 0
        diff   = np.diff(padded.astype(np.int16))
        starts = np.where(diff == 1)[0]
        ends   = np.where(diff == -1)[0]   # exclusive end
        durations = ends - starts
        for s, e in zip(starts[durations < min_dur], ends[durations < min_dur]):
            out[s:e] = 0

    return out


def thresholds(score: np.ndarray) -> List[float]:
    nz = score[np.isfinite(score) & (score > 0)]
    if len(nz) == 0:
        return [float(np.nanmax(score))]
    return [float(v) for v in np.unique(np.percentile(nz, THRESH_PERCENTILES)) if np.isfinite(v)]


def sweep_thresholds(val_score: np.ndarray, *, evaluator,
                     threshold_values=None, merge_gaps=None, min_durs=None) -> pd.DataFrame:
    """Evaluate validation only. No test scores or labels enter this function."""
    if not np.isfinite(val_score).all():
        raise ValueError("Non-finite validation scores")
    values = thresholds(val_score) if threshold_values is None else threshold_values
    # A no-alarm candidate guarantees a defined conservative selection.
    values = np.unique([*values, float(np.max(val_score))])
    rows = []
    for thr in values:
        raw = (val_score > thr).astype(np.int8)
        for mg in (MERGE_GAPS if merge_gaps is None else merge_gaps):
            for md in (MIN_DURS if min_durs is None else min_durs):
                pred = postprocess(raw, mg, md)
                event = evaluator.score(pred)
                rows.append({"threshold": float(thr), "merge_gap": int(mg), "min_dur": int(md),
                             "val_esa_f05": event["EW_F_0.50"],
                             "val_event_f05": event["EW_F_0.50"],
                             "val_event_precision": event["EW_precision"],
                             "val_event_recall": event["EW_recall"],
                             "val_pred_anomaly_rate": float(pred.mean()),
                             "val_saturated": bool(pred.mean() > SATURATION_RATE),
                             **{"val_" + k: v for k, v in event.items()}})
    return pd.DataFrame(rows)


def select_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Freeze one decision rule using corrected validation F0.5 and a 1% cap.

    Ties favor less nominal alarm time and then a higher threshold. Test columns
    cannot affect selection, even if a caller attaches them to the frame.
    """
    sub = df[(df.val_pred_anomaly_rate <= .01) & (~df.val_saturated)].copy()
    if sub.empty:
        raise ValueError("No validation candidate satisfies the 1% cap")
    chosen = sub.sort_values(["val_esa_f05", "val_false_positive_seconds", "threshold", "merge_gap", "min_dur"],
                             ascending=[False, True, False, True, True], kind="stable").iloc[0].to_dict()
    chosen.update(selection_name="best_val_esa_f05_val_pred_rate_lte_0.01",
                  selection_type="clean_validation_selected")
    return pd.DataFrame([chosen])


def pick_leakfree_row(seldf: pd.DataFrame) -> Dict:
    """
    LEAK-FREE selection (Section 7 protocol). Choose the (threshold, merge_gap,
    min_dur) that becomes the saved masks and plotted result using ONLY
    validation performance:

        best val ESA F0.5  subject to  val predicted-anomaly-rate <= 1%.

    The test_* columns are never sorted on here. This replaces the previous
    block that ranked validation-selected candidates by test_esa_f05 (that was
    the leak: it let test performance decide which val rule 'won').

    Fallback order if the 1% row is absent for a run: the tightest available
    val-rate cap among best_val_esa_f05_* rows, then best val_esa_f05 overall.
    We NEVER fall back to diagnostic_test_selected rows.
    """
    clean = seldf[seldf.selection_type == "clean_validation_selected"].copy()
    if len(clean) == 0:
        raise RuntimeError(
            "no clean_validation_selected rows; refusing to select on test data")

    preferred = clean[
        clean.selection_name == "best_val_esa_f05_val_pred_rate_lte_0.01"]
    if len(preferred):
        return preferred.iloc[0].to_dict()

    esa_rows = clean[clean.selection_name.astype(str).str.startswith(
        "best_val_esa_f05_val_pred_rate_lte_")].copy()
    if len(esa_rows):
        esa_rows["_rate_cap"] = pd.to_numeric(
            esa_rows.selection_name.str.extract(r"lte_([0-9.]+)$")[0],
            errors="coerce")
        esa_rows = esa_rows.sort_values("_rate_cap")
        return esa_rows.iloc[0].to_dict()

    return clean.sort_values("val_esa_f05", ascending=False).iloc[0].to_dict()
