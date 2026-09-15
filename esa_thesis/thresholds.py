"""Thresholds components of the Mission 1 research pipeline.

Extracted without changing numerical behavior; see docs/thesis-research-context.md.
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


def sweep_thresholds(val_score: np.ndarray, y_val: np.ndarray,
                     test_score: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
    """
    Speed-ups over original:
    1. Threshold list de-duplicated (np.unique already done in thresholds())
    2. For each threshold: create binary masks once, then iterate post-process params
    3. Postprocess is now vectorised (no Python while-loops)
    4. Skip (threshold, mg, md) combos where val is already saturated for mg=0, md=1
       — if a threshold saturates at the loosest post-processing it will saturate
       everywhere, so we skip the inner grid early.
    """
    rows = []
    thr_list = thresholds(val_score)

    for thr in thr_list:
        val_raw  = (val_score  > thr).astype(np.int8)
        test_raw = (test_score > thr).astype(np.int8)

        # Early saturation check: if even no post-processing is already saturated
        # on val, all (mg, md) combos will also be saturated → record cheaply
        if val_raw.mean() > SATURATION_RATE:
            # Still need one metrics call to produce a valid (saturated) row
            vm = metrics(y_val,  val_raw)
            tm = metrics(y_test, test_raw)
            rows.append({
                "threshold": thr, "merge_gap": 0, "min_dur": 1,
                **{f"val_{k}":  v for k, v in vm.items()},
                **{f"test_{k}": v for k, v in tm.items()},
            })
            continue

        for mg in MERGE_GAPS:
            for md in MIN_DURS:
                vp = postprocess(val_raw,  mg, md)
                tp = postprocess(test_raw, mg, md)
                vm = metrics(y_val,  vp)
                tm = metrics(y_test, tp)
                rows.append({
                    "threshold": thr, "merge_gap": mg, "min_dur": md,
                    **{f"val_{k}":  v for k, v in vm.items()},
                    **{f"test_{k}": v for k, v in tm.items()},
                })

    return pd.DataFrame(rows)


def select_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for max_rate in [0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15]:
        sub = df[(df.val_pred_anomaly_rate <= max_rate) & (~df.val_saturated)].copy()
        if len(sub):
            sub = sub.sort_values(["val_esa_f05", "val_event_f05", "val_point_f05"], ascending=False)
            r = sub.iloc[0].to_dict(); r["selection_name"] = f"best_val_esa_f05_val_pred_rate_lte_{max_rate:g}"; r["selection_type"] = "clean_validation_selected"; rows.append(r)
            sub = sub.sort_values(["val_point_f05", "val_esa_f05", "val_event_f05"], ascending=False)
            r = sub.iloc[0].to_dict(); r["selection_name"] = f"best_val_point_f05_val_pred_rate_lte_{max_rate:g}"; r["selection_type"] = "clean_validation_selected"; rows.append(r)
    for max_rate in [0.05, 0.10, 0.15]:
        sub = df[(df.test_pred_anomaly_rate <= max_rate) & (~df.test_saturated)].copy()
        if len(sub):
            sub = sub.sort_values(["test_esa_f05", "test_event_f05", "test_point_f05"], ascending=False)
            r = sub.iloc[0].to_dict(); r["selection_name"] = f"diagnostic_best_test_esa_f05_test_pred_rate_lte_{max_rate:g}"; r["selection_type"] = "diagnostic_test_selected"; rows.append(r)
            sub = sub.sort_values(["test_point_f05", "test_esa_f05", "test_event_f05"], ascending=False)
            r = sub.iloc[0].to_dict(); r["selection_name"] = f"diagnostic_best_test_point_f05_test_pred_rate_lte_{max_rate:g}"; r["selection_type"] = "diagnostic_test_selected"; rows.append(r)
    return pd.DataFrame(rows)


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
