"""Corrected ESA event-wise evaluation on raw annotation IDs and timestamps.

Equivalent to the event-wise portion of bundled ESAScores (not affiliation or
channel-localization scores). Vectorized interval operations make validation
sweeps practical. See tests/thesis/test_esa_evaluation.py for reference parity.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd

METRIC_VERSION = 'esa-ew-id-duration-v1'
CATEGORIES = ('Anomaly', 'Rare Event')


def _merge(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([int(start), int(end)])
    return np.asarray(merged, dtype=np.int64).reshape(-1, 2)


def timestamps_ns(values):
    values = np.asarray(values)
    if np.issubdtype(values.dtype, np.integer):
        result = values.astype(np.int64)
    else:
        dates = pd.DatetimeIndex(pd.to_datetime(values, utc=True))
        if dates.hasnans:
            raise ValueError('Timestamps contain NaT')
        result = dates.as_unit('ns').asi8
    if result.ndim != 1 or len(result) < 2 or np.any(np.diff(result) <= 0):
        raise ValueError('Need at least two strictly increasing timestamps')
    return result


@dataclass
class EventEvaluator:
    timestamps: np.ndarray
    annotations: pd.DataFrame
    categories: tuple = CATEGORIES

    def __post_init__(self):
        self.timestamps = timestamps_ns(self.timestamps)
        self.start, self.end = int(self.timestamps[0]), int(self.timestamps[-1])
        required = {'ID', 'StartTime', 'EndTime', 'Category'}
        if not required.issubset(self.annotations.columns):
            raise ValueError(f'Annotation columns required: {sorted(required)}')
        labels = self.annotations.copy()
        if labels[list(required)].isna().any().any():
            raise ValueError('Annotation IDs, times and categories must not be null')
        for name in ('StartTime', 'EndTime'):
            labels[name] = pd.to_datetime(labels[name], utc=True).dt.as_unit('ns').astype('int64')
        if (labels.EndTime < labels.StartTime).any():
            raise ValueError('Annotation end precedes start')
        labels = labels[(labels.EndTime >= self.start) & (labels.StartTime <= self.end)].copy()
        labels.StartTime = labels.StartTime.clip(lower=self.start)
        labels.EndTime = labels.EndTime.clip(upper=self.end)
        # Non-selected annotations remain neutral for false alarms, as in ESAScores.
        self.all_intervals = _merge(labels[['StartTime', 'EndTime']].itertuples(index=False, name=None))
        selected = labels[labels.Category.isin(self.categories)]
        self.by_id = {str(aid): _merge(group[['StartTime', 'EndTime']].itertuples(index=False, name=None))
                      for aid, group in selected.groupby('ID', sort=True)}
        annotated = sum(int(e) - int(s) for s, e in self.all_intervals)
        self.nominal_ns = self.end - self.start - annotated

    @classmethod
    def from_files(cls, timestamps, labels_path, types_path, categories=CATEGORIES):
        labels = pd.read_csv(labels_path).merge(pd.read_csv(types_path), on='ID', how='left', validate='many_to_one')
        return cls(timestamps, labels, tuple(categories))

    def prediction_intervals(self, prediction):
        prediction = np.asarray(prediction)
        if prediction.shape != self.timestamps.shape or not np.isin(prediction, [0, 1]).all():
            raise ValueError('Prediction must be a binary vector matching timestamps')
        boundaries = np.diff(np.r_[0, prediction, 0].astype(np.int8))
        starts = np.flatnonzero(boundaries == 1)
        ends = np.flatnonzero(boundaries == -1)
        closed = ends == len(prediction)
        return self.timestamps[starts], self.timestamps[np.minimum(ends, len(prediction) - 1)], closed

    @staticmethod
    def _overlaps(starts, ends, closed, lo, hi):
        return (starts <= hi) & ((ends > lo) | ((ends == lo) & closed))

    def score(self, prediction):
        starts, ends, closed = self.prediction_intervals(prediction)
        matched = np.zeros(len(starts), dtype=bool)
        detected = 0
        redundant = 0
        for intervals in self.by_id.values():
            detected_id = False
            for lo, hi in intervals:
                hit = self._overlaps(starts, ends, closed, lo, hi)
                count = int(hit.sum())
                detected_id |= count > 0
                redundant += max(0, count - 1)
                matched |= hit
            detected += int(detected_id)
        touches_any = np.zeros(len(starts), dtype=bool)
        overlap_ns = 0
        for lo, hi in self.all_intervals:
            touches_any |= self._overlaps(starts, ends, closed, lo, hi)
            # Each union interval and prediction interval is disjoint internally.
            lengths = np.maximum(0, np.minimum(ends, hi) - np.maximum(starts, lo))
            overlap_ns += int(lengths.sum())
        fp = int((~matched & ~touches_any).sum())
        fn = len(self.by_id) - detected
        alarm_ns = int((ends - starts).sum())
        fp_ns = max(0, alarm_ns - overlap_ns)
        # Explicit extension: if no nominal time exists, no nominal-time penalty.
        tnr = 1.0 - fp_ns / self.nominal_ns if self.nominal_ns else 1.0
        precision = detected / max(detected + fp, 1) * tnr
        recall = detected / max(detected + fn, 1)
        denominator = .25 * precision + recall
        f05 = 1.25 * precision * recall / denominator if denominator else 0.0
        return {'EW_F_0.50': f05, 'EW_precision': precision, 'EW_recall': recall,
                'TPe': detected, 'FPe': fp, 'FNe': fn,
                'num_true_events': len(self.by_id), 'num_pred_events': len(starts),
                'alarming_precision': detected / max(detected + redundant, 1),
                'nominal_seconds': self.nominal_ns / 1e9,
                'false_positive_seconds': fp_ns / 1e9,
                'nominal_false_alarm_rate': 1 - tnr,
                'false_alarms_per_day': fp / ((self.end - self.start) / 1e9 / 86400),
                'metric_version': METRIC_VERSION}

    def metadata(self):
        return {'metric_version': METRIC_VERSION, 'scope': 'all supplied annotation IDs',
                'categories': list(self.categories), 'start': str(pd.Timestamp(self.start, tz='UTC')),
                'end': str(pd.Timestamp(self.end, tz='UTC')), 'event_ids': list(self.by_id),
                'zero_nominal_duration_policy': 'no nominal-time penalty',
                'prediction_intervals': '[timestamp_i,timestamp_next_zero); final endpoint closed'}
