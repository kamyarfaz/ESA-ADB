"""Annotation diversity and label-influence diagnostics, not model selection."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from .evaluation.esa import EventEvaluator, CATEGORIES


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_recorded(root, paths):
    """Verify only these dependencies against the preserved original manifest."""
    root = Path(root).resolve()
    manifest = root / 'migration/manifest.json'
    entries = {r['path']: r for r in json.loads(manifest.read_text())['files']}
    hashes = {}
    for path in paths:
        path = Path(path)
        if path.is_symlink():
            raise ValueError(f'Symlink dependency: {path}')
        key = path.resolve().relative_to(root).as_posix()
        row = entries.get(key)
        if row is None or 'sha256' not in row:
            raise ValueError(f'Unrecorded dependency: {key}')
        actual = digest(path)
        if actual != row['sha256'] or path.stat().st_size != row['size']:
            raise ValueError(f'Original transfer checksum mismatch: {key}')
        hashes[key] = actual
    return hashes


def load_annotations(labels, types):
    frame = pd.read_csv(labels).merge(pd.read_csv(types), on='ID', how='left',
                                      validate='many_to_one')
    required = ['ID', 'Channel', 'StartTime', 'EndTime', 'Category', 'Class']
    if frame[required].isna().any().any():
        raise ValueError('Missing annotation metadata')
    for key in ('StartTime', 'EndTime'):
        frame[key] = pd.to_datetime(frame[key], utc=True)
    if (frame.EndTime < frame.StartTime).any():
        raise ValueError('Invalid annotation interval')
    if (frame.groupby('ID').Category.nunique() > 1).any():
        raise ValueError('Inconsistent event category')
    return frame


def period_annotations(frame, start, end):
    """Half-open period, closed source annotations; end clips at end - 1 ns."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start.tzinfo is None:
        start = start.tz_localize('UTC')
    if end.tzinfo is None:
        end = end.tz_localize('UTC')
    if not start < end:
        raise ValueError('Invalid period')
    part = frame[(frame.StartTime < end) & (frame.EndTime >= start)].copy()
    part['carried_in'] = part.StartTime < start
    part.StartTime = part.StartTime.clip(lower=start)
    part.EndTime = part.EndTime.clip(upper=end-pd.Timedelta(1, 'ns'))
    return part


def temporal_components(frame):
    """Connect target IDs only when real closed intervals intersect.

    Separate intervals of one ID share a node; enclosing spans are never used.
    Neutral categories cannot create bridges between target IDs.
    """
    frame = frame[frame.Category.isin(CATEGORIES)]
    parent = {str(x): str(x) for x in frame.ID.unique()}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    active = []
    for row in frame.sort_values(['StartTime', 'EndTime', 'ID']).itertuples():
        aid = str(row.ID)
        active = [(end, other) for end, other in active if end >= row.StartTime]
        for _, other in active:
            a, b = find(aid), find(other)
            if a != b:
                parent[max(a, b)] = min(a, b)
        active.append((row.EndTime, aid))
    groups = {}
    for aid in parent:
        groups.setdefault(find(aid), []).append(aid)
    return sorted([sorted(g) for g in groups.values()], key=lambda g: g[0])


def union_seconds(frame):
    intervals = sorted((r.StartTime.value, r.EndTime.value) for r in frame.itertuples())
    merged = []
    for lo, hi in intervals:
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return sum(hi-lo for lo, hi in merged) / 1e9


def neutralize(frame, ids):
    """Preserve annotation time as neutral; remove IDs only from targets."""
    out = frame.copy()
    known = set(out.loc[out.Category.isin(CATEGORIES), 'ID'].astype(str))
    if not set(ids).issubset(known):
        raise ValueError('Withheld ID is not a target')
    out.loc[out.ID.astype(str).isin(ids), 'Category'] = 'Withheld (neutral)'
    return out


def detection_ids(evaluator, prediction):
    starts, ends, closed = evaluator.prediction_intervals(prediction)
    return [aid for aid, spans in evaluator.by_id.items()
            if any(evaluator._overlaps(starts, ends, closed, lo, hi).any()
                   for lo, hi in spans)]


def without_targets(metrics, detected, withheld, total_ids):
    """Exact metric change when full temporal components become neutral.

    Predictions, false events and nominal durations are unchanged. A prediction
    that loses a target match still touches a neutral annotation, so it does not
    become a false event under ESA conventions. Tested against EventEvaluator.
    """
    keep = set(total_ids) - set(withheld)
    if not keep:
        return None
    tp = len(set(detected) & keep)
    precision = tp / max(tp + metrics['FPe'], 1) * (1-metrics['nominal_false_alarm_rate'])
    recall = tp / len(keep)
    den = .25*precision + recall
    return {'EW_F_0.50': 1.25*precision*recall/den if den else 0.,
            'EW_precision': precision, 'EW_recall': recall, 'TPe': tp,
            'FPe': metrics['FPe'], 'FNe': len(keep)-tp}
