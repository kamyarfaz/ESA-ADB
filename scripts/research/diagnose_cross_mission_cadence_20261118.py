"""Read-only, training-only feasibility diagnosis; does not prepare or train v2."""
import gc
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results_longrun/cross_mission_transfer_20261118_v1'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def mark(q, lo, hi, width=0):
    result = np.zeros(len(q), bool)
    for a, b in zip(lo, hi):
        left = np.searchsorted(q, a - width, side='right' if width else 'left')
        right = np.searchsorted(q, b, side='right')
        result[left:right] = True
    return result


def windows(excluded):
    prefix = np.r_[0, np.cumsum(excluded, dtype=np.int64)]
    starts = np.arange(0, len(excluded) - 255, 16)
    return int(np.sum(prefix[starts + 256] == prefix[starts]))


def main():
    folder = OUT / 'prepared/m1/train'
    meta = json.loads((folder / 'metadata.json').read_text())
    times = np.load(folder / 'times.npy')
    availability = np.load(folder / 'available.npy', mmap_mode='r')
    ages = np.load(folder / 'age_ns.npy', mmap_mode='r')
    excluded = np.load(folder / 'excluded.npy').astype(bool)
    ann = pd.read_csv(ROOT / 'data/ESA-Mission1/labels.csv')
    lo = pd.to_datetime(ann.StartTime, utc=True).astype('int64').to_numpy()
    hi = pd.to_datetime(ann.EndTime, utc=True).astype('int64').to_numpy()
    within = (hi >= times[0]) & (lo < pd.Timestamp(meta['bounds'][1]).value)
    ann = ann.loc[within].reset_index(drop=True)
    lo, hi = lo[within], hi[within]
    labels_only = mark(times, lo, hi, width=30 * 10**9)
    for j, ch in enumerate(meta['features']):
        age = np.array(ages[:, j])
        source = times - age
        good = np.flatnonzero(age >= 0)
        source[age < 0] = source[good[0]]
        same = ann.Channel.to_numpy() == ch
        labels_only |= mark(source, lo[same], hi[same])
    original = labels_only | ~availability.all(axis=1)
    np.testing.assert_array_equal(original, excluded)
    assert windows(original) == 0
    candidate = labels_only.copy()
    details = []
    slow = {'channel_62', 'channel_63'}
    for j, ch in enumerate(meta['features']):
        if ch not in slow:
            candidate |= ~availability[:, j]
            continue
        path = ROOT / f'data/ESA-Mission1/channels/{ch}.zip'
        series = pd.read_pickle(path).iloc[:, 0]
        series.index = pd.DatetimeIndex(series.index).tz_localize(None)
        start, end = map(pd.Timestamp, meta['bounds'])
        series = series[(series.index >= start) & (series.index < end)]
        gaps = np.diff(series.index.asi8) / 1e9
        q = pd.to_datetime(times)
        old = series.reindex(q, method='ffill', tolerance=pd.Timedelta(seconds=600))
        np.testing.assert_array_equal(np.isfinite(old.to_numpy(dtype=np.float32)), availability[:, j])
        proposed = series.reindex(q, method='ffill', tolerance=pd.Timedelta(seconds=900))
        valid = np.isfinite(proposed.to_numpy(dtype=np.float32))
        candidate |= ~valid
        details.append(dict(channel=ch, raw_sha256=sha(path),
                            median_gap_seconds=float(np.median(gaps)),
                            p99_gap_seconds=float(np.percentile(gaps, 99)),
                            original_hold_seconds=600, proposed_hold_seconds=900,
                            original_availability=float(availability[:, j].mean()),
                            proposed_availability=float(valid.mean()),
                            original_complete_windows=windows(~availability[:, j]),
                            proposed_complete_windows=windows(~valid)))
        print(ch, details[-1], flush=True)
        del series, old, proposed, valid, gaps
        gc.collect()
    assert not np.any(candidate & ~original), 'Proposal must not remove previously eligible rows'
    receipt = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                   diagnostic_only=True, protocol_change_approved=False,
                   training_period=meta['bounds'], assessment_used=False,
                   training_or_inference_started=False,
                   method='Full pandas causal lookup with finite-value checks for both slow channels; all other availability unchanged; independently reconstruct all training exclusions and window counts.',
                   original_nominal_rows=int((~original).sum()), original_nominal_windows=windows(original),
                   proposed_nominal_rows=int((~candidate).sum()), proposed_nominal_windows=windows(candidate),
                   channels=details, script_sha256=sha(__file__),
                   frozen_protocol_sha256=sha(OUT / 'preparation_frozen/protocol.json'),
                   prepared_train_receipt_sha256=sha(folder / 'complete.json'),
                   limitations='Availability feasibility only: no learning, score comparison, or evidence of physically valid interpolation. Longer hold is a protocol change requiring authorization.')
    (OUT / 'cadence-feasibility-diagnosis.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
