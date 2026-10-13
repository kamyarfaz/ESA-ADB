"""Quantify hold/restoration on existing channels 40–47, strictly pre-2003."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from esa_thesis.calibration_audit import digest, load_annotations, verify_recorded
from esa_thesis.preprocessing_audit import held_values, restore_candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output directory')
    root = Path(__file__).resolve().parents[2]
    features = [f'channel_{i}' for i in range(40, 48)]
    paired = [f'is_anomaly_{c}' for c in features]
    source = root/'data/preprocessed/multivariate/ESA-Mission1-semi-supervised/84_months.train.csv'
    paths = [root/'data/ESA-Mission1'/n for n in ('labels.csv', 'anomaly_types.csv')]
    raw_paths = [root/f'data/ESA-Mission1/channels/{c}.zip' for c in features]
    before = verify_recorded(root, [source, *paths, *raw_paths])
    labels = load_annotations(*paths)
    start, end = pd.Timestamp('2000-01-01', tz='UTC'), pd.Timestamp('2003-01-01', tz='UTC')
    labels = labels[(labels.StartTime<end)&(labels.EndTime>=start)]
    parts = []
    for chunk in pd.read_csv(source, usecols=['timestamp', *features, *paired], chunksize=100000):
        t = pd.to_datetime(chunk.timestamp, utc=True)
        keep = (t>=start)&(t<end)
        parts.append(chunk.loc[keep])
        if (t>=end).any():
            break
    prepared = pd.concat(parts, ignore_index=True)
    times = pd.to_datetime(prepared.timestamp, utc=True).astype('int64').to_numpy()
    expected = pd.date_range(start, end, freq='30s', inclusive='left').asi8
    np.testing.assert_array_equal(times, expected)
    if len(times)!=3156480 or not np.isfinite(prepared[features].to_numpy()).all():
        raise ValueError('Prepared values invalid')
    year = pd.to_datetime(times, utc=True).year.to_numpy()
    summary, changes = [], []
    for channel, path in zip(features, raw_paths):
        raw = pd.read_pickle(path)  # Trusted, hash-verified local archive.
        raw.index = pd.to_datetime(raw.index, utc=True)
        raw = raw[(raw.index>=start)&(raw.index<end)]
        rt, rv = raw.index.asi8, raw[channel].to_numpy()
        raw_labels = np.zeros(len(raw), dtype=np.uint8)
        events = labels[labels.Channel==channel]
        for event in events.itertuples():
            lo, hi = np.searchsorted(rt, [event.StartTime.value, event.EndTime.value], side='left')
            hi = np.searchsorted(rt, event.EndTime.value, side='right')
            raw_labels[lo:hi] = {'Anomaly':1, 'Rare Event':2, 'Communication Gap':3}[event.Category]
        hold = held_values(rt, rv, times)
        hold_labels = held_values(rt, raw_labels, times)
        legacy = hold.copy()
        legacy_labels = hold_labels.copy()
        restore_t, positions = restore_candidates(rt, raw_labels)
        inside = (restore_t>=times[0])&(restore_t<=times[-1])
        restore_t, positions = restore_t[inside], positions[inside]
        grid_positions = np.searchsorted(times, restore_t)
        legacy[grid_positions] = rv[positions]
        legacy_labels[grid_positions] = raw_labels[positions]
        actual = prepared[channel].to_numpy()
        actual_labels = prepared[f'is_anomaly_{channel}'].to_numpy()
        hold_diff = ~np.isclose(actual, hold, rtol=1e-6, atol=1e-12)
        legacy_diff = ~np.isclose(actual, legacy, rtol=1e-6, atol=1e-12)
        for y in (2000, 2001, 2002):
            keep = year==y
            summary.append(dict(channel=channel, year=y, prepared_rows=int(keep.sum()),
                                raw_rows=int((raw.index.year==y).sum()),
                                hold_value_mismatches=int((hold_diff&keep).sum()),
                                restored_value_mismatches=int((legacy_diff&keep).sum()),
                                hold_label_mismatches=int(((actual_labels!=hold_labels)&keep).sum()),
                                restored_label_mismatches=int(((actual_labels!=legacy_labels)&keep).sum()),
                                initial_future_fill_rows=int(((times<rt[0])&keep).sum()),
                                restoration_rows=int((year[grid_positions]==y).sum())))
        for target, raw_pos, grid_pos in zip(restore_t, positions, grid_positions):
            matching = events[(events.StartTime.astype('int64')<=rt[raw_pos]) &
                              (events.EndTime.astype('int64')>=rt[raw_pos]) &
                              events.Category.isin(['Anomaly', 'Rare Event'])]
            changes.append(dict(channel=channel, timestamp=str(pd.Timestamp(target, tz='UTC')),
                                original_timestamp=str(raw.index[raw_pos]),
                                hold_value=float(hold[grid_pos]), restored_value=float(rv[raw_pos]),
                                prepared_value=float(actual[grid_pos]),
                                value_changed=not np.isclose(hold[grid_pos], rv[raw_pos], rtol=1e-6, atol=1e-12),
                                label_changed=bool(hold_labels[grid_pos]!=raw_labels[raw_pos]),
                                event_ids=';'.join(sorted(matching.ID.unique()))))
        print(f'{channel}: hold mismatches={hold_diff.sum()}, restored mismatches={legacy_diff.sum()}', flush=True)
    after = verify_recorded(root, [source, *paths, *raw_paths])
    if before!=after:
        raise ValueError('Inputs changed')
    args.output.mkdir(parents=True)
    table = pd.DataFrame(summary)
    table.to_csv(args.output/'summary.csv', index=False)
    pd.DataFrame(changes).to_csv(args.output/'restoration_rows.csv', index=False)
    success = bool((table.restored_value_mismatches==0).all() and (table.restored_label_mismatches==0).all())
    (args.output/'verification.json').write_text(json.dumps(dict(
        legacy_reconstruction_matches=success, interval='[2000-01-01,2003-01-01)',
        prepared_rows=len(times), input_sha256=before,
        script_sha256=digest(Path(__file__)), module_sha256=digest(root/'esa_thesis/preprocessing_audit.py'),
        plan_sha256=digest(root/'docs/research/preprocessing-retention-plan-2026-10-13.md'),
        limits='Raw archives decoded in full, then cropped. Eight existing inputs only. '
               'No training, benchmark test read, channel selection or causal-pipeline claim.'), indent=2)+'\n')
    if not success:
        raise ValueError('Unexplained legacy reconstruction differences; inspect saved evidence')


if __name__=='__main__':
    main()
