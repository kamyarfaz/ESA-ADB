"""Inventory pre-2003 annotation coverage; no telemetry or assessment scores."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output directory')
    paths = [Path('data/ESA-Mission1/labels.csv'),
             Path('data/ESA-Mission1/anomaly_types.csv')]
    def hashes():
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    before = hashes()
    data = pd.read_csv(paths[0]).merge(pd.read_csv(paths[1]), on='ID', validate='many_to_one')
    for key in ('StartTime', 'EndTime'):
        data[key] = pd.to_datetime(data[key], utc=True)
    cutoff = pd.Timestamp('2003-01-01', tz='UTC')
    data = data[data.Category.isin(['Anomaly', 'Rare Event']) & (data.StartTime < cutoff)].copy()
    features = [f'channel_{i}' for i in range(40, 48)]
    rows = []
    def inventory(name, start, end):
        start, end = pd.Timestamp(start, tz='UTC'), pd.Timestamp(end, tz='UTC')
        assert start < end <= cutoff
        selected = data[(data.StartTime < end) & (data.EndTime >= start)]
        for scope, frame in [('system', selected), ('inputs_40_47', selected[selected.Channel.isin(features)])]:
            ids = sorted(frame.ID.unique())
            rows.append(dict(period=name, start=start.isoformat(), end_exclusive=end.isoformat(),
                             scope=scope, event_ids=len(ids), ids=';'.join(ids),
                             carried_in_ids=';'.join(sorted(frame.loc[frame.StartTime < start, 'ID'].unique()))))
        return set(selected.ID)
    for year in (2000, 2001, 2002):
        for q in range(1, 5):
            start = pd.Timestamp(year=year, month=3*q-2, day=1)
            inventory(f'{year}Q{q}', str(start.date()), str((start+pd.DateOffset(months=3)).date()))
    intersections = {}
    for year in (2001, 2002):
        calibration = inventory(f'{year}_inner_calibration', f'{year}-01-01', f'{year}-07-01')
        validation = inventory(f'{year}_inner_validation', f'{year}-07-01', f'{year+1}-01-01')
        intersections[str(year)] = sorted(calibration & validation)
    assert before == hashes()
    args.output.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(args.output/'coverage.csv', index=False)
    (args.output/'provenance.json').write_text(json.dumps(dict(
        source_sha256=before, cutoff_exclusive=str(cutoff),
        inner_calibration_validation_shared_ids=intersections,
        limitation='Annotation overlap counts, not independent episodes or telemetry availability. Full metadata read; only pre-2003 rows analyzed. No assessment scores or telemetry read.'), indent=2)+'\n')
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
