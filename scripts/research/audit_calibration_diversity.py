"""Describe existing pre-2007 splits; never select a split from event coverage."""
import argparse
import json
from pathlib import Path

import pandas as pd

from esa_thesis.calibration_audit import (
    digest, load_annotations, period_annotations, temporal_components,
    union_seconds, verify_recorded,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output directory')
    root = Path(__file__).resolve().parents[2]
    paths = [root/'data/ESA-Mission1'/n for n in ('labels.csv', 'anomaly_types.csv')]
    before = verify_recorded(root, paths)
    annotations = load_annotations(*paths)
    cutoff = pd.Timestamp('2007-01-01', tz='UTC')
    annotations = period_annotations(annotations, '2000-01-01', cutoff)
    periods = [(str(y), f'{y}-01-01', f'{y+1}-01-01') for y in range(2000, 2007)]
    pairs = []
    for y in range(2001, 2006):
        middle = f'{y}-07-01' if y < 2003 else f'{y}-04-01'
        periods.extend([(f'{y}_calibration', f'{y}-01-01', middle),
                        (f'{y}_assessment', middle, f'{y+1}-01-01')])
        pairs.append(y)
    rows, events, components, transfers = [], [], [], []
    selected = {}
    for name, start, end in periods:
        assert pd.Timestamp(end, tz='UTC') <= cutoff
        part = period_annotations(annotations, start, end)
        part = part[part.Category.isin(['Anomaly', 'Rare Event'])]
        for scope, frame in [('system', part), ('inputs_40_47', part[
                part.Channel.isin([f'channel_{i}' for i in range(40, 48)])])]:
            selected[name, scope] = frame
            groups = temporal_components(frame)
            assignments = {aid: i for i, group in enumerate(groups) for aid in group}
            rows.append(dict(period=name, scope=scope, start=start, end_exclusive=end,
                             ids=frame.ID.nunique(), classes=frame.Class.nunique(),
                             channels=frame.Channel.nunique(), components=len(groups),
                             anomalies=frame.loc[frame.Category=='Anomaly', 'ID'].nunique(),
                             rare_events=frame.loc[frame.Category=='Rare Event', 'ID'].nunique(),
                             annotated_union_seconds=union_seconds(frame),
                             carried_in_ids=';'.join(sorted(frame.loc[frame.carried_in, 'ID'].unique()))))
            for i, group in enumerate(groups):
                components.append(dict(period=name, scope=scope, component=i,
                                       ids=';'.join(group), id_count=len(group)))
            for aid, ev in frame.groupby('ID'):
                events.append(dict(period=name, scope=scope, event_id=aid,
                                   category=ev.Category.iloc[0], event_class=ev.Class.iloc[0],
                                   channels=';'.join(sorted(ev.Channel.unique())),
                                   component=assignments[aid], union_seconds=union_seconds(ev)))
    for y in pairs:
        for scope in ('system', 'inputs_40_47'):
            cal = selected[f'{y}_calibration', scope]
            val = selected[f'{y}_assessment', scope]
            ids = val[['ID', 'Class']].drop_duplicates()
            unseen = ids[~ids.Class.isin(cal.Class)]
            transfers.append(dict(fold=y, scope=scope, calibration_ids=cal.ID.nunique(),
                                  assessment_ids=val.ID.nunique(),
                                  assessment_ids_class_absent_from_calibration=len(unseen),
                                  unseen_classes=';'.join(sorted(set(val.Class)-set(cal.Class))),
                                  unseen_channels=';'.join(sorted(set(val.Channel)-set(cal.Channel))),
                                  shared_ids=';'.join(sorted(set(cal.ID)&set(val.ID)))))
    if verify_recorded(root, paths) != before:
        raise ValueError('Sources changed')
    args.output.mkdir(parents=True)
    for filename, data in [('periods', rows), ('events', events), ('components', components),
                           ('class_transfer', transfers)]:
        pd.DataFrame(data).to_csv(args.output/f'{filename}.csv', index=False)
    (args.output/'verification.json').write_text(json.dumps(dict(
        source_sha256=before, cutoff_exclusive=str(cutoff),
        script_sha256=digest(Path(__file__)),
        module_sha256=digest(root/'esa_thesis/calibration_audit.py'),
        limits='Annotation overlap diagnostics, not independent episodes or model metrics. '
               'Metadata files contain all years but analytical rows are strictly pre-2007. '
               'No telemetry, scores, or test-period summaries loaded.'), indent=2)+'\n')
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(transfers).to_string(index=False))


if __name__ == '__main__':
    main()
