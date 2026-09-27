"""Read trusted local ESA pickles; audit raw telemetry in [2000, 2003) only."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    metadata = pd.read_csv(root / 'data/ESA-Mission1/channels.csv')
    rows = []
    for channel in metadata.Channel:
        path = root / 'data/ESA-Mission1/channels' / (channel + '.zip')
        before = path.stat()
        frame = pd.read_pickle(path)
        frame = frame.loc[(frame.index >= '2000-01-01') & (frame.index < '2003-01-01')]
        assert len(frame) and channel in frame
        values = frame[channel].to_numpy()
        finite = values[np.isfinite(values)]
        dt = np.diff(frame.index.asi8) / 1e9
        positive = dt[dt > 0]
        row = dict(feature=channel, rows=len(frame), first=str(frame.index.min()),
                   last=str(frame.index.max()), duplicate_timestamps=int(frame.index.duplicated().sum()),
                   backwards_steps=int((dt < 0).sum()), nonfinite=int((~np.isfinite(values)).sum()),
                   raw_min=float(finite.min()), raw_max=float(finite.max()),
                   median_positive_interval_seconds=float(np.median(positive)),
                   p99_positive_interval_seconds=float(np.quantile(positive, .99)),
                   max_interval_seconds=float(positive.max()),
                   intervals_over_one_hour=int((dt > 3600).sum()),
                   derivative_in_preparation=4 <= int(channel.split('_')[1]) <= 11,
                   source_bytes=before.st_size, source_mtime_ns=before.st_mtime_ns)
        after = path.stat()
        assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        rows.append(row)
        print(channel, 'audited', flush=True)
    audit = pd.DataFrame(rows)
    assert len(audit) == 76 and audit.feature.is_unique
    audit.to_csv(args.output / 'raw_telemetry.csv', index=False)
    profile = pd.read_csv(root / 'docs/research/tables/channel-study-2026-09-22/channel_profile.csv')
    review = profile.merge(audit, on='feature', how='left', validate='one_to_one')
    review['decision'] = np.where(review.constant_training_nominal,
        'removal candidate: validate rare events and transformation first',
        'retain pending full-resolution redundancy and event review')
    review.loc[review.review_flag.str.contains('sample', case=False), 'decision'] = 'retain: coarse sample can hide rare changes'
    review['raw_audit_scope'] = np.where(review.kind == 'telemetry',
        'all raw training observations, including annotated periods', 'telecommand raw audit pending')
    review.to_csv(args.output / 'channel_review.csv', index=False)
    (args.output / 'protocol.json').write_text(json.dumps(dict(
        interval='[2000-01-01,2003-01-01)', selection='training only; no assessment statistics',
        coverage='76 raw telemetry channels; prepared profile includes 11 telecommands',
        caveats=['Intervals over one hour are descriptive, not proven missing packets.',
                 'Raw statistics include annotated periods; prepared profile is nominal only.',
                 'No channels removed; no data transformed; source size/mtime checked, not content hashes.',
                 'Physical metadata is anonymized; no engineering meaning inferred.']), indent=2) + '\n')


if __name__ == '__main__':
    main()
