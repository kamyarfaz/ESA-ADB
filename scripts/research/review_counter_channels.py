"""Training-only full-resolution counter review; never rewrites model inputs."""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    start, end = pd.Timestamp('2000-01-01'), pd.Timestamp('2003-01-01')
    labels = pd.read_csv(root/'data/ESA-Mission1/labels.csv')
    for c in ['StartTime', 'EndTime']:
        labels[c] = pd.to_datetime(labels[c], utc=True).dt.tz_localize(None)
    labels = labels[(labels.StartTime < end) & (labels.EndTime >= start)]
    labels = labels.merge(pd.read_csv(root/'data/ESA-Mission1/anomaly_types.csv')[['ID','Category']], on='ID', validate='many_to_one')
    labels.to_csv(args.output/'training_annotations.csv', index=False)
    rows = []
    fig, axes = plt.subplots(4, 2, figsize=(13, 11), constrained_layout=True)
    for channel, ax in zip(['channel_'+str(i) for i in range(4, 12)], axes.flat):
        source = root/'data/ESA-Mission1/channels'/f'{channel}.zip'
        before = source.stat()
        x = pd.read_pickle(source)
        x = x.loc[(x.index >= start) & (x.index < end), channel]
        assert x.index.is_monotonic_increasing and x.index.is_unique
        assert np.isfinite(x.to_numpy()).all()
        # Both endpoints must belong to training. No terminal zero is invented.
        delta = np.diff(x.to_numpy())
        left = x.index[:-1]
        changed = np.flatnonzero(delta != 0)
        relevant = labels[labels.Channel == channel]
        event_change = np.zeros(len(changed), dtype=bool)
        for row in relevant.itertuples():
            if row.Category in ['Anomaly', 'Rare Event']:
                event_change |= (left[changed] >= row.StartTime) & (left[changed] <= row.EndTime)
        rows.append(dict(feature=channel, raw_rows=len(x), raw_unique=x.nunique(),
                         nonzero_forward_differences=len(changed),
                         differences_outside_event_annotations=int((~event_change).sum()),
                         differences_in_event_annotations=int(event_change.sum()),
                         training_event_ids=relevant.loc[relevant.Category.isin(['Anomaly','Rare Event']), 'ID'].nunique(),
                         min_difference=float(delta.min()), max_difference=float(delta.max())))
        if len(changed):
            ax.scatter(left[changed], delta[changed], s=7, color='#2366a8')
            # Full-resolution context around first change, not a sampled-year trace.
            center = changed[0]
            local = x.iloc[max(0, center-20):min(len(x), center+22)]
            detail, dax = plt.subplots(figsize=(9, 3), constrained_layout=True)
            dax.step(local.index, local.to_numpy(), where='post', marker='.', markersize=3)
            dax.set(title=f'{channel}: native samples around first training change', ylabel='Raw anonymized value')
            dax.tick_params(axis='x', labelrotation=20)
            detail.savefig(args.output/f'{channel}_first_change.png', dpi=140)
            plt.close(detail)
        else:
            ax.text(.5, .5, 'No raw changes in training', ha='center', transform=ax.transAxes)
        ax.set_xlim(start, end)
        ax.axhline(0, color='gray', linewidth=.5)
        ax.set_title(f'{channel}: {len(changed):,} nonzero differences')
        ax.set_ylabel('Next value − current value')
        ax.grid(alpha=.15)
        after = source.stat()
        assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        print(channel, rows[-1], flush=True)
    fig.suptitle('Every nonzero native-sample difference, 2000–2002\nBoth endpoints inside training; no hourly subsampling; zeros omitted')
    fig.savefig(args.output/'counter_changes.png', dpi=150)
    plt.close(fig)
    table = pd.DataFrame(rows)
    table.to_csv(args.output/'counter_review.csv', index=False)
    assert len(table) == 8
    assert (table.nonzero_forward_differences == table.differences_outside_event_annotations + table.differences_in_event_annotations).all()
    (args.output/'protocol.json').write_text(json.dumps(dict(
        scope='[2000-01-01,2003-01-01); channels 4–11 native resolution',
        boundary='Both difference endpoints inside training; final raw sample not differenced.',
        labels='Raw channel annotations; overlap uses left difference timestamp, inclusive annotation ends.',
        caution='Outside event annotations does not mean nominal: gap/other annotations are not excluded. No assessment data used; no inputs removed.'), indent=2)+'\n')


if __name__ == '__main__':
    main()
