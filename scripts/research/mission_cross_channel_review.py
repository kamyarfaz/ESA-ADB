"""Training-only, raw-data channel correspondence diagnostics; never train models."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / 'docs/research/mission-cross-channel-protocol-2026-11-02.md'
EVIDENCE = ROOT / 'docs/research/tables/mission-cross-channel-2026-11-02'
STEP_NS = 300 * 10**9
LAGS = (1, 6, 12, 72, 288)
QKEYS = ('q01', 'q10', 'q25', 'q75', 'q90', 'q99')
AKEYS = tuple('acf_' + str(n) for n in LAGS)
DKEYS = ('log_delta_median', 'log_delta_p95', 'delta_zero_fraction')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def timestamp_ns(values):
    return pd.DatetimeIndex(pd.to_datetime(values, utc=True)).as_unit('ns').asi8


def held_values(times, values, grid, intervals, max_age_ns=2 * STEP_NS):
    """Causal hold; invalidate stale values and any annotation crossed by the hold."""
    times, grid = np.asarray(times), np.asarray(grid)
    if len(times) == 0 or np.any(np.diff(times) <= 0):
        raise ValueError('Require nonempty, strictly increasing source times')
    loc = np.searchsorted(times, grid, side='right') - 1
    safe = np.maximum(loc, 0)
    valid = (loc >= 0) & (grid - times[safe] <= max_age_ns)
    if len(intervals):
        intervals = np.asarray(intervals, dtype=np.int64)
        intervals = intervals[np.argsort(intervals[:, 0])]
        last = np.searchsorted(intervals[:, 0], grid, side='right') - 1
        max_end = np.maximum.accumulate(intervals[:, 1])
        crosses = (last >= 0) & (max_end[np.maximum(last, 0)] >= times[safe])
        valid &= ~crosses
    result = np.asarray(values, dtype=np.float64)[safe].copy()
    result[~valid] = np.nan
    result[~np.isfinite(result)] = np.nan
    return result


def correlation(a, b):
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 3 or np.ptp(a[keep]) == 0 or np.ptp(b[keep]) == 0:
        return np.nan
    return float(np.corrcoef(a[keep], b[keep])[0, 1])


def profile(x):
    finite = x[np.isfinite(x)]
    result = {'rows': len(x), 'eligible': len(finite)}
    if len(finite) < 100:
        return result
    q = np.quantile(finite, [.01, .1, .25, .5, .75, .9, .99])
    scale = q[4] - q[2]
    result.update(median=q[3], iqr=scale, constant=bool(np.ptp(finite) == 0))
    if scale <= 0:
        return result
    result.update(zip(QKEYS, (q[[0, 1, 2, 4, 5, 6]] - q[3]) / scale))
    for lag in LAGS:
        # Endpoints must be available; gaps inside the lag are explicitly excluded.
        bad = np.r_[0, np.cumsum(~np.isfinite(x))]
        ok = (bad[lag + 1:] - bad[:-lag - 1]) == 0
        a, b = x[:-lag].copy(), x[lag:].copy()
        a[~ok] = np.nan
        result['acf_' + str(lag)] = correlation(a, b)
    d = np.abs(np.diff(x))
    d = d[np.isfinite(d)]
    if len(d):
        result.update(log_delta_median=np.log1p(np.median(d) / scale),
                      log_delta_p95=np.log1p(np.quantile(d, .95) / scale),
                      delta_zero_fraction=float((d == 0).mean()))
    return result


def stable(v, cutoff):
    v = np.asarray(v)
    return bool(np.isfinite(v).all() and (np.abs(v) >= cutoff).all()
                and ((v > 0).all() or (v < 0).all()))


def pair_tables(frame, mission, output):
    blocks = [('pooled', frame)] + [(str(q), x) for q, x in frame.groupby(frame.index.to_period('Q'))]
    matrices = {}
    rows = []
    for label, block in blocks:
        # Even pooled changes may not span a quarter boundary.
        d = block.diff()
        quarter = block.index.to_period('Q')
        d.iloc[np.flatnonzero(np.r_[True, quarter[1:] != quarter[:-1]])] = np.nan
        matrices[label] = (block.corr(), block.corr(method='spearman'), d.corr())
        counts = block.notna().astype(np.int64).T.dot(block.notna().astype(np.int64))
        dcounts = d.notna().astype(np.int64).T.dot(d.notna().astype(np.int64))
        for i, a in enumerate(frame.columns):
            for b in frame.columns[i + 1:]:
                r, s, dr = matrices[label]
                rows.append(dict(mission=mission, block=label, a=a, b=b,
                                 pearson=r.loc[a, b], spearman=s.loc[a, b],
                                 delta_pearson=dr.loc[a, b], pairs=int(counts.loc[a, b]),
                                 delta_pairs=int(dcounts.loc[a, b])))
    detail = pd.DataFrame(rows)
    detail.to_csv(output / f'mission{mission}_pair_blocks.csv', index=False)
    summary = []
    for (a, b), group in detail.groupby(['a', 'b'], sort=False):
        pooled = group.loc[group.block == 'pooled'].iloc[0]
        q = group.loc[group.block != 'pooled']
        summary.append(dict(mission=mission, a=a, b=b, pooled_pearson=pooled.pearson,
                            pooled_spearman=pooled.spearman, pooled_delta=pooled.delta_pearson,
                            min_abs_quarter_pearson=q.pearson.abs().min() if q.pearson.notna().all() else np.nan,
                            min_abs_quarter_delta=q.delta_pearson.abs().min() if q.delta_pearson.notna().all() else np.nan,
                            stable_level_090=stable(q.pearson, .90),
                            stable_changes_050=stable(q.delta_pearson, .50),
                            quarters=len(q), min_quarter_pairs=int(q.pairs.min())))
    summary = pd.DataFrame(summary)
    summary.to_csv(output / f'mission{mission}_pair_summary.csv', index=False)
    for name, matrix in zip(('pearson', 'spearman', 'delta'), matrices['pooled']):
        matrix.to_csv(output / f'mission{mission}_{name}.csv')
    return matrices['pooled'], summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    integrity = json.loads((EVIDENCE / 'raw-integrity.json').read_text())
    assert integrity['verified']
    fingerprints = {f['path']: f for m in integrity['missions'] for f in m['extracted_files']}
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    frozen = {'protocol_sha256': sha(PROTOCOL), 'script_sha256': sha(__file__),
              'integrity_sha256': sha(EVIDENCE / 'raw-integrity.json'),
              'training_end_exclusive': {'1': '2006-01-01', '2': '2001-01-01'},
              'grid_seconds': 300, 'max_hold_seconds': 600,
              'assessment_used_for_selection': False}
    (output / 'frozen.json').write_text(json.dumps(frozen, indent=2) + '\n')
    meta = {}
    for mission in (1, 2):
        p = ROOT / f'data/ESA-Mission{mission}/channels.csv'
        assert sha(p) == fingerprints[str(p.relative_to(ROOT))]['sha256']
        meta[mission] = pd.read_csv(p)
        meta[mission].insert(0, 'mission', mission)
    pd.concat(meta.values()).to_csv(output / 'metadata_inventory.csv', index=False)
    anchors = meta[1].loc[meta[1].Channel.isin([f'channel_{n}' for n in range(40, 48)])]
    types = set(zip(anchors.Subsystem, anchors['Physical Unit']))
    candidates = meta[2].loc[(meta[2].Target == 'YES') &
                           pd.Series(list(zip(meta[2].Subsystem, meta[2]['Physical Unit']))).isin(types)]
    assert len(anchors) == 8 and len(candidates) == 22
    candidates.to_csv(output / 'mission2_metadata_candidates.csv', index=False)
    coverage, profiles, plots = [], [], {}
    for mission, inventory in [(1, anchors), (2, candidates)]:
        end = pd.Timestamp(frozen['training_end_exclusive'][str(mission)])
        grid = pd.date_range('2000-01-01', end, freq='5min', inclusive='left')
        labels_path = ROOT / f'data/ESA-Mission{mission}/labels.csv'
        assert sha(labels_path) == fingerprints[str(labels_path.relative_to(ROOT))]['sha256']
        labels = pd.read_csv(labels_path)
        labels['lo'] = timestamp_ns(labels.StartTime)
        labels['hi'] = timestamp_ns(labels.EndTime)
        labels = labels[(labels.lo < end.value) & (labels.hi >= grid[0].value)].copy()
        data = {}
        for channel in sorted(inventory.Channel, key=lambda c: int(c.split('_')[1])):
            path = ROOT / f'data/ESA-Mission{mission}/channels/{channel}.zip'
            assert sha(path) == fingerprints[str(path.relative_to(ROOT))]['sha256']
            before = path.stat()
            raw = pd.read_pickle(path)
            # Full pickle deserialization is unavoidable. Restrict values before any statistics.
            raw = raw.loc[(raw.index >= grid[0]) & (raw.index < end)]
            assert len(raw.columns) == 1 and np.issubdtype(raw.iloc[:, 0].dtype, np.number)
            t = timestamp_ns(raw.index)
            values = raw.iloc[:, 0].to_numpy()
            intervals = labels.loc[labels.Channel == channel, ['lo', 'hi']].to_numpy()
            x = held_values(t, values, grid.asi8, intervals)
            data[channel] = x
            coverage.append(dict(mission=mission, channel=channel, raw_training_rows=len(t),
                                 first_training_time=str(raw.index[0]), last_training_time=str(raw.index[-1]),
                                 median_raw_spacing_seconds=float(np.median(np.diff(t)) / 1e9),
                                 grid_rows=len(grid), eligible_grid_rows=int(np.isfinite(x).sum()),
                                 coverage=float(np.isfinite(x).mean())))
            assert (before.st_size, before.st_mtime_ns) == (path.stat().st_size, path.stat().st_mtime_ns)
            for quarter in grid.to_period('Q').unique():
                profiles.append(dict(mission=mission, channel=channel, quarter=str(quarter),
                                     **profile(x[grid.to_period('Q') == quarter])))
            print(f'mission{mission} {channel}: training coverage {np.isfinite(x).mean():.3%}', flush=True)
            del raw, values, x
        frame = pd.DataFrame(data, index=grid)
        np.savez_compressed(output / f'mission{mission}_training_diagnostic_grid.npz',
                            times=grid.asi8, channels=frame.columns.to_numpy(dtype=str), values=frame.to_numpy())
        plots[mission], _ = pair_tables(frame, mission, output)
        del frame, data
    pd.DataFrame(coverage).to_csv(output / 'coverage.csv', index=False)
    features = pd.DataFrame(profiles)
    features.to_csv(output / 'quarter_profiles.csv', index=False)
    pooled = features.groupby(['mission', 'channel'])[list(QKEYS + AKEYS + DKEYS)].median()
    pooled.to_csv(output / 'median_quarter_profiles.csv')
    rankings = []
    for a in anchors.itertuples(index=False):
        row = anchors.loc[anchors.Channel == a.Channel].iloc[0]
        matches = candidates.loc[(candidates.Subsystem == row.Subsystem) &
                                 (candidates['Physical Unit'] == row['Physical Unit'])]
        for b in matches.Channel:
            families = []
            for keys in (QKEYS, AKEYS, DKEYS):
                u, v = pooled.loc[(1, a.Channel), list(keys)], pooled.loc[(2, b), list(keys)]
                families.append(float(np.abs(u - v).mean()) if u.notna().all() and v.notna().all() else np.nan)
            rankings.append(dict(mission1=a.Channel, mission2=b, subsystem=row.Subsystem,
                                 physical_unit=row['Physical Unit'], distribution_distance=families[0],
                                 autocorrelation_distance=families[1], changes_distance=families[2],
                                 heuristic_distance=float(np.mean(families))))
    ranking = pd.DataFrame(rankings)
    ranking['rank_within_metadata_pool'] = ranking.groupby('mission1').heuristic_distance.rank(method='min')
    ranking.sort_values(['mission1', 'rank_within_metadata_pool']).to_csv(output / 'cross_mission_similarity.csv', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)
    for i, mission in enumerate((1, 2)):
        for j, (matrix, title) in enumerate(zip((plots[mission][0], plots[mission][2]), ('Levels', 'Adjacent changes'))):
            ax = axes[i, j]
            im = ax.imshow(matrix, vmin=-1, vmax=1, cmap='RdBu_r')
            names = [x.split('_')[1] for x in matrix.columns]
            ax.set_xticks(range(len(names)), names, rotation=90)
            ax.set_yticks(range(len(names)), names)
            ax.set_title(f'Mission {mission} · {title} · training only')
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.8, label='Within-mission Pearson r; blank = undefined')
    fig.savefig(output / 'training_correlations.png', dpi=160)
    plt.close(fig)
    assert sha(PROTOCOL) == frozen['protocol_sha256'] and sha(__file__) == frozen['script_sha256']
    receipt = {'complete': True, 'frozen': frozen, 'raw_integrity_verified': True,
               'profiled_channels': len(coverage), 'quarter_profiles': len(profiles),
               'cross_mission_candidate_pairs': len(ranking),
               'files': [{'path':p.name, 'bytes':p.stat().st_size, 'sha256':sha(p)}
                         for p in sorted(output.iterdir()) if p.is_file()]}
    (output / 'complete.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
