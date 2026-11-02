"""Independent saved-grid correlation replay and source sample spot checks."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def corr(a, b, ranked=False):
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan
    if ranked:
        a, b = rankdata(a), rankdata(b)
    a, b = a - a.mean(), b - b.mean()
    return np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    receipt = json.loads((args.run / 'complete.json').read_text())
    assert receipt['complete']
    for f in receipt['files']:
        path = args.run / f['path']
        assert path.stat().st_size == f['bytes'] and digest(path) == f['sha256']
    assert digest(root / 'scripts/research/mission_cross_channel_review.py') == receipt['frozen']['script_sha256']
    assert digest(root / 'docs/research/mission-cross-channel-protocol-2026-11-02.md') == receipt['frozen']['protocol_sha256']
    integrity = json.loads((root / 'docs/research/tables/mission-cross-channel-2026-11-02/raw-integrity.json').read_text())
    source_hash = {f['path']:f['sha256'] for m in integrity['missions'] for f in m['extracted_files']}
    samples, rows = 0, 0
    for mission in (1, 2):
        with np.load(args.run / f'mission{mission}_training_diagnostic_grid.npz') as z:
            times, channels, values = z['times'], z['channels'], z['values']
        dates = pd.to_datetime(times)
        cutoff = pd.Timestamp(receipt['frozen']['training_end_exclusive'][str(mission)])
        assert dates.min() >= pd.Timestamp('2000-01-01') and dates.max() < cutoff
        assert (np.diff(times) == 300 * 10**9).all()
        labels = pd.read_csv(root / f'data/ESA-Mission{mission}/labels.csv')
        labels['start'] = pd.to_datetime(labels.StartTime, utc=True).dt.tz_localize(None)
        labels['end'] = pd.to_datetime(labels.EndTime, utc=True).dt.tz_localize(None)
        labels = labels[(labels.start < cutoff) & (labels.end >= dates.min())]
        rng = np.random.default_rng(20261102 + mission)
        for j, c in enumerate(channels):
            path = root / f'data/ESA-Mission{mission}/channels/{c}.zip'
            assert digest(path) == source_hash[str(path.relative_to(root))]
            raw = pd.read_pickle(path)
            raw = raw.loc[(raw.index >= dates.min()) & (raw.index < cutoff)]
            lab = labels[labels.Channel == c]
            edge = np.r_[dates.searchsorted(lab.start), dates.searchsorted(lab.end)]
            indices = np.unique(np.clip(np.r_[0, len(dates)-1,
                                rng.integers(0, len(dates), 64), edge-1, edge, edge+1], 0, len(dates)-1))
            for i in indices:
                when = dates[i]
                source_i = raw.index.get_indexer([when], method='pad', tolerance=pd.Timedelta(minutes=10))[0]
                expected = np.nan
                if source_i >= 0:
                    source_time = raw.index[source_i]
                    crosses = ((lab.start <= when) & (lab.end >= source_time)).any()
                    if not crosses:
                        expected = float(raw.iloc[source_i, 0])
                np.testing.assert_allclose(values[i, j], expected, rtol=0, atol=0, equal_nan=True)
                samples += 1
            del raw
        print(f'Mission {mission}: raw-source grid checks passed', flush=True)
        table = pd.read_csv(args.run / f'mission{mission}_pair_blocks.csv')
        quarter = dates.to_period('Q').astype(str).to_numpy()
        index = {c:i for i, c in enumerate(channels)}
        for label, block in table.groupby('block', sort=False):
            mask = np.ones(len(times), bool) if label == 'pooled' else quarter == label
            x = values[mask]
            q = quarter[mask]
            changes = np.diff(x, axis=0)
            changes[q[1:] != q[:-1]] = np.nan
            for row in block.itertuples():
                a, b = index[row.a], index[row.b]
                np.testing.assert_allclose([corr(x[:, a], x[:, b]), corr(x[:, a], x[:, b], ranked=True),
                                            corr(changes[:, a], changes[:, b])],
                                           [row.pearson, row.spearman, row.delta_pearson],
                                           rtol=1e-9, atol=1e-9, equal_nan=True)
                assert ((np.isfinite(x[:, a])) & (np.isfinite(x[:, b]))).sum() == row.pairs
                assert (np.isfinite(changes[:, a]) & np.isfinite(changes[:, b])).sum() == row.delta_pairs
                rows += 1
        print(f'Mission {mission}: all correlation rows replayed', flush=True)
    features = pd.read_csv(args.run / 'quarter_profiles.csv')
    groups = [('q01','q10','q25','q75','q90','q99'),
              ('acf_1','acf_6','acf_12','acf_72','acf_288'),
              ('log_delta_median','log_delta_p95','delta_zero_fraction')]
    medians = features.groupby(['mission','channel']).median(numeric_only=True)
    distances = pd.read_csv(args.run / 'cross_mission_similarity.csv')
    for row in distances.itertuples():
        means=[]
        for columns in groups:
            a = medians.loc[(1,row.mission1), list(columns)].to_numpy()
            b = medians.loc[(2,row.mission2), list(columns)].to_numpy()
            means.append(np.mean(np.abs(a-b)))
        np.testing.assert_allclose(means + [np.mean(means)],
                                   [row.distribution_distance,row.autocorrelation_distance,row.changes_distance,row.heuristic_distance],
                                   rtol=1e-9, atol=1e-9, equal_nan=True)
    ranks = distances.groupby('mission1').heuristic_distance.rank(method='min')
    np.testing.assert_allclose(ranks, distances.rank_within_metadata_pool, equal_nan=True)
    result={'verified':True,'saved_file_hashes':len(receipt['files']),
            'source_samples_replayed':samples,'correlation_rows_replayed':rows,
            'similarity_distances_replayed':len(distances),
            'assessment_values_used':False,
            'limitations':'Source checks are sampled; correlations fully replayed from saved descriptive grids. Not neural or assessment verification.'}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    main()
