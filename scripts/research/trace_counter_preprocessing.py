"""Trace training counter changes through forward-hold sampling and nominal mask."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    out = parser.parse_args().output
    out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    channels = ['channel_'+str(i) for i in range(4,12)]
    raw = {}
    rows = {}
    for c in channels:
        x = pd.read_pickle(root/'data/ESA-Mission1/channels'/f'{c}.zip')
        x = x.loc[(x.index >= '2000-01-01') & (x.index < '2003-01-01'), c]
        ts = x.index.asi8
        d = np.diff(x.to_numpy())
        changed = np.flatnonzero(d != 0)
        # A hold interval [left,right) survives if it contains any 30s grid point.
        first_grid = ((ts[:-1] + 30_000_000_000-1)//30_000_000_000)*30_000_000_000
        survives = first_grid < ts[1:]
        raw[c] = (ts, d)
        rows[c] = dict(feature=c, raw_changes=len(changed),
            changes_with_grid_point=int(survives[changed].sum()),
            changes_missed_by_grid=int((~survives[changed]).sum()),
            compared_rows=0, mismatch_rows=0, max_abs_error=0.,
            prepared_nonzero_rows=0, nominal_nonzero_rows=0,
            excluded_nonzero_rows=0)
    source = root/'data/preprocessed/multivariate/ESA-Mission1-semi-supervised/84_months.train.csv'
    stat = source.stat()
    labels = [c for c in pd.read_csv(source,nrows=0).columns if c.startswith('is_anomaly_')]
    assert len(labels)==87
    causes = {c:np.zeros(len(labels),dtype=np.int64) for c in channels}
    total = 0
    for frame in pd.read_csv(source,usecols=['timestamp']+channels+labels,chunksize=50000):
        t = pd.to_datetime(frame.timestamp)
        keep = (t >= '2000-01-01') & (t < '2003-01-01')
        if not keep.any(): break
        frame = frame.loc[keep]; ns = t[keep].astype('int64').to_numpy()
        y = frame[labels].to_numpy()
        assert np.isin(y,[0,1,2,3]).all()
        nominal = (y==0).all(axis=1)
        for c in channels:
            ts,d = raw[c]
            ix = np.searchsorted(ts,ns,side='right')-1
            valid = (ix>=0)&(ix<len(d))
            actual = frame[c].to_numpy()
            assert np.isfinite(actual).all()
            error = np.abs(actual[valid]-d[ix[valid]])
            r=rows[c]
            r['compared_rows']+=int(valid.sum())
            r['mismatch_rows']+=int((~np.isclose(actual[valid],d[ix[valid]],rtol=1e-6,atol=1e-12)).sum())
            r['max_abs_error']=max(r['max_abs_error'],float(error.max()))
            nonzero=actual!=0
            r['prepared_nonzero_rows']+=int(nonzero.sum())
            r['nominal_nonzero_rows']+=int((nonzero&nominal).sum())
            r['excluded_nonzero_rows']+=int((nonzero&~nominal).sum())
            causes[c]+=(y[nonzero]!=0).sum(axis=0)
        total+=len(frame)
        if total%500000==0: print('Traced',total,'rows',flush=True)
        if not keep.all(): break
    assert total==3156480
    assert (stat.st_size,stat.st_mtime_ns)==(source.stat().st_size,source.stat().st_mtime_ns)
    result=pd.DataFrame(rows.values())
    assert (result.raw_changes==result.changes_with_grid_point+result.changes_missed_by_grid).all()
    assert (result.prepared_nonzero_rows==result.nominal_nonzero_rows+result.excluded_nonzero_rows).all()
    result.to_csv(out/'trace.csv',index=False)
    pd.DataFrame(causes,index=labels).rename_axis('nonzero_label').to_csv(out/'exclusion_labels.csv')
    (out/'protocol.json').write_text(json.dumps(dict(
        interval='[2000-01-01,2003-01-01)', rows=total,
        comparison='Forward differences then forward hold; only samples with both raw difference endpoints inside training.',
        tolerance='rtol=1e-6, atol=1e-12; raw float32 vs CSV decimal representation',
        nominal='all 87 paired labels zero',
        caveat='Exclusion-label counts overlap. Reconstructs numeric hold only, not annotation restoration. Source size/mtime checked for prepared CSV; not cryptographic integrity.',
        source_bytes=stat.st_size,source_mtime_ns=stat.st_mtime_ns),indent=2)+'\n')
    print(result.to_string(index=False))


if __name__=='__main__': main()
