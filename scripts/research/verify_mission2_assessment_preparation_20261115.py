"""Independent source lookup and training-mask preflight; assessment-only completion audit."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from esa_thesis.calibration_audit import digest
from esa_thesis.full_combinations import nominal_starts
from esa_thesis.mission2_study import OUTPUT, PREP, RAW, verify_prepared

p=verify_prepared()
annotations=pd.read_csv(RAW/'labels.csv')
lo=pd.to_datetime(annotations.StartTime,utc=True).astype('int64').to_numpy()
hi=pd.to_datetime(annotations.EndTime,utc=True).astype('int64').to_numpy()
checked=0;records=[]
for split in ('assessment',):
    folder=PREP/split;meta=json.loads((folder/'metadata.json').read_text())
    times=np.load(folder/'times.npy');x=np.load(folder/'values.npy',mmap_mode='r')
    availability=np.load(folder/'available.npy',mmap_mode='r');ages=np.load(folder/'age_ns.npy',mmap_mode='r')
    excluded=np.load(folder/'excluded.npy')
    start,end=map(pd.Timestamp,meta['bounds']);start_ns,end_ns=start.value,end.value
    assert (np.diff(times)==18*10**9).all() and times[0]==start_ns and times[-1]<end_ns
    # Deterministic edges plus annotation-boundary neighbors and uniform samples.
    ix=set(np.linspace(0,len(times)-1,200,dtype=int))|set(range(10))|set(range(len(times)-10,len(times)))
    for edge in np.r_[lo,hi]:
        n=int(np.searchsorted(times,edge))
        ix.update(k for k in (n-1,n,n+1) if 0<=k<len(times))
    ix=np.array(sorted(ix));q=times[ix];expected_excluded=np.zeros(len(q),bool)
    if split=='train':
        for a,b in zip(lo,hi):expected_excluded|=(q<=b)&(q+18*10**9>a)
    for j,ch in enumerate(p['features']):
        raw=pd.read_pickle(RAW/'channels'/(ch+'.zip')).iloc[:,0]
        raw.index=pd.DatetimeIndex(raw.index).tz_localize(None)
        raw=raw[(raw.index>=start)&(raw.index<end)]
        qt=pd.to_datetime(q)
        positions=raw.index.get_indexer(qt,method='pad')
        expected=raw.reindex(qt,method='ffill',tolerance=pd.Timedelta(seconds=600)).to_numpy(dtype=np.float32)
        expected[~np.isfinite(expected)]=np.nan
        np.testing.assert_allclose(x[ix,j],expected,rtol=0,atol=0,equal_nan=True)
        np.testing.assert_array_equal(availability[ix,j],np.isfinite(expected))
        source=raw.index.asi8[np.maximum(positions,0)]
        np.testing.assert_array_equal(ages[ix,j],np.where(positions>=0,q-source,-1))
        expected_excluded|=~np.isfinite(expected)
        if split=='train':
            for a,b in zip(lo[annotations.Channel==ch],hi[annotations.Channel==ch]):
                expected_excluded|=(source>=a)&(source<=b)
        checked+=len(ix)
    np.testing.assert_array_equal(excluded[ix]!=0,expected_excluded)
    record=dict(split=split,rows=len(times),source_spots=len(ix)*len(p['features']))
    if split=='train':
        starts=nominal_starts(excluded)
        c=np.r_[0,np.cumsum(excluded!=0)]
        assert (c[starts+256]==c[starts]).all()
        record.update(nominal_windows=len(starts),batches_per_epoch=int(np.ceil(len(starts)/64)))
    records.append(record)
receipt=dict(passed=True,raw_source_sample_checks=checked,splits=records,assessment_values_used=True,
             limitations='Prepared file hashes all verified; raw source/mask lookup sampled with pandas independent lookup, not full replay.',
             verifier_sha256=digest(Path(__file__)),preparation_sha256=digest(OUTPUT/'preparation.json'))
path=Path('docs/research/tables/mission2-three-technique-2026-11-14/assessment-preparation-verification.json');path.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
