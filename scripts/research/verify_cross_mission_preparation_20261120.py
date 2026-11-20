"""Independent pandas source lookup, mask/window and scaler preparation preflight."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from esa_thesis.calibration_audit import digest
from esa_thesis.cross_mission_study_v2 import OUTPUT, PREP, verify_prepared


def main():
    p=verify_prepared();records=[];checked=0
    for key in p['splits']:
        folder=PREP/key;meta=json.loads((folder/'metadata.json').read_text())
        m=meta['mission'];split=meta['split'];step=meta['cadence_seconds']*10**9
        assert meta['max_age_seconds']==[900 if m==1 and c in ('channel_62','channel_63') else 600 for c in meta['features']]
        times=np.load(folder/'times.npy');x=np.load(folder/'values.npy',mmap_mode='r')
        av=np.load(folder/'available.npy',mmap_mode='r');age=np.load(folder/'age_ns.npy',mmap_mode='r')
        excluded=np.load(folder/'excluded.npy');start,end=map(pd.Timestamp,meta['bounds'])
        assert times[0]==start.value and times[-1]<end.value and (np.diff(times)==step).all()
        annotations=pd.read_csv(ROOT/f'data/ESA-Mission{m}/labels.csv')
        lo=pd.to_datetime(annotations.StartTime,utc=True).astype('int64').to_numpy()
        hi=pd.to_datetime(annotations.EndTime,utc=True).astype('int64').to_numpy()
        points=set(np.linspace(0,len(times)-1,211,dtype=int))|set(range(10))|set(range(len(times)-10,len(times)))
        for edge in np.r_[lo,hi]:
            i=int(np.searchsorted(times,edge));points.update(j for j in (i-1,i,i+1) if 0<=j<len(times))
        ix=np.array(sorted(points));q=times[ix];expect_excluded=np.zeros(len(q),bool)
        if split=='train':
            for a,b in zip(lo,hi):expect_excluded|=(q<=b)&(q+step>a)
        for j,ch in enumerate(meta['features']):
            raw=pd.read_pickle(ROOT/f'data/ESA-Mission{m}/channels/{ch}.zip').iloc[:,0]
            raw.index=pd.DatetimeIndex(raw.index).tz_localize(None)
            raw=raw[(raw.index>=start)&(raw.index<end)]
            qt=pd.to_datetime(q);positions=raw.index.get_indexer(qt,method='pad')
            expected=raw.reindex(qt,method='ffill',tolerance=pd.Timedelta(seconds=900 if m==1 and ch in ('channel_62','channel_63') else 600)).to_numpy(dtype=np.float32)
            expected[~np.isfinite(expected)]=np.nan
            np.testing.assert_allclose(x[ix,j],expected,rtol=0,atol=0,equal_nan=True)
            np.testing.assert_array_equal(av[ix,j],np.isfinite(expected))
            source=raw.index.asi8[np.maximum(positions,0)]
            np.testing.assert_array_equal(age[ix,j],np.where(positions>=0,q-source,-1))
            expect_excluded|=~np.isfinite(expected)
            if split=='train':
                for a,b in zip(lo[annotations.Channel==ch],hi[annotations.Channel==ch]):expect_excluded|=(source>=a)&(source<=b)
            checked+=len(ix)
        np.testing.assert_array_equal(excluded[ix]!=0,expect_excluded)
        record=dict(key=key,source_spots=len(ix)*len(meta['features']),rows=len(times))
        if split=='train':
            prefix=np.r_[0,np.cumsum(excluded!=0)]
            starts=np.arange(0,len(times)-255,16)
            starts=starts[prefix[starts+256]==prefix[starts]]
            if len(starts)==0:raise ValueError('No nominal training windows')
            if key=='m1/train':assert len(starts)==349383, 'Does not match approved training-only feasibility check'
            record.update(nominal_windows=len(starts),batches_per_epoch=int(np.ceil(len(starts)/32)))
            record['fallback_scale_channels']=[]
            if key=='transfer_m2/train':norm=json.loads((OUTPUT/'target-normalization.json').read_text())
            for j,ch in enumerate(meta['features']):
                values=np.array(x[:,j][excluded==0]);values=values[np.isfinite(values)]
                if len(values)<100:raise ValueError('Insufficient nominal scaler data')
                median=np.float32(np.median(values));iqr=np.float32(np.quantile(values,.75)-np.quantile(values,.25))
                if abs(iqr)<1e-8:record['fallback_scale_channels'].append(ch);iqr=np.float32(1)
                if key=='transfer_m2/train':
                    assert norm['features'][j]==ch
                    assert median==np.float32(norm['median'][j]) and iqr==np.float32(norm['iqr'][j])
        records.append(record);print('Verified',key,record,flush=True)
    receipt=dict(passed=True,raw_source_sample_checks=checked,splits=records,assessment_values_used=False,
                 limitations='All prepared file hashes verified; pandas raw value/age/availability/mask lookup is sampled, not a full second resampling.',
                 preparation_sha256=digest(OUTPUT/'preparation.json'),verifier_sha256=digest(__file__))
    (OUTPUT/'preflight-verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('INDEPENDENT PREPARATION PREFLIGHT PASSED',checked,flush=True)

if __name__=='__main__':main()
