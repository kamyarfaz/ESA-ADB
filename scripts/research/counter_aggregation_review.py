"""Training-only counter event retention; no dataset transformation or model run."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
STEP=30_000_000_000

def summarize(ts,values,start,end):
    assert len(ts)>1 and (np.diff(ts)>0).all() and np.isfinite(values).all()
    d=np.diff(values.astype(np.float64));changed=d!=0
    left=ts[:-1][changed];right=ts[1:][changed];delta=d[changed]
    old_grid=((left+STEP-1)//STEP)*STEP
    available=((right+STEP-1)//STEP)*STEP
    valid=(available>=start)&(available<end)
    bins,counts=np.unique(available[valid],return_counts=True)
    sums=np.zeros(len(bins));absolute=np.zeros(len(bins))
    ix=np.searchsorted(bins,available[valid]);np.add.at(sums,ix,delta[valid]);np.add.at(absolute,ix,abs(delta[valid]))
    assert counts.sum()==valid.sum() and np.all(absolute>0)
    assert np.all((available-right>=0)&(available-right<STEP))
    return dict(raw_changes=int(changed.sum()),old_changes_missed_by_grid=int((old_grid>=right).sum()),causal_changes_retained=int(valid.sum()),deferred_beyond_interval=int((~valid).sum()),occupied_causal_bins=len(bins),multiple_change_bins=int((counts>1).sum()),net_zero_change_bins=int((sums==0).sum()),max_change_gap_seconds=float(np.max((right-left)/1e9)) if len(right) else 0,max_bin_delay_seconds=float(np.max((available-right)/1e9)) if len(right) else 0)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);out=p.parse_args().output;out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[2];start=pd.Timestamp('2000-01-01').value;end=pd.Timestamp('2006-01-01').value;rows=[]
    for c in range(4,12):
        name=f'channel_{c}';path=root/'data/ESA-Mission1/channels'/f'{name}.zip';before=path.stat()
        frame=pd.read_pickle(path);series=frame.loc[(frame.index>='2000-01-01')&(frame.index<'2006-01-01'),name]
        ts=series.index.asi8;values=series.to_numpy()
        r=summarize(ts,values,start,end);r.update(feature=name,raw_samples=len(ts),max_sample_gap_seconds=float(np.diff(ts).max()/1e9),first_observation=str(series.index[0]),last_observation=str(series.index[-1]));rows.append(r)
        assert (before.st_size,before.st_mtime_ns)==(path.stat().st_size,path.stat().st_mtime_ns)
        print(name,r['raw_changes'],flush=True)
    pd.DataFrame(rows).to_csv(out/'retention.csv',index=False)
    (out/'protocol.json').write_text(json.dumps(dict(interval='[2000-01-01,2006-01-01)',availability='Difference assigned to later observation, then ceiling 30-second boundary; bin (t-30s,t]',features='candidate count, signed sum and absolute sum; no arrays installed',boundary='First observation has no delta; both endpoints inside training; bins at/after 2006 excluded',limitations='Known numeric changes only, not event recall. Native timestamps assumed availability; no packet arrival times. Large gaps/reset semantics unresolved. No nominal mask, calibration or assessment used. Old missed counts describe forward-hold geometry, not full annotation-aware preprocessing.'),indent=2))
if __name__=='__main__':main()
