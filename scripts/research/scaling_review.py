"""Measure existing scaler effects on 2000–2005 only; never modify data."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    out=parser.parse_args().output;out.mkdir(parents=True,exist_ok=False)
    source=Path(__file__).resolve().parents[2]/'data/preprocessed/multivariate/ESA-Mission1-semi-supervised/84_months.train.csv'
    before=source.stat();cols=pd.read_csv(source,nrows=0).columns
    features=[c for c in cols if c!='timestamp' and not c.startswith('is_anomaly_')]
    labels=[c for c in cols if c.startswith('is_anomaly_')]
    assert len(features)==87 and len(labels)==87
    n=6312960
    x=np.memmap(out/'training_values.tmp',mode='w+',dtype='float32',shape=(n,87))
    normal=np.zeros(n,dtype=bool);offset=0
    for frame in pd.read_csv(source,chunksize=50000):
        frame=frame[frame.timestamp<'2006-01-01']
        if frame.empty:break
        v=frame[features].to_numpy(dtype=np.float32);lab=frame[labels].to_numpy()
        assert np.isfinite(v).all() and np.isin(lab,[0,1,2,3]).all()
        size=len(v);x[offset:offset+size]=v;normal[offset:offset+size]=(lab==0).all(axis=1);offset+=size
        if offset%1000000==0:print('Loaded',offset,flush=True)
    assert offset==n and normal.sum()==5615041
    rows=[]
    for j,f in enumerate(features):
        v=np.array(x[:,j]);pts=v[normal];q25,q75=np.percentile(pts,[25,75]);median=np.float32(np.median(pts));iqr=q75-q25
        fallback=abs(iqr)<1e-8;scale=np.float32(1 if fallback else iqr);z=(v-median)/scale
        rows.append(dict(feature=f,median=float(median),raw_iqr=iqr,effective_scale=float(scale),iqr_fallback=bool(fallback),nominal_min=float(pts.min()),nominal_max=float(pts.max()),nominal_nonmedian_count=int((pts!=median).sum()),nominal_clip_fraction=float((abs(z[normal])>10).mean()),all_training_clip_fraction=float((abs(z)>10).mean()),excluded_clip_fraction=float((abs(z[~normal])>10).mean()),nominal_scaled_std=float(np.clip(z[normal],-10,10).std(dtype=np.float64))))
    pd.DataFrame(rows).to_csv(out/'scaling_effects.csv',index=False)
    after=source.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    (out/'protocol.json').write_text(json.dumps(dict(interval='[2000-01-01,2006-01-01)',rows=n,nominal_rows=int(normal.sum()),features=features,mask='all 87 paired labels zero',method='Exact full-resolution float32 nominal median/IQR; IQR <1e-8 replaced with 1; clip +/-10, matching current scaler',excluded='All non-globally-nominal training samples; not a test metric',source_size=before.st_size,source_mtime_ns=before.st_mtime_ns),indent=2))
    del x
    (out/'training_values.tmp').unlink()
    print('Completed',out,flush=True)
if __name__=='__main__':main()
