"""Read-only channel audit of the first training fold, never assessment data."""
import argparse
import hashlib
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
    source = root/'data/preprocessed/multivariate/ESA-Mission1-semi-supervised/84_months.train.csv'
    header = pd.read_csv(source, nrows=0).columns.tolist()
    features = [c for c in header if c != 'timestamp' and not c.startswith('is_anomaly_')]
    labels = ['is_anomaly_'+c for c in features]
    if not set(labels).issubset(header):
        raise ValueError('Missing paired annotations')
    k=len(features);count=0;nominal_count=0;previous_time=None;previous_x=None;previous_nominal=False
    sums=np.zeros(k);squares=np.zeros(k);lo=np.full(k,np.inf);hi=np.full(k,-np.inf)
    unchanged=np.zeros(k,dtype=np.int64);steps=0;nonfinite=np.zeros(k,dtype=np.int64)
    samples=[];sample_times=[];rows_before=0
    for frame in pd.read_csv(source,chunksize=50000):
        t=pd.to_datetime(frame.timestamp,utc=True)
        keep=t<pd.Timestamp('2003-01-01',tz='UTC');frame=frame.loc[keep];t=t[keep]
        if frame.empty:break
        ns=t.astype('int64').to_numpy()
        if previous_time is not None and ns[0]-previous_time!=30_000_000_000:raise ValueError('Gap')
        if not np.all(np.diff(ns)==30_000_000_000):raise ValueError('Nonuniform time grid')
        previous_time=ns[-1]
        x=frame[features].to_numpy(dtype=np.float64);y=frame[labels].to_numpy()
        if not np.isin(y,[0,1,2,3]).all():raise ValueError('Invalid annotation')
        bad=~np.isfinite(x);nonfinite+=bad.sum(axis=0)
        if bad.any():raise ValueError('Nonfinite values; do not silently clean')
        nominal=(y==0).all(axis=1);v=x[nominal]
        nominal_count+=len(v);count+=len(x)
        sums+=v.sum(axis=0);squares+=(v*v).sum(axis=0)
        if len(v):lo=np.minimum(lo,v.min(axis=0));hi=np.maximum(hi,v.max(axis=0))
        pair=nominal[1:] & nominal[:-1]
        unchanged+=(x[1:][pair]==x[:-1][pair]).sum(axis=0);steps+=int(pair.sum())
        if previous_x is not None and previous_nominal and nominal[0]:
            unchanged+=(x[0]==previous_x);steps+=1
        previous_x=x[-1].copy();previous_nominal=nominal[-1]
        # Fixed hourly grid across all months. Selection precedes nominal filtering.
        selected=((np.arange(len(x))+rows_before)%120==0)&nominal
        samples.append(x[selected]);sample_times.extend(t[selected].astype(str).tolist());rows_before+=len(x)
        if count%500000==0:print('Audited',count,'rows',flush=True)
        if not keep.all():break
    sample=pd.DataFrame(np.concatenate(samples),columns=features)
    sample.insert(0,'timestamp',sample_times)
    sample.to_csv(args.output/'nominal_hourly_sample.csv',index=False)
    metadata=pd.read_csv(root/'data/ESA-Mission1/channels.csv').set_index('Channel')
    rows=[]
    for j,c in enumerate(features):
        vals=sample[c];freq=vals.value_counts(normalize=True)
        info=metadata.loc[c].to_dict() if c in metadata.index else {}
        constant=bool(lo[j]==hi[j]);dominant=float(freq.iloc[0])
        rows.append({'feature':c,'kind':'telemetry' if c.startswith('channel_') else 'telecommand',
          'subsystem':info.get('Subsystem','telecommand'),'unit_code':info.get('Physical Unit','not supplied'),
          'group':str(info.get('Group','')),'target':info.get('Target','not supplied'),
          'nominal_rows':nominal_count,'sample_rows':len(vals),'nonfinite':int(nonfinite[j]),
          'min':lo[j],'max':hi[j],'mean':sums[j]/nominal_count,
          'std':float(np.sqrt(max(0,squares[j]/nominal_count-(sums[j]/nominal_count)**2))),
          'sample_unique':int(vals.nunique()),'sample_mode_fraction':dominant,
          'unchanged_30s_fraction':float(unchanged[j]/steps),
          'constant_training_nominal':constant,
          'review_flag':'constant in nominal training' if constant else ('sample mode >=99.9%' if dominant>=.999 else 'variable'),
          'physical_meaning':'anonymized; no engineering interpretation inferred'})
    profile=pd.DataFrame(rows);profile.to_csv(args.output/'channel_profile.csv',index=False)
    variable=[r['feature'] for r in rows if r['sample_unique']>1]
    pearson=sample[variable].corr();spearman=sample[variable].rank().corr()
    pearson.to_csv(args.output/'pearson.csv');spearman.to_csv(args.output/'spearman.csv')
    pairs=[]
    for i,a in enumerate(variable):
        for b in variable[i+1:]:
            r=float(pearson.loc[a,b]);s=float(spearman.loc[a,b])
            if abs(r)>=.95 or abs(s)>=.95:
                by_year=[]
                for year in ['2000','2001','2002']:
                    v=sample.loc[sample.timestamp.str.startswith(year),[a,b]]
                    by_year.append(float(v[a].corr(v[b])) if v[a].nunique()>1 and v[b].nunique()>1 else None)
                pairs.append({'feature_a':a,'feature_b':b,'pearson':r,'spearman':s,
                              'pearson_2000':by_year[0],'pearson_2001':by_year[1],'pearson_2002':by_year[2]})
    pd.DataFrame(pairs,columns=['feature_a','feature_b','pearson','spearman','pearson_2000','pearson_2001','pearson_2002']).to_csv(args.output/'correlated_pairs.csv',index=False)
    # Monthly summaries preserve the ordering and use only the sampled nominal points.
    sample['month']=sample.timestamp.str[:7]
    monthly=sample.groupby('month')[features].median();monthly.to_csv(args.output/'monthly_sample_medians.csv')
    protocol={'scope':'2000-01-01 inclusive to 2003-01-01 exclusive; first-fold training only',
      'source':str(source.relative_to(root)),'source_size':source.stat().st_size,'source_mtime_ns':source.stat().st_mtime_ns,
      'rows':count,'nominal_rows':nominal_count,'features':len(features),'hourly_nominal_sample_rows':len(sample),
      'normal_definition':'all 87 paired annotation columns equal zero',
      'sampling':'one fixed-grid row every 120 records, then retain nominal rows; no interpolation',
      'constant_definition':'exact min=max across all nominal training rows',
      'correlation':'Pearson and rank Spearman on hourly nominal sample; flagged if either absolute value >=0.95',
      'limitations':'Prepared data only; imputation can hide raw missingness. Hourly sampling can miss short events or alias cycles. No physical unit meanings inferred. Constant nominal signals may still detect anomalies. No channels removed; no tuning or assessment access.',
      'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    print(json.dumps(protocol,indent=2));print(profile.review_flag.value_counts().to_string())


if __name__=='__main__':main()
