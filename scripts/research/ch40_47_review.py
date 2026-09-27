"""Focused data diagnostics for fixed channels 40–47; train/calibration only."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);out=p.parse_args().output;out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[2];raw=root/'data/ESA-Mission1';features=[f'channel_{i}' for i in range(40,48)]
    meta=pd.read_csv(raw/'channels.csv');meta[meta.Channel.isin(features)].to_csv(out/'metadata.csv',index=False)
    rows=[]
    for c in features:
        f=pd.read_pickle(raw/'channels'/f'{c}.zip');s=f.loc[(f.index>='2000-01-01')&(f.index<'2007-01-01'),c]
        for year in range(2000,2007):
            a=s[s.index.year==year];v=a.to_numpy();gap=np.diff(a.index.asi8)/1e9
            rows.append(dict(channel=c,year=year,n=len(a),nonfinite=int((~np.isfinite(v)).sum()),duplicates=int(a.index.duplicated().sum()),median_gap_s=float(np.median(gap)),max_gap_s=float(gap.max()),gaps_over_300s=int((gap>300).sum()),unchanged_fraction=float((np.diff(v)==0).mean()),minimum=float(v.min()),maximum=float(v.max())))
        print('Native',c,flush=True)
    pd.DataFrame(rows).to_csv(out/'raw_yearly.csv',index=False)
    source=root/'data/preprocessed/multivariate/ESA-Mission1-semi-supervised/84_months.train.csv';before=source.stat()
    labels=[c for c in pd.read_csv(source,nrows=0).columns if c.startswith('is_anomaly_')];frames=[]
    for frame in pd.read_csv(source,usecols=['timestamp']+features+labels,chunksize=100000):
        t=pd.to_datetime(frame.timestamp);keep=t<'2007-01-01';frame=frame.loc[keep]
        if frame.empty:break
        x=frame[features].to_numpy();lab=frame[labels].to_numpy();assert np.isfinite(x).all() and np.isin(lab,[0,1,2,3]).all()
        small=frame[['timestamp']+features].copy();small['nominal']=(lab==0).all(axis=1);frames.append(small)
    data=pd.concat(frames,ignore_index=True);assert len(data)==7364160;data.timestamp=pd.to_datetime(data.timestamp);data['month']=data.timestamp.dt.to_period('M').astype(str)
    monthly=[]
    for month,g in data.groupby('month'):
        normal=g[g.nominal]
        for c in features:
            a=normal[c].to_numpy();monthly.append(dict(month=month,channel=c,rows=len(g),nominal_rows=len(a),nominal_p01=float(np.quantile(a,.01)) if len(a) else np.nan,nominal_median=float(np.median(a)) if len(a) else np.nan,nominal_p99=float(np.quantile(a,.99)) if len(a) else np.nan))
    m=pd.DataFrame(monthly);m.to_csv(out/'monthly_nominal.csv',index=False)
    for year,g in data.groupby(data.timestamp.dt.year):g.loc[g.nominal,features].corr().to_csv(out/f'pearson_{year}.csv')
    annotations=pd.read_csv(raw/'labels.csv').merge(pd.read_csv(raw/'anomaly_types.csv'),on='ID',validate='many_to_one')
    annotations.StartTime=pd.to_datetime(annotations.StartTime,utc=True);annotations.EndTime=pd.to_datetime(annotations.EndTime,utc=True)
    selected=annotations[(annotations.EndTime>='2000-01-01')&(annotations.StartTime<'2007-01-01')&annotations.Channel.isin(features)].copy()
    selected['clipped_duration_s']=(selected.EndTime.clip(upper=pd.Timestamp('2007-01-01',tz='UTC'))-selected.StartTime.clip(lower=pd.Timestamp('2000-01-01',tz='UTC'))).dt.total_seconds()
    selected.to_csv(out/'training_calibration_annotations.csv',index=False)
    event_rows=[]
    for year in range(2000,2007):
        a=annotations[(annotations.EndTime>=f'{year}-01-01')&(annotations.StartTime<f'{year+1}-01-01')&annotations.Category.isin(['Anomaly','Rare Event'])]
        ids=set(a.ID);local=set(a.loc[a.Channel.isin(features),'ID']);event_rows.append(dict(year=year,global_event_ids=len(ids),ids_annotated_on_40_47=len(local),ids_without_40_47_annotation=len(ids-local)))
    pd.DataFrame(event_rows).to_csv(out/'event_coverage.csv',index=False)
    fig,axes=plt.subplots(4,2,figsize=(14,12),sharex=True,constrained_layout=True)
    for ax,c in zip(axes.flat,features):
        a=m[m.channel==c];t=pd.to_datetime(a.month);ax.fill_between(t,a.nominal_p01,a.nominal_p99,alpha=.25);ax.plot(t,a.nominal_median,lw=1);ax.axvline(pd.Timestamp('2006-01-01'),color='red',ls='--');ax.set_title(c);ax.set_ylabel('Prepared units')
    fig.suptitle('Channels 40–47: monthly nominal median and 1–99% range\n2000–2005 training; 2006 calibration (after dashed line); no assessment data')
    fig.savefig(out/'monthly_profiles.png',dpi=140);plt.close(fig)
    assert (before.st_size,before.st_mtime_ns)==(source.stat().st_size,source.stat().st_mtime_ns)
    (out/'protocol.json').write_text(json.dumps(dict(features=features,interval='[2000-01-01,2007-01-01)',training='2000–2005',calibration='2006',prepared_rows=len(data),nominal_mask='all 87 paired labels zero',limitations='Raw gap summaries are per year; no boundary-gap attribution. Anonymous metadata does not establish physical meaning. Global event IDs without local annotations may still have correlated evidence; not a recall ceiling. No test data, weights changed or training.'),indent=2))
    print('Complete',out,flush=True)
if __name__=='__main__':main()
