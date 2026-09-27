"""Descriptive 2000–2006 source review; never loads labels or model results."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    out=p.parse_args().output;out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[2];source=root/'data/ESA-Mission1'
    rows=[];monthly=[];provenance=[]
    for kind,folder,key in [('telemetry','channels','Channel'),('telecommand','telecommands','Telecommand')]:
        metadata=pd.read_csv(source/f'{folder}.csv')
        assert set(metadata[key])=={f.stem for f in (source/folder).glob('*.zip')}
        for n,meta in enumerate(metadata.to_dict('records')):
            feature=meta[key];path=source/folder/f'{feature}.zip';before=path.stat()
            x=pd.read_pickle(path)
            x.index=pd.to_datetime(x.index)
            x=x.loc[(x.index>='2000-01-01')&(x.index<'2007-01-01'),feature]
            for year in range(2000,2007):
                v=x.loc[(x.index>=f'{year}-01-01')&(x.index<f'{year+1}-01-01')]
                delta=np.diff(v.index.asi8)/1e9
                finite=np.isfinite(v.to_numpy());valid=v.to_numpy()[finite]
                row=dict(feature=feature,kind=kind,year=year,observations=len(v),
                    nonfinite=int((~finite).sum()),duplicates=int(v.index.duplicated().sum()),
                    backwards=int((delta<0).sum()),unique_values=int(v.nunique()),
                    active_months=int(v.index.to_period('M').nunique()),
                    priority=meta.get('Priority'),subsystem=meta.get('Subsystem'),
                    unit_code=meta.get('Physical Unit'),currently_prepared=kind=='telemetry' or meta.get('Priority',0)>=3)
                if kind=='telemetry':
                    row.update(min=float(valid.min()) if len(valid) else None,
                        max=float(valid.max()) if len(valid) else None,
                        mean=float(valid.mean(dtype=np.float64)) if len(valid) else None,
                        std=float(valid.std(dtype=np.float64)) if len(valid) else None,
                        changes=int((np.diff(v.to_numpy())!=0).sum()),
                        median_interval_seconds=float(np.median(delta[delta>0])) if (delta>0).any() else None,
                        max_interval_seconds=float(delta.max()) if len(delta) else None)
                rows.append(row)
            if kind=='telemetry':
                m=x.resample('MS').agg(['min','max','mean','count']).reindex(pd.date_range('2000-01-01','2006-12-01',freq='MS'))
                for month,r in m.iterrows():monthly.append(dict(feature=feature,month=str(month.date()),**r.to_dict()))
            after=path.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
            provenance.append(dict(feature=feature,bytes=before.st_size,mtime_ns=before.st_mtime_ns))
            if (n+1)%25==0:print(kind,n+1,flush=True)
    yearly=pd.DataFrame(rows);monthly=pd.DataFrame(monthly)
    assert len(yearly)==774*7 and not yearly.duplicated(['feature','year']).any()
    assert len(monthly)==76*84
    assert (monthly.groupby('feature')['count'].sum().sort_index().to_numpy()==yearly[yearly.kind=='telemetry'].groupby('feature').observations.sum().sort_index().to_numpy()).all()
    yearly.to_csv(out/'yearly_inputs.csv',index=False)
    monthly.to_csv(out/'monthly_telemetry.csv',index=False)
    pd.DataFrame(provenance).to_csv(out/'source_manifest.csv',index=False)
    summary=[]
    for year,d in yearly.groupby('year'):
        tele=d[d.kind=='telemetry'];cmd=d[d.kind=='telecommand']
        summary.append(dict(year=year,telemetry_channels=len(tele),variable_telemetry=int((tele.unique_values>1).sum()),
            telemetry_observations=int(tele.observations.sum()),active_command_types=int((cmd.observations>0).sum()),
            command_occurrences=int(cmd.observations.sum()),nonfinite=int(d.nonfinite.sum()),duplicates=int(d.duplicates.sum()),backwards=int(d.backwards.sum())))
    summary=pd.DataFrame(summary);summary.to_csv(out/'yearly_summary.csv',index=False)
    features=yearly.loc[yearly.kind=='telemetry','feature'].drop_duplicates().tolist()
    for page in range(10):
        fig,axes=plt.subplots(4,2,figsize=(13,10),constrained_layout=True)
        subset=features[page*8:(page+1)*8]
        for ax,feature in zip(axes.flat,subset):
            d=monthly[monthly.feature==feature];t=pd.to_datetime(d.month)
            ax.fill_between(t,d['min'],d['max'],alpha=.25,label='Monthly min–max')
            ax.plot(t,d['mean'],lw=1,label='Monthly mean')
            ax.axvspan(pd.Timestamp('2006-01-01'),pd.Timestamp('2007-01-01'),color='orange',alpha=.10)
            ax.set_title(feature);ax.grid(alpha=.15)
        for ax in list(axes.flat)[len(subset):]:ax.set_visible(False)
        axes.flat[0].legend(fontsize=8)
        fig.suptitle('2000–2006 raw telemetry • descriptive only, no labels\n2006 shaded: inspected descriptively, reserved from tuning; monthly envelopes can hide timing')
        fig.savefig(out/f'atlas_{page+1:02d}.png',dpi=130);plt.close(fig)
    (out/'protocol.json').write_text(json.dumps(dict(
        interval='[2000-01-01,2007-01-01)',scope='76 telemetry channels and 698 telecommand types; annual statistics',
        use='Descriptive exploration only, not channel selection or model tuning.',
        labels='No anomaly-label file or model results loaded.',
        validation='2006 covariates inspected; no longer completely unseen. Reserved from tuning; do not use these summaries to adapt a model and then claim independent 2006 validation.',
        limitations='Raw statistics include all periods without nominal filtering. Within-year intervals omit cross-year gaps. Size/mtime checks are not checksums. Anonymized units. No predictive utility inferred.',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
    print(summary.to_string(index=False))


if __name__=='__main__':main()
