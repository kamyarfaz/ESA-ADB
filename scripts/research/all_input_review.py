"""Training-only inventory of every Mission 1 telemetry and telecommand source."""
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    out = parser.parse_args().output
    out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    source = root/'data/ESA-Mission1'
    months = pd.date_range('2000-01-01','2002-12-01',freq='MS')
    inventory, monthly = [], []
    annotations=pd.read_csv(root/'docs/research/tables/counter-review-2026-09-23/training_annotations.csv')
    for kind, metafile, name in [('telemetry','channels.csv','Channel'),('telecommand','telecommands.csv','Telecommand')]:
        meta=pd.read_csv(source/metafile)
        folder=source/('channels' if kind=='telemetry' else 'telecommands')
        assert set(meta[name])=={p.stem for p in folder.glob('*.zip')}
        for index, record in enumerate(meta.to_dict('records')):
            feature=record[name]; path=folder/f'{feature}.zip'; stat=path.stat()
            frame=pd.read_pickle(path)
            frame.index=pd.to_datetime(frame.index)
            frame=frame.loc[(frame.index>='2000-01-01')&(frame.index<'2003-01-01')]
            assert feature in frame
            v=frame[feature]; finite=np.isfinite(v.to_numpy())
            ordered=v.sort_index()
            change_count=int((np.diff(ordered.to_numpy())!=0).sum()) if len(v)>1 else 0
            relevant=annotations[annotations.Channel==feature]
            row=dict(feature=feature,kind=kind,training_rows=len(v),
                duplicate_timestamps=int(v.index.duplicated().sum()),
                backwards_steps=int((np.diff(v.index.asi8)<0).sum()),
                nonfinite=int((~finite).sum()),raw_unique=int(v.nunique()),
                raw_changes=change_count if kind=='telemetry' else None,
                first=str(v.index.min()) if len(v) else '',last=str(v.index.max()) if len(v) else '',
                priority=record.get('Priority'),subsystem=record.get('Subsystem'),
                unit_code=record.get('Physical Unit'),target=record.get('Target'),
                currently_prepared=kind=='telemetry' or record.get('Priority',0)>=3,
                training_event_ids=int(relevant.loc[relevant.Category.isin(['Anomaly','Rare Event']),'ID'].nunique()),
                source_bytes=stat.st_size,source_mtime_ns=stat.st_mtime_ns)
            counts=ordered.resample('MS').count().reindex(months,fill_value=0)
            row['active_months']=int((counts>0).sum())
            # Event streams contain occurrences, not continuously observed measurements.
            if kind=='telecommand':
                row['review']='no occurrences in first-fold training' if not len(v) else 'event context candidate; constant payload is not uninformative'
                for m,n in counts.items():monthly.append(dict(feature=feature,kind=kind,month=str(m.date()),count=int(n)))
            else:
                row['review']='raw constant: investigate before input exclusion' if v.nunique()<=1 else 'variable telemetry; review quality and complementarity'
                stats=ordered.resample('MS').agg(['min','max','mean','std','count']).reindex(months)
                for m,r in stats.iterrows(): monthly.append(dict(feature=feature,kind=kind,month=str(m.date()),**r.to_dict()))
            after=path.stat()
            assert (stat.st_size,stat.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
            inventory.append(row)
            if (index+1)%25==0:print(kind,index+1,'reviewed',flush=True)
    inv=pd.DataFrame(inventory); month=pd.DataFrame(monthly)
    assert len(inv)==774 and inv.feature.is_unique
    assert len(month)==774*36
    inv.to_csv(out/'all_inputs.csv',index=False);month.to_csv(out/'monthly_raw_summary.csv',index=False)
    profile=pd.read_csv(root/'docs/research/tables/channel-study-2026-09-22/channel_profile.csv')
    review=inv.merge(profile[['feature','constant_training_nominal','sample_unique','sample_mode_fraction','unchanged_30s_fraction']],on='feature',how='left',validate='one_to_one')
    review.loc[review.constant_training_nominal.fillna(False),'review']='constant prepared nominal: investigate sampling/filtering; retain source'
    review.to_csv(out/'input_review.csv',index=False)
    # Atlas covers every telemetry channel; envelopes use every native training value.
    telemetry=inv.loc[inv.kind=='telemetry','feature'].tolist()
    for page in range(10):
        fig,axes=plt.subplots(4,2,figsize=(13,10),constrained_layout=True)
        for ax,feature in zip(axes.flat,telemetry[page*8:(page+1)*8]):
            d=month[month.feature==feature];t=pd.to_datetime(d.month)
            ax.fill_between(t,d['min'],d['max'],alpha=.25,label='Monthly min–max')
            ax.plot(t,d['mean'],linewidth=1,label='Monthly mean')
            ax.set_title(feature);ax.grid(alpha=.15);ax.tick_params(axis='x',labelrotation=20)
        for ax in list(axes.flat)[len(telemetry[page*8:(page+1)*8]):]:ax.set_visible(False)
        axes.flat[0].legend(fontsize=8)
        fig.suptitle('Raw telemetry • all native samples summarized monthly • 2000–2002\nAnonymized values; includes annotated periods; envelopes do not show short-event timing')
        fig.savefig(out/f'telemetry_atlas_{page+1:02d}.png',dpi=130);plt.close(fig)
    tc=month[month.kind=='telecommand'].pivot(index='feature',columns='month',values='count')
    tc=tc.reindex(inv.loc[inv.kind=='telecommand','feature'])
    fig,ax=plt.subplots(figsize=(12,9),constrained_layout=True)
    im=ax.imshow(np.log1p(tc.to_numpy()),aspect='auto',interpolation='nearest',cmap='viridis')
    ax.set_xticks(range(0,36,3),tc.columns[::3],rotation=45,ha='right')
    ax.set_yticks(range(0,698,50),tc.index[::50]);ax.set_title('All 698 telecommand types • monthly training occurrence counts')
    fig.colorbar(im,ax=ax,label='log(1 + number of recorded occurrences)')
    fig.savefig(out/'telecommand_activity.png',dpi=150);plt.close(fig)
    protocol=dict(interval='[2000-01-01,2003-01-01)',telemetry=76,telecommands=698,
        physical_meanings='Anonymized codes; no engineering semantics inferred.',
        review='No source deletion or automatic model-input exclusion. No assessment statistics or training.',
        limitations='Monthly raw envelopes include annotations and do not quantify nominal usefulness. Telecommand payload constancy does not mean inactivity. Correlations remain earlier hourly nominal exploratory estimates. No new preprocessing or predictive utility validated.',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(inv.groupby('kind').training_rows.agg(['count','sum']).to_string())


if __name__=='__main__':main()
