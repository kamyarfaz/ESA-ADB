"""Descriptive operating-range and clipping checks; no detector tuning."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from esa_thesis import config

CHANNELS=['channel_14','channel_21','channel_29']


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    root=config.TRAIN_FILE.parents[4];run=root/'results_longrun/development/ae_expanded_seed42/2005'
    checkpoint=run/'model_checkpoint.pt';saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    positions=[saved['features'].index(c) for c in CHANNELS];med=np.asarray(saved['scaler_median'])[positions];iqr=np.asarray(saved['scaler_iqr'])[positions]
    labels=[c for c in pd.read_csv(config.TRAIN_FILE,nrows=0).columns if c.startswith('is_anomaly_')]
    xs=[];ys=[];ts=[]
    print('Reading channels 14, 21, 29 and nominal labels through 2005...',flush=True)
    with pd.read_csv(config.TRAIN_FILE,usecols=['timestamp',*CHANNELS,*labels],chunksize=100000) as reader:
        for f in reader:
            t=pd.to_datetime(f.timestamp,utc=True).astype('int64').to_numpy();keep=t<pd.Timestamp('2006-01-01',tz='UTC').value
            x=f.loc[keep,CHANNELS].to_numpy(np.float32);y=f.loc[keep,labels].to_numpy()
            if not np.isfinite(x).all() or not np.isin(y,[0,1,2,3]).all():raise ValueError('Invalid input')
            xs.append(x);ys.append((y>0).any(axis=1));ts.append(t[keep])
            if not keep.all():break
    x=np.concatenate(xs);non_nominal=np.concatenate(ys);times=np.concatenate(ts)
    assert np.all(np.diff(times)==30_000_000_000)
    del xs,ys,ts
    z=(x-med)/iqr
    cuts=[0,np.searchsorted(times,pd.Timestamp('2005-01-01',tz='UTC').value),np.searchsorted(times,pd.Timestamp('2005-04-01',tz='UTC').value),len(times)]
    period_rows=[];step_reference={}
    for name,lo,hi in zip(['training','calibration','assessment'],cuts[:-1],cuts[1:]):
        for j,c in enumerate(CHANNELS):
            for scope,keep in [('all',np.ones(hi-lo,dtype=bool)),('mission_nominal',~non_nominal[lo:hi])]:
                v=x[lo:hi,j][keep];zz=z[lo:hi,j][keep]
                row={'period':name,'channel':c,'scope':scope,'samples':len(v), 'raw_min':float(v.min()),'raw_max':float(v.max()),
                     'raw_p01':float(np.percentile(v,1)),'raw_median':float(np.median(v)),'raw_p99':float(np.percentile(v,99)),
                     'scaled_median':float(np.median(zz)),'scaled_p99':float(np.percentile(zz,99)), 'clip_fraction':float((np.abs(zz)>config.CLIP_Z).mean())}
                diff=np.abs(np.diff(z[lo:hi,j]));both=keep[:-1]&keep[1:];d=diff[both]
                row.update(step_p99=float(np.percentile(d,99)),step_p999=float(np.percentile(d,99.9)),step_max=float(d.max()))
                period_rows.append(row)
                if name=='training' and scope=='mission_nominal':step_reference[c]=row['step_p999']
    pd.DataFrame(period_rows).to_csv(args.output/'period_summary.csv',index=False)
    monthly=[]
    for a,b in zip(pd.date_range('2000-01-01','2006-01-01',freq='MS',tz='UTC')[:-1],pd.date_range('2000-01-01','2006-01-01',freq='MS',tz='UTC')[1:]):
        lo,hi=np.searchsorted(times,[a.value,b.value]);keep=~non_nominal[lo:hi]
        for j,c in enumerate(CHANNELS):
            v=z[lo:hi,j][keep]
            monthly.append({'month':str(a),'channel':c,'samples':len(v),'median':float(np.median(v)) if len(v) else np.nan,
                            'p01':float(np.percentile(v,1)) if len(v) else np.nan,'p99':float(np.percentile(v,99)) if len(v) else np.nan})
    pd.DataFrame(monthly).to_csv(args.output/'monthly_nominal.csv',index=False)
    attribution=pd.read_csv(root/'results_longrun/development/channel_diagnostics_2026-09-21/alarm_channel_errors.csv')
    selected=attribution[(attribution.fold==2005)&(attribution.variant=='expanded')&attribution.dominant]
    rows=[]
    for alarm in selected.itertuples():
        start=pd.Timestamp(alarm.start).value;end=start+int(alarm.seconds*1e9);lo,hi=np.searchsorted(times,[start,end])
        # AE error can be spread up to 255 samples away from a contributing sample.
        left=max(cuts[2],lo-config.SEQ_LEN+1);right=min(len(times),hi+config.SEQ_LEN-1)
        for j,c in enumerate(CHANNELS):
            v=z[left:right,j];d=np.abs(np.diff(v));jump=int(np.argmax(d))
            rows.append({'start':alarm.start,'kind':alarm.kind,'dominant_channel':alarm.channel,'channel':c,'peak_score':alarm.peak_score,
                         'seconds':alarm.seconds,'padded_scaled_min':float(v.min()),'padded_scaled_max':float(v.max()),
                         'padded_clip_fraction':float((np.abs(v)>config.CLIP_Z).mean()),'max_step':float(d[jump]),
                         'max_step_time':str(pd.Timestamp(times[left+jump+1],tz='UTC')),
                         'steps_above_training_p999':int((d>step_reference[c]).sum())})
    pd.DataFrame(rows).to_csv(args.output/'alarm_contexts.csv',index=False)
    manifest={'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'channels':CHANNELS,'scaler_median':med.tolist(),'scaler_iqr':iqr.tolist(),
              'source_csv':str(config.TRAIN_FILE),'csv_size':config.TRAIN_FILE.stat().st_size,'csv_mtime_ns':config.TRAIN_FILE.stat().st_mtime_ns,
              'nominal_definition':'all supplied annotation columns equal zero','alarm_context_padding_samples':config.SEQ_LEN-1,
              'limitation':'descriptive development analysis; padding includes future observations for retrospective AE context; not causal detection or proof of physical operating modes'}
    (args.output/'protocol.json').write_text(json.dumps(manifest,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(11,8),sharex=True)
    frame=pd.DataFrame(monthly)
    for ax,c in zip(axes,CHANNELS):
        f=frame[frame.channel==c];dates=pd.to_datetime(f.month)
        ax.fill_between(dates,f.p01,f.p99,color='#547ea5',alpha=.25,label='Nominal 1st–99th percentile')
        ax.plot(dates,f['median'],color='#245b88',label='Nominal median')
        ax.axvline(pd.Timestamp('2005-01-01',tz='UTC'),color='black',linestyle='--')
        ax.axvline(pd.Timestamp('2005-04-01',tz='UTC'),color='#a34a28',linestyle='--')
        ax.set_ylabel(c+'\nscaled value');ax.grid(alpha=.2)
    axes[0].legend(loc='upper left');axes[0].set_title('Monthly nominal telemetry using the fixed 2005 training scaler\nBlack: calibration start; orange: assessment start')
    fig.tight_layout();fig.savefig(args.output/'nominal_ranges.png',dpi=150);plt.close(fig)
    print(pd.DataFrame(period_rows).to_string(index=False),flush=True)


if __name__=='__main__':main()
