"""Prepared telemetry and frozen score plots; no model fitting or score changes."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from esa_thesis import config
from esa_thesis.development import load_period
from esa_thesis.diagnose_channels import FEATURES,TRAIN_SHA,digest
from esa_thesis.masked_pilot import assessment_scopes
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out=Path('docs/research/tables/inner-event-inputs-2026-10-08-v2')
if out.exists():raise ValueError('Use new output')
print('Checking CSV',flush=True);assert digest(config.TRAIN_FILE)==TRAIN_SHA
print('Loading 2001–2002 prepared telemetry',flush=True)
x,y,t=load_period(config.TRAIN_FILE,FEATURES,'2001-01-01','2003-01-01')
labels=pd.read_csv(config.ANNOTATIONS_FILE)
for col in ('StartTime','EndTime'):labels[col]=pd.to_datetime(labels[col],utc=True).astype('int64')
out.mkdir(parents=True);stats=[]
for fold in ('2001','2002'):
 root=Path('results_longrun/development/inner_selection_cuda_v2');folder=root/fold/'seed42'
 d=json.loads((folder/'frozen_decisions.json').read_text())['calibration_selected']
 saved=torch.load(folder/f"epoch_{d['epoch']}"/'model.pt',map_location='cpu',weights_only=False)
 z=(x-saved['scaler_median'])/saved['scaler_iqr']
 for period in ('calibration','validation'):
  times=np.load(root/fold/'calibration_timestamps.npy' if period=='calibration' else folder/'validation_timestamps.npy')
  ev=assessment_scopes(times)['input_channels_40_47']
  traces=[]
  for arm,name in [('joint','inner_selection_cuda_v2'),('isolated','inner_isolation_cuda_v1')]:
   f=Path('results_longrun/development')/name/fold/'seed42'
   dec=json.loads((f/'frozen_decisions.json').read_text())['calibration_selected']
   score=np.load(f/f"epoch_{dec['epoch']}"/'calibration_score.npy' if period=='calibration' else f/'calibration_selected_validation_score.npy')
   traces.append((arm,score/dec['rule']['threshold']))
  for aid,intervals in ev.by_id.items():
   lo=max(times[0],intervals[0,0]-256*30_000_000_000);hi=min(times[-1],intervals[-1,1]+256*30_000_000_000)
   keep=(t>=lo)&(t<=hi);ix=np.flatnonzero(keep)
   inside=np.zeros(len(t),bool)
   for a,b in intervals:inside|=(t>=a)&(t<=b)
   annot=labels[(labels.ID==aid)&(labels.StartTime<=times[-1])&(labels.EndTime>=times[0])]
   # Min/max envelope per contiguous bin preserves short peaks in long events.
   bins=np.array_split(ix,min(800,len(ix)));bt=pd.to_datetime([t[b[len(b)//2]] for b in bins],utc=True)
   fig,axes=plt.subplots(5,2,figsize=(13,11),constrained_layout=True)
   for j,ch in enumerate(FEATURES):
    ax=axes.flat[j];low=[z[b,j].min() for b in bins];high=[z[b,j].max() for b in bins]
    ax.fill_between(bt,low,high,alpha=.45);ax.plot(bt,[(a+b)/2 for a,b in zip(low,high)],lw=.5)
    for _,r in annot[annot.Channel==ch].iterrows():ax.axvspan(pd.Timestamp(max(lo,r.StartTime),tz='UTC'),pd.Timestamp(min(hi,r.EndTime),tz='UTC'),color='red',alpha=.15)
    ax.set_title(ch);ax.set_ylabel('Scaled, before clipping');ax.tick_params(axis='x',rotation=20,labelsize=6)
    channel_inside=np.zeros(len(t),bool)
    for _,r in annot[annot.Channel==ch].iterrows():channel_inside |= (t>=max(times[0],r.StartTime))&(t<=min(times[-1],r.EndTime))
    vals=z[channel_inside if channel_inside.any() else inside,j]
    if len(vals):stats.append(dict(fold=fold,period=period,event=aid,channel=ch,annotated=ch in set(annot.Channel),minimum=float(vals.min()),maximum=float(vals.max()),clipped_fraction=float(np.mean(np.abs(vals)>config.CLIP_Z))))
   k=np.flatnonzero((times>=lo)&(times<=hi));sb=np.array_split(k,min(800,len(k)));st=pd.to_datetime([times[b[len(b)//2]] for b in sb],utc=True)
   for ax,(arm,score) in zip(axes.flat[8:],traces):
    ax.fill_between(st,[score[b].min() for b in sb],[score[b].max() for b in sb],alpha=.4)
    ax.plot(st,[score[b].max() for b in sb],lw=1,color='tab:blue')
    ax.axhline(1,color='black',ls='--');ax.set_title(arm+' frozen score / threshold');ax.tick_params(axis='x',rotation=20,labelsize=6)
    for a,b in intervals:ax.axvspan(pd.Timestamp(a,tz='UTC'),pd.Timestamp(b,tz='UTC'),color='red',alpha=.15)
   fig.suptitle(f'{fold} {period} {aid}; seed42; prepared inputs, min/max envelopes; red=annotation')
   fig.savefig(out/f'{fold}_{period}_{aid}.png',dpi=110);plt.close(fig)
   print(fold,period,aid,flush=True)
pd.DataFrame(stats).to_csv(out/'event_input_ranges.csv',index=False)
assert digest(config.TRAIN_FILE)==TRAIN_SHA
(out/'verification.json').write_text(json.dumps(dict(status='complete',train_sha256=TRAIN_SHA,seed=42,limits='Prepared telemetry, not native raw; min/max bin envelopes; scalar score cannot distinguish averaging from model error; event ID intervals may span multiple channels'),indent=2)+'\n')
