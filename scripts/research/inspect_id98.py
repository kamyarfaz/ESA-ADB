"""Inspect exact local annotation intervals, not the multi-day system envelope."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from esa_thesis import config
from esa_thesis.diagnose_channels import digest
from esa_thesis.evaluation.esa import _merge
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path('docs/research/tables/id98-local-review-2026-09-26');out.mkdir(parents=True,exist_ok=False)
root=Path('results_longrun/diagnostics/baseline_2005_epoch5_v1')
files=[config.ANNOTATIONS_FILE,config.ANOMALY_TYPES_FILE,root/'calibration_inputs.npz',root/'calibration_window_errors.npz',Path('esa_thesis/augmentation.py')]
before={str(p):digest(p) for p in files}
labels=pd.read_csv(files[0]);labels=labels[labels.ID=='id_98'].copy()
for col in ['StartTime','EndTime']:labels[col]=pd.to_datetime(labels[col],utc=True)
local=labels[labels.Channel.isin([f'channel_{i}' for i in range(41,47)])]
local.to_csv(out/'local_annotations.csv',index=False)
intervals=_merge([(a.value,b.value) for a,b in zip(local.StartTime,local.EndTime)])
with np.load(files[2]) as a:t=a['timestamps'];x=a['scaled']
with np.load(files[3]) as a:nominal=a['nominal']
rows=[]
for index,(lo,hi) in enumerate(intervals):
 event=(t>=lo)&(t<=hi)
 before_mask=(t>=lo-30*60*10**9)&(t<lo)&nominal
 after_mask=(t>hi)&(t<=hi+30*60*10**9)&nominal
 for j in range(1,7):
  v=x[event,j];ref=x[before_mask|after_mask,j]
  rows.append(dict(interval=index,start=str(pd.Timestamp(lo)),end=str(pd.Timestamp(hi)),duration_seconds=(hi-lo)/1e9,grid_samples=int(event.sum()),channel=f'channel_{40+j}',event_min=float(v.min()),event_max=float(v.max()),event_median=float(np.median(v)),nominal_reference_median=float(np.median(ref)),nominal_reference_min=float(ref.min()),nominal_reference_max=float(ref.max())))
 fig,axes=plt.subplots(6,1,figsize=(11,11),sharex=True,constrained_layout=True)
 keep=(t>=lo-15*60*10**9)&(t<=hi+15*60*10**9)
 for ax,j in zip(axes,range(1,7)):
  ax.plot(pd.to_datetime(t[keep]),x[keep,j],lw=1)
  for _,r in local[local.Channel==f'channel_{40+j}'].iterrows():
   if r.EndTime.value>=t[keep][0] and r.StartTime.value<=t[keep][-1]:ax.axvspan(r.StartTime.tz_localize(None),r.EndTime.tz_localize(None),color='orange',alpha=.3)
  ax.set_ylabel(f'ch{40+j}')
 axes[0].set_title(f'id_98 local interval {index+1}; orange = channel-specific annotation\nTraining-scaled telemetry; 30-second grid')
 fig.savefig(out/f'interval_{index+1}.png',dpi=120);plt.close(fig)
pd.DataFrame(rows).to_csv(out/'local_interval_statistics.csv',index=False)
# Correlation is only descriptive on fixed90minute context; a one-sample
# annotation cannot support an event-only correlation estimate.
lo=intervals[0,0]-15*60*10**9;hi=intervals[-1,1]+15*60*10**9
context=(t>=lo)&(t<=hi)
for name,mask in [('event_context',context),('previous_day_nominal',(t>=lo-86400*10**9)&(t<=hi-86400*10**9)&nominal)]:
 pd.DataFrame(x[mask,1:7],columns=[f'channel_{i}' for i in range(41,47)]).corr().to_csv(out/f'{name}_correlation.csv')
assert before=={str(p):digest(p) for p in files}
summary=dict(local_union_seconds=int(sum(hi-lo for lo,hi in intervals)/1e9),local_intervals=[dict(start=str(pd.Timestamp(lo)),end=str(pd.Timestamp(hi)),seconds=int((hi-lo)/1e9),grid_samples=int(((t>=lo)&(t<=hi)).sum())) for lo,hi in intervals],synthetic_duration_seconds=[max(4,int(config.PA_SUBWIN[0]*config.SEQ_LEN))*30,max(5,int(config.PA_SUBWIN[1]*config.SEQ_LEN))*30],synthetic_channel_count=[1,4],inputs=before,limitations='One previously inspected calibration event; point samples and adjacent references, not causal inference. No training/assessment access. Short local intervals do not support reliable event-only correlations.')
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k!='inputs'},indent=2))
