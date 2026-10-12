"""Compare frozen calibration and validation event scores without retuning."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from esa_thesis.diagnose_channels import digest
from esa_thesis.masked_pilot import assessment_scopes
from esa_thesis.thresholds import postprocess

out=Path('docs/research/tables/inner-event-audit-2026-10-08')
if out.exists():raise ValueError('Preserve existing audit')
rows=[];periods=[]
for arm,name in [('joint','inner_selection_cuda_v2'),('isolated','inner_isolation_cuda_v1')]:
 root=Path('results_longrun/development')/name
 for fold in ('2001','2002'):
  for seed in (42,43,44):
   folder=root/fold/f'seed{seed}'
   manifest=json.loads((folder/'complete.json').read_text())
   assert all(digest(folder/f)==h for f,h in manifest.items())
   d=json.loads((folder/'frozen_decisions.json').read_text())['calibration_selected'];rule=d['rule']
   for period in ('calibration','validation'):
    times=np.load(root/fold/'calibration_timestamps.npy' if period=='calibration' else folder/'validation_timestamps.npy')
    scores=np.load(folder/f"epoch_{d['epoch']}"/'calibration_score.npy' if period=='calibration' else folder/'calibration_selected_validation_score.npy')
    pred=postprocess(scores>rule['threshold'],int(rule['merge_gap']),int(rule['min_dur']))
    scopes=assessment_scopes(times);ev=scopes['system_all_channels'];local=scopes['input_channels_40_47']
    metrics=ev.score(pred)
    if period=='calibration':assert abs(metrics['EW_F_0.50']-rule['val_esa_f05'])<1e-9
    else:np.testing.assert_array_equal(pred,np.load(folder/'calibration_selected_validation_prediction.npy'))
    nominal=np.ones(len(times),bool)
    for lo,hi in ev.all_intervals:nominal &= ~((times>=lo)&(times<=hi))
    normal=np.sort(scores[nominal])
    periods.append(dict(arm=arm,fold=fold,seed=seed,period=period,threshold=rule['threshold'],nominal_q99_ratio=float(np.percentile(normal,99)/rule['threshold']),**metrics))
    starts,ends,closed=ev.prediction_intervals(pred)
    for aid,intervals in ev.by_id.items():
     keep=np.zeros(len(times),bool);detected=False
     for lo,hi in intervals:
      keep|=(times>=lo)&(times<=hi)
      detected|=bool(ev._overlaps(starts,ends,closed,lo,hi).any())
     vals=scores[keep];peak=float(vals.max()) if len(vals) else np.nan
     rows.append(dict(arm=arm,fold=fold,seed=seed,period=period,event_id=aid,local=aid in local.by_id,detected=detected,
       annotated_seconds=sum((hi-lo)/1e9 for lo,hi in intervals),sample_count=int(keep.sum()),peak_ratio=peak/rule['threshold'],
       nominal_fraction_at_least_event_peak=float((len(normal)-np.searchsorted(normal,peak,side='left'))/len(normal)) if len(vals) else np.nan))
out.mkdir(parents=True);df=pd.DataFrame(rows);df.to_csv(out/'events.csv',index=False);pd.DataFrame(periods).to_csv(out/'periods.csv',index=False)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
for i,arm in enumerate(('joint','isolated')):
 for j,fold in enumerate(('2001','2002')):
  ax=axes[i,j]
  for k,period in enumerate(('calibration','validation')):
   d=df[(df.arm==arm)&(df.fold==fold)&(df.period==period)&df.local]
   for seed,offset,color in [(42,-.15,'tab:blue'),(43,0,'tab:orange'),(44,.15,'tab:green')]:
    vals=d[d.seed==seed].peak_ratio
    ax.scatter(np.full(len(vals),k+offset),vals,label=f'seed {seed}' if k==0 else None,alpha=.65,color=color)
  ax.axhline(1,color='black',ls='--');ax.set_yscale('log');ax.set_xticks([0,1],['Calibration','Validation']);ax.set_ylabel('Local event peak / frozen threshold');ax.set_title(f'{arm}, {fold}');ax.legend(fontsize=7)
fig.savefig(out/'event_peaks.png',dpi=150);plt.close(fig)
(out/'verification.json').write_text(json.dumps(dict(verified_runs=12,periods=24,policy='Frozen primary checkpoints and rules; no threshold search',limits='Aggregate timestamp scores; nominal samples can include window spillover; peak exceedance fraction is diagnostic, not F0.5 or event FP rate'),indent=2)+'\n')
print(df[df.local].groupby(['arm','fold','period']).agg(events=('event_id','size'),hits=('detected','sum'),min_peak_ratio=('peak_ratio','min'),max_peak_ratio=('peak_ratio','max'),median_nominal_exceedance=('nominal_fraction_at_least_event_peak','median')).to_string())
