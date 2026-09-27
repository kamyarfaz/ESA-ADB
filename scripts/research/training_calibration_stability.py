"""Compare like-for-like nominal window errors; training/calibration only."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from esa_thesis.diagnose_channels import digest,aggregate
from esa_thesis import config
from esa_thesis.protocol import evaluator_for
from esa_thesis.thresholds import postprocess

out=Path('docs/research/tables/training-calibration-stability-2026-09-26')
out.mkdir(parents=True,exist_ok=False)
sources={2003:'baseline_2003_epoch25_gpu_v1',2004:'baseline_2004_epoch35_v2',2005:'baseline_2005_epoch5_v1'}
rows=[];tails=[];provenance={}
def summarize(fold,period,stage,errors):
 for j,name in enumerate([f'channel_{i}' for i in range(40,48)]+['max_across_channels']):
  v=errors[:,j] if j<8 else errors.max(axis=1)
  if len(v)==0:raise ValueError('Empty reference period')
  rows.append(dict(fold=fold,period=period,stage=stage,channel=name,windows=len(v),**{f'q{q}':float(np.percentile(v,q)) for q in [50,95,99,99.9,100]}))
for year,diag in sources.items():
 root=Path('results_longrun/diagnostics')/diag
 baseline=Path('results_longrun/development')/('ae_baseline_2003_seed42' if year==2003 else 'ae_baseline_remaining_seed42')/str(year)
 files=[root/'training_window_errors.npz',root/'calibration_window_errors.npz',root/'results.json',baseline/'frozen_rule.json',config.ANNOTATIONS_FILE,config.ANOMALY_TYPES_FILE]
 before={str(p):digest(p) for p in files}
 assert json.loads(files[2].read_text())['reproduction']=='passed within 1e-6'
 rule=json.loads(files[3].read_text())['rule']
 with np.load(files[0]) as a:train=a['errors'];train_times=a['timestamps']
 with np.load(files[1]) as a:errors=a['errors'];starts=a['starts'];times=a['timestamps'];nominal=a['nominal']
 assert np.isfinite(train).all() and np.isfinite(errors).all()
 assert np.array_equal(times,pd.date_range(f'{year}-01-01',f'{year}-04-01',freq='30s',inclusive='left').asi8)
 channels=aggregate(errors,starts,len(times));pred=postprocess(channels.max(axis=0)>rule['threshold'],int(rule['merge_gap']),int(rule['min_dur']))
 metrics=evaluator_for(times).score(pred)
 for key in ['EW_F_0.50','TPe','FPe','FNe','false_positive_seconds']:assert np.isclose(metrics[key],rule['val_'+key],rtol=0,atol=1e-6)
 # Exclude every window touching any non-nominal prepared label, including
 # outside-input annotations. Compare pre-aggregation MSE to pre-aggregation MSE.
 count=np.r_[0,np.cumsum(~nominal)];clean=(count[starts+config.SEQ_LEN]-count[starts])==0
 years=pd.DatetimeIndex(train_times).year
 for y in sorted(set(years)):summarize(year,str(y),'training_sample',train[years==y])
 summarize(year,'all_training','training_sample',train)
 dates=pd.DatetimeIndex(times);months=dates.month.to_numpy()
 for month in [1,2,3]:
  same_month=months[starts]==months[starts+config.SEQ_LEN-1]
  keep=clean&same_month&(months[starts]==month)
  e=errors[keep];summarize(year,f'{year}-{month:02d}','calibration_nominal',e)
  for j in range(8):
   reference=float(np.percentile(train[:,j],99.9));v=e[:,j]
   tails.append(dict(fold=year,month=month,channel=f'channel_{40+j}',windows=len(v),training_q999=reference,calibration_fraction_above_training_q999=float(np.mean(v>reference)),calibration_q999=float(np.percentile(v,99.9))))
 assert before=={str(p):digest(p) for p in files};provenance[str(year)]=before
pd.DataFrame(rows).to_csv(out/'nominal_window_quantiles.csv',index=False)
pd.DataFrame(tails).to_csv(out/'tail_exceedance.csv',index=False)
(out/'protocol.json').write_text(json.dumps(dict(status='complete',inputs=provenance,policy='Only saved training/calibration windows; no assessment access; no model/rule changes',comparison='Window reconstruction MSE on globally nominal whole windows, no aggregation mixing',limitations='Training is evenly spaced2000window/year sample, calibration includes all eligible stride32 windows. Overlapping windows are dependent. Within-checkpoint descriptive comparison only; models/scalers differ across folds. In-sample training optimism; sparse sampled tails; not drift causality or significance proof.'),indent=2)+'\n')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
frame=pd.DataFrame(rows);fig,axes=plt.subplots(3,1,figsize=(11,10),constrained_layout=True)
for ax,year in zip(axes,sources):
 sub=frame[(frame.fold==year)&(frame.channel=='max_across_channels')&(frame.period!='all_training')]
 for q in ['q50','q99','q99.9']:ax.plot(sub.period,sub[q],marker='o',label=q)
 ax.set_yscale('log');ax.set_ylabel('Window MSE (log scale)');ax.set_title(f'{year} checkpoint: nominal training sample and Jan–Mar calibration');ax.legend(ncol=3)
 ax.tick_params(axis='x',rotation=25)
fig.savefig(out/'nominal_score_stability.png',dpi=140);plt.close(fig)
print(frame[(frame.channel=='max_across_channels')][['fold','period','windows','q50','q99','q99.9','q100']].to_string(index=False))
print('Largest nominal calibration fractions exceeding sampled training channel q99.9:')
print(pd.DataFrame(tails).sort_values('calibration_fraction_above_training_q999',ascending=False).head(12).to_string(index=False))
