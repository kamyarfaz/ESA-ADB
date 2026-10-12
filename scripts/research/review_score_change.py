"""Independently replay saved transformation with pandas EWM and verify metrics."""
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from esa_thesis.diagnose_channels import digest
from esa_thesis.masked_pilot import assessment_scopes
from esa_thesis.thresholds import postprocess
root=Path('results_longrun/development/score_change_v1');source=Path('results_longrun/development/inner_selection_cuda_v2')
out=Path('docs/research/tables/score-change-results-2026-10-08')
if out.exists():raise ValueError('Preserve review')
protocol=json.loads((root/'protocol.json').read_text());assert protocol['transform_sha256']==digest('esa_thesis/score_change.py')
frame=pd.read_csv(root/'results.csv');count=0
for fold in ('2001','2002'):
 for seed in (42,43,44):
  src=source/fold/f'seed{seed}';dst=root/fold/f'seed{seed}'
  frozen=json.loads((dst/'frozen_rule.json').read_text());epoch=frozen['epoch']
  cal=np.load(src/f'epoch_{epoch}'/'calibration_score.npy');val=np.load(src/'calibration_selected_validation_score.npy')
  logs=np.log1p(np.r_[cal,val].astype(np.float64))
  mean=pd.Series(logs).ewm(halflife=2880,adjust=False).mean().to_numpy()
  derived=np.abs(logs-np.r_[logs[0],mean[:-1]])
  np.testing.assert_allclose(derived[:len(cal)],np.load(dst/'calibration_score.npy'),rtol=0,atol=1e-10)
  np.testing.assert_allclose(derived[len(cal):],np.load(dst/'validation_score.npy'),rtol=0,atol=1e-10)
  for arm in ('original','score_change'):
   pred=np.load(dst/f'{arm}_prediction.npy')
   if arm=='score_change':
    r=frozen['rule'];np.testing.assert_array_equal(pred,postprocess(np.load(dst/'validation_score.npy')>r['threshold'],int(r['merge_gap']),int(r['min_dur'])))
   for scope,ev in assessment_scopes(np.load(src/'validation_timestamps.npy')).items():
    metrics=ev.score(pred);row=frame[(frame.fold==int(fold))&(frame.seed==seed)&(frame.arm==arm)&(frame.scope==scope)].iloc[0]
    for key,value in metrics.items():
     if isinstance(value,str):assert row[key]==value
     else:assert np.isclose(row[key],value,atol=1e-9,rtol=0)
    count+=1
out.mkdir(parents=True)
for f in ('results.csv','summary.csv','protocol.json','verification.json'):shutil.copy2(root/f,out/f)
(out/'independent_verification.json').write_text(json.dumps(dict(metric_rows=count,independent_transform='pandas EWM, halflife2880 adjustFalse, lagged; tolerance1e-10',status='complete'),indent=2)+'\n')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for ax,fold in zip(axes,(2001,2002)):
 for arm in ('original','score_change'):
  d=frame[(frame.fold==fold)&(frame.scope=='system_all_channels')&(frame.arm==arm)]
  ax.plot(d.seed,d['EW_F_0.50'],marker='o',label=arm)
 ax.set_ylim(-.02,1);ax.set_xticks([42,43,44]);ax.set_xlabel('Seed');ax.set_ylabel('System F0.5');ax.set_title(f'Real Jul–Dec {fold}');ax.legend()
fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)
print(frame[frame.scope=='system_all_channels'][['fold','seed','arm','calibration_f05','EW_F_0.50','TPe','FPe','FNe']].to_string(index=False))
