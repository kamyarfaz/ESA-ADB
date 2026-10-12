"""Verify real-event metrics and export the deterministic reference comparison."""
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from esa_thesis.masked_pilot import assessment_scopes
from esa_thesis.thresholds import postprocess,select_rows
root=Path('results_longrun/development/telemetry_reference_v1');out=Path('docs/research/tables/telemetry-reference-results-2026-10-08')
assert json.loads((root/'verification.json').read_text())['status']=='complete'
if out.exists():raise ValueError('Preserve review')
frame=pd.read_csv(root/'results.csv');verified=0
for fold in (2001,2002):
 f=root/str(fold);r=json.loads((f/'frozen_rule.json').read_text())['rule']
 selected=select_rows(pd.read_csv(f/'calibration_candidates.csv')).iloc[0]
 for key in ('threshold','merge_gap','min_dur','val_esa_f05'):assert np.isclose(r[key],selected[key],rtol=0,atol=1e-12)
 cal=np.load(f/'calibration_score.npy');cp=postprocess(cal>r['threshold'],int(r['merge_gap']),int(r['min_dur']))
 assert abs(assessment_scopes(np.load(f/'calibration_timestamps.npy'))['system_all_channels'].score(cp)['EW_F_0.50']-r['val_esa_f05'])<1e-9
 pred=np.load(f/'validation_prediction.npy');np.testing.assert_array_equal(pred,postprocess(np.load(f/'validation_score.npy')>r['threshold'],int(r['merge_gap']),int(r['min_dur'])))
 for scope,ev in assessment_scopes(np.load(f/'validation_timestamps.npy')).items():
  row=frame[(frame.fold==fold)&(frame.arm=='telemetry_reference')&(frame.scope==scope)].iloc[0]
  for key,value in ev.score(pred).items():
   if isinstance(value,str):assert row[key]==value
   else:assert np.isclose(row[key],value,atol=1e-9,rtol=0)
  verified+=1
out.mkdir(parents=True)
for name in ('summary.csv','results.csv','events.csv','verification.json','protocol.json'):shutil.copy2(root/name,out/name)
(out/'metric_verification.json').write_text(json.dumps(dict(verified_reference_rows=verified,calibration_rules_verified=2),indent=2)+'\n')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
s=pd.read_csv(root/'summary.csv');fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for ax,fold in zip(axes,(2001,2002)):
 d=frame[(frame.fold==fold)&(frame.scope=='system_all_channels')]
 labels=['Joint '+str(v) for v in d[d.arm=='joint'].seed]+['Direct telemetry']
 values=list(d[d.arm=='joint']['EW_F_0.50'])+list(d[d.arm=='telemetry_reference']['EW_F_0.50'])
 ax.bar(labels,values);ax.set_ylim(0,1);ax.set_ylabel('System F0.5');ax.set_title(f'Real Jul–Dec {fold}');ax.tick_params(axis='x',rotation=20)
fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)
print(frame[frame.arm=='telemetry_reference'][['fold','scope','calibration_f05','EW_F_0.50','TPe','FPe','FNe','nominal_false_alarm_rate']].to_string(index=False))
