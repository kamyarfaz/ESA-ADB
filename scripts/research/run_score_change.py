"""Predeclared fixed-checkpoint score transform; real-data paired comparison."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from esa_thesis.score_change import transform, HALF_LIFE_SAMPLES
from esa_thesis.development import QUANTILES
from esa_thesis.diagnose_channels import digest
from esa_thesis.masked_pilot import assessment_scopes
from esa_thesis.thresholds import sweep_thresholds,select_rows,postprocess

root=Path('results_longrun/development/inner_selection_cuda_v2')
out=Path('results_longrun/development/score_change_v1')
if out.exists():raise ValueError('Preserve existing results')
plan=json.loads((root/'protocol.json').read_text())
assert all(digest(f)==sha for f,sha in plan['annotations'].items())
assert all(digest(Path('esa_thesis')/f)==sha for f,sha in plan['source_sha256'].items())
out.mkdir(parents=True)
protocol=dict(source=str(root),source_protocol_sha256=digest(root/'protocol.json'),half_life=HALF_LIFE_SAMPLES,
 document_sha256=digest('docs/research/score-change-plan-2026-10-08.md'),transform_sha256=digest('esa_thesis/score_change.py'),runner_sha256=digest(__file__),quantiles=QUANTILES)
(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
rows=[];checks=[]
for fold in ('2001','2002'):
 for seed in (42,43,44):
  folder=root/fold/f'seed{seed}';dest=out/fold/f'seed{seed}';dest.mkdir(parents=True)
  manifest=json.loads((folder/'complete.json').read_text());assert all(digest(folder/f)==h for f,h in manifest.items())
  d=json.loads((folder/'frozen_decisions.json').read_text())['calibration_selected']
  times=np.load(root/fold/'calibration_timestamps.npy')
  original=np.load(folder/f"epoch_{d['epoch']}"/'calibration_score.npy')
  values,state=transform(original)
  grid=sweep_thresholds(values,evaluator=assessment_scopes(times)['system_all_channels'],threshold_values=np.percentile(values,QUANTILES),merge_gaps=[0,16,64],min_durs=[1,8,32])
  rule=select_rows(grid).iloc[0].to_dict();grid.to_csv(dest/'calibration_candidates.csv',index=False)
  (dest/'frozen_rule.json').write_text(json.dumps(dict(rule=rule,state=state,epoch=d['epoch']),indent=2)+'\n')
  np.save(dest/'calibration_score.npy',values)
  # Validation loaded only after the rule/state is persisted.
  vt=np.load(folder/'validation_timestamps.npy');v=np.load(folder/'calibration_selected_validation_score.npy')
  transformed,final=transform(v,state);np.save(dest/'validation_score.npy',transformed)
  predictions={}
  for arm,scores,r in [('original',v,d['rule']),('score_change',transformed,rule)]:
   pred=postprocess(scores>r['threshold'],int(r['merge_gap']),int(r['min_dur']));predictions[arm]=pred
   if arm=='original':np.testing.assert_array_equal(pred,np.load(folder/'calibration_selected_validation_prediction.npy'))
   np.save(dest/f'{arm}_prediction.npy',pred)
   for scope,ev in assessment_scopes(vt).items():
    rows.append(dict(fold=fold,seed=seed,arm=arm,scope=scope,epoch=d['epoch'],calibration_f05=r['val_esa_f05'],**ev.score(pred)))
  # Verify persisted candidate outputs and streaming equivalence.
  combined,_=transform(np.r_[original,v]);np.testing.assert_allclose(combined[len(original):],transformed,rtol=0,atol=1e-12)
  np.testing.assert_array_equal(np.load(dest/'score_change_prediction.npy'),predictions['score_change'])
  checks.append(dict(fold=fold,seed=seed,source_manifest_sha256=digest(folder/'complete.json'),calibration_state=state,final_state=final))
  print(fold,seed,'complete',flush=True)
frame=pd.DataFrame(rows);frame.to_csv(out/'results.csv',index=False)
summary=frame.groupby(['fold','arm','scope']).agg(f05_mean=('EW_F_0.50','mean'),f05_sd=('EW_F_0.50','std'),fp_mean=('FPe','mean'),recall_mean=('EW_recall','mean'),nominal_false_alarm_rate_mean=('nominal_false_alarm_rate','mean')).reset_index()
summary.to_csv(out/'summary.csv',index=False)
s=summary[summary.scope=='system_all_channels'].set_index(['fold','arm'])
passed=all(s.loc[(f,'score_change'),'f05_mean']>s.loc[(f,'original'),'f05_mean'] and s.loc[(f,'score_change'),'fp_mean']<=s.loc[(f,'original'),'fp_mean'] for f in ('2001','2002'))
assert all(digest(f)==sha for f,sha in plan['annotations'].items())
(out/'verification.json').write_text(json.dumps(dict(status='complete',runs=checks,advancement_gate_passed=bool(passed),promoted=False),indent=2)+'\n')
print(summary.to_string(index=False),flush=True);print('Gate',passed,flush=True)
