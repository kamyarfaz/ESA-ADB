"""Both saved models, complete2005calibration, own frozen rules; no tuning."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from esa_thesis.diagnose_channels import digest,event_diagnostics
from esa_thesis.models import MultivariateAE
from esa_thesis.evaluation.compare_scoring import paired_scores
from esa_thesis.masked_pilot import assessment_scopes
from esa_thesis.thresholds import postprocess
out=Path('results_longrun/diagnostics/calibration_model_comparison_2005_v1');out.mkdir(parents=True,exist_ok=False)
base=Path('results_longrun/diagnostics/baseline_2005_epoch5_v1')
roots={n:Path('results_longrun/development')/d/'2005' for n,d in [('pseudo','ae_baseline_remaining_seed42'),('reconstruction','ae_reconstruction_seed42')]}
files=[base/'calibration_inputs.npz',base/'calibration_channel_scores.npz',base/'results.json']
for r in roots.values():files.extend([r/'model_checkpoint.pt',r/'frozen_rule.json'])
before={str(p):digest(p) for p in files}
assert json.loads((base/'results.json').read_text())['reproduction']=='passed within 1e-6'
with np.load(base/'calibration_inputs.npz') as a:x=a['scaled'];times=a['timestamps']
np.testing.assert_array_equal(times,pd.date_range('2005-01-01','2005-04-01',freq='30s',inclusive='left').asi8)
saved={n:torch.load(r/'model_checkpoint.pt',map_location='cpu',weights_only=False) for n,r in roots.items()}
for key in ['features','scaler_median','scaler_iqr']:np.testing.assert_array_equal(saved['pseudo'][key],saved['reconstruction'][key])
torch.set_num_threads(1);model=MultivariateAE(8);model.load_state_dict(saved['reconstruction']['model_state_dict'])
print('Scoring full calibration with reconstruction-only checkpoint...',flush=True)
scores,uncovered=paired_scores(model,x,device='cpu',batch_size=64);assert uncovered==0
np.save(out/'reconstruction_calibration_score.npy',scores['window_mean'])
with np.load(base/'calibration_channel_scores.npz') as a:pscore=a['scores'].max(axis=0)
scopes=assessment_scopes(times);rows=[];events=[]
for name,score in [('pseudo',pscore),('reconstruction',scores['window_mean'])]:
 rule=json.loads((roots[name]/'frozen_rule.json').read_text())['rule']
 pred=postprocess(score>rule['threshold'],int(rule['merge_gap']),int(rule['min_dur']))
 metrics=scopes['system_all_channels'].score(pred)
 for k in ['EW_F_0.50','EW_precision','EW_recall','TPe','FPe','FNe']:
  if not np.isclose(metrics[k],rule['val_'+k],rtol=0,atol=1e-6):raise ValueError(f'{name}: frozen metric mismatch {k}: {metrics[k]} vs {rule["val_"+k]}')
 for scope,ev in scopes.items():rows.append(dict(model=name,scope=scope,threshold=rule['threshold'],**ev.score(pred)))
 e=event_diagnostics(scopes['system_all_channels'],pred,score);e['model']=name;e['annotated_on_inputs']=e.event_id.isin(scopes['input_channels_40_47'].by_id)
 e['peak_to_own_threshold']=e.max_score_on_inside_samples/rule['threshold'];events.append(e)
 np.save(out/f'{name}_prediction.npy',pred)
assert before=={str(p):digest(p) for p in files}
pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False);pd.concat(events).to_csv(out/'events.csv',index=False)
(out/'verification.json').write_text(json.dumps(dict(status='complete',inputs=before,scope='Entire Jan–Mar2005 calibration; both models own frozen rules; no training/assessment/tuning',limitations='Historical checkpoint comparison, not strict causal ablation; inputs reused from verified diagnostic. No current fullCSV rehash.'),indent=2)+'\n')
print(pd.DataFrame(rows)[['model','scope','EW_F_0.50','TPe','FPe','FNe']].to_string(index=False))
print(pd.concat(events)[['model','event_id','detected','annotated_on_inputs','peak_to_own_threshold']].to_string(index=False))
