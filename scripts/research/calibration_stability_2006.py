"""Read-only sensitivity audit of the frozen epoch10 baseline; no assessment access."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from esa_thesis.diagnose_channels import digest, event_diagnostics
from esa_thesis.masked_pilot import assessment_scopes
from esa_thesis.thresholds import postprocess

root=Path('results_longrun/long_baseline/ch40_47_train2000_2005_seed42/full_period')
source=Path('results_longrun/diagnostics/ch40_47_epoch10_v2')
out=Path('docs/research/tables/calibration-stability-2006-2026-09-26')
out.mkdir(parents=True,exist_ok=False)
paths=[source/'calibration_channel_scores.npz',source/'protocol.json',root/'frozen_rule.json',root/'history.csv']
before={str(p):digest(p) for p in paths}
protocol=json.loads((source/'protocol.json').read_text())
assert digest(root/'frozen_rule.json')==protocol['frozen_rule_sha256']
rule=json.loads((root/'frozen_rule.json').read_text())['rule']
with np.load(paths[0]) as data:
 t=data['timestamps'];scores=data['scores'].max(axis=0);saved=data['prediction']
assert np.array_equal(t,pd.date_range('2006-01-01','2007-01-01',freq='30s',inclusive='left').asi8)
assert np.isfinite(scores).all()
pred=postprocess(scores>rule['threshold'],int(rule['merge_gap']),int(rule['min_dur']))
np.testing.assert_array_equal(pred,saved)
scopes=assessment_scopes(t)
metrics=scopes['system_all_channels'].score(pred)
for k in ['EW_F_0.50','TPe','FPe','FNe']:
 assert np.isclose(metrics[k],rule['val_'+k],rtol=0,atol=1e-9)
rows=[]
for factor in [.5,.75,.9,1,1.1,1.25,1.5,2]:
 p=postprocess(scores>rule['threshold']*factor,int(rule['merge_gap']),int(rule['min_dur']))
 for scope,ev in scopes.items():rows.append(dict(factor=factor,threshold=rule['threshold']*factor,scope=scope,**ev.score(p)))
pd.DataFrame(rows).to_csv(out/'threshold_sensitivity.csv',index=False)
quarters=[]
for q in range(1,5):
 keep=pd.DatetimeIndex(t).quarter==q
 for scope,ev in assessment_scopes(t[keep]).items():quarters.append(dict(quarter=q,scope=scope,**ev.score(pred[keep])))
pd.DataFrame(quarters).to_csv(out/'frozen_quarter_scores.csv',index=False)
events=event_diagnostics(scopes['system_all_channels'],pred,scores)
events['annotated_on_inputs']=events.event_id.isin(scopes['input_channels_40_47'].by_id)
events.to_csv(out/'events.csv',index=False)
history=pd.read_csv(root/'history.csv').dropna(subset=['calibration_f05'])
history.to_csv(out/'checkpoint_scores.csv',index=False)
assert before=={str(p):digest(p) for p in paths}
(out/'audit.json').write_text(json.dumps(dict(inputs=before,reproduction=metrics,policy='2006 calibration only; threshold perturbations diagnostic, not adopted; no assessment files read'),indent=2)+'\n')
print(pd.DataFrame(rows).query("scope == 'system_all_channels'")[['factor','EW_F_0.50','TPe','FPe','FNe']].to_string(index=False))
print(pd.DataFrame(quarters).query("scope == 'system_all_channels'")[['quarter','EW_F_0.50','TPe','FPe','FNe']].to_string(index=False))
print(events[['event_id','detected','annotated_on_inputs']].to_string(index=False))
