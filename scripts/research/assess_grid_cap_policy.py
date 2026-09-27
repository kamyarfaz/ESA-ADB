"""Freeze calibration-selected rules, then score all historical development folds."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from esa_thesis import config
from esa_thesis.diagnose_channels import digest
from esa_thesis.masked_pilot import assessment_scopes
from esa_thesis.thresholds import postprocess

out=Path('results_longrun/development/grid_cap_policy_v1')
out.mkdir(parents=True,exist_ok=False)
comparison=Path('results_longrun/diagnostics/grid_cap_comparison_v2/comparison.csv')
verification=comparison.parent/'verification.json'
assert json.loads(verification.read_text())['status']=='complete'
df=pd.read_csv(comparison);rules={};hashes={str(comparison):digest(comparison),str(verification):digest(verification)}
for year in (2003,2004,2005):
 root=Path('results_longrun/development')/('ae_baseline_2003_seed42' if year==2003 else 'ae_baseline_remaining_seed42')/str(year)
 original=root/'frozen_rule.json';hashes[str(original)]=digest(original)
 candidates=df[(df.year==year)&(df.grid=='dense')&(df.policy=='nominal_false_alarm')]
 assert len(candidates)==1
 rules[str(year)]=dict(root=str(root),original=json.loads(original.read_text())['rule'],candidate=candidates.iloc[0].to_dict())
# Freeze every rule before reading ANY assessment score or prediction.
(out/'frozen_rules.json').write_text(json.dumps(rules,indent=2)+'\n')
(out/'protocol.json').write_text(json.dumps(dict(assessment='April-December2003/2004/2005, all folds, both scopes',policy='Dense calibration quantiles + nominal false-alarm duration <=1%; same checkpoints; no assessment selection',limitations='Previously inspected development data, not untouched test. No2007–2013 access. Existing cached scores; no training.',calibration_inputs=hashes),indent=2)+'\n')
rows=[]
for year,entry in rules.items():
 root=Path(entry['root']);files=[root/'assessment_score.npy',root/'assessment_prediction.npy',root/'results.json',config.ANNOTATIONS_FILE,config.ANOMALY_TYPES_FILE]
 before={str(f):digest(f) for f in files}
 score=np.load(files[0]);saved_pred=np.load(files[1]);saved=json.loads(files[2].read_text())
 t=pd.date_range(f'{year}-04-01',f'{int(year)+1}-01-01',freq='30s',inclusive='left').asi8
 assert score.shape==t.shape and np.isfinite(score).all()
 scopes=assessment_scopes(t)
 for name in ['original','candidate']:
  rule=entry[name];pred=postprocess(score>rule['threshold'],int(rule['merge_gap']),int(rule['min_dur']))
  if name=='original':
   np.testing.assert_array_equal(pred,saved_pred)
   actual=scopes['system_all_channels'].score(pred)
   for k,v in actual.items():
    if isinstance(v,(int,float)):assert np.isclose(v,saved[k],rtol=0,atol=1e-9),(year,k)
  for scope,ev in scopes.items():rows.append(dict(year=int(year),arm=name,scope=scope,calibration_f05=rule['val_esa_f05'],threshold=rule['threshold'],merge_gap=rule['merge_gap'],min_dur=rule['min_dur'],predicted_fraction=float(pred.mean()),**ev.score(pred)))
  if name=='candidate':np.save(out/f'{year}_candidate_prediction.npy',pred)
 assert before=={str(f):digest(f) for f in files}
 hashes.update(before)
assert all(digest(Path(p))==h for p,h in hashes.items())
pd.DataFrame(rows).to_csv(out/'results.csv',index=False)
(out/'verification.json').write_text(json.dumps(dict(status='complete',input_sha256=hashes,original_reproduction='All original predictions and metrics reproduced'),indent=2)+'\n')
print(pd.DataFrame(rows)[['year','arm','scope','EW_F_0.50','TPe','FPe','FNe']].to_string(index=False))
