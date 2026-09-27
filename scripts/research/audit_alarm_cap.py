"""Calibration-only comparison on historical candidate grids; no decisions changed."""
from pathlib import Path
import json
import pandas as pd
import numpy as np
from esa_thesis.diagnose_channels import digest
from esa_thesis.thresholds import select_rows
out=Path('docs/research/tables/alarm-cap-audit-2026-09-26')
out.mkdir(parents=True,exist_ok=False)
rows=[];hashes={}
for year in (2003,2004,2005):
 root=Path('results_longrun/development')/('ae_baseline_2003_seed42' if year==2003 else 'ae_baseline_remaining_seed42')/str(year)
 path=root/'calibration_candidates.csv';hashes[str(path)]=digest(path)
 df=pd.read_csv(path);original=select_rows(df).iloc[0]
 frozen=json.loads((root/'frozen_rule.json').read_text())['rule']
 for key in ['threshold','merge_gap','min_dur','val_esa_f05']:
  assert np.isclose(original[key],frozen[key],rtol=0,atol=1e-9)
 for policy,mask in [('total_alarm_fraction_lte_1pct',df.val_pred_anomaly_rate<=.01),('nominal_false_alarm_fraction_lte_1pct',df.val_nominal_false_alarm_rate<=.01)]:
  candidates=df[mask & ~df.val_saturated]
  selected=candidates.sort_values(['val_esa_f05','val_false_positive_seconds','threshold','merge_gap','min_dur'],ascending=[False,True,False,True,True],kind='stable').iloc[0]
  rows.append(dict(year=year,policy=policy,eligible_candidates=len(candidates),**selected.to_dict()))
 df[(df.merge_gap==0)&(df.min_dur==1)].to_csv(out/f'{year}_unmerged_grid.csv',index=False)
 assert digest(path)==hashes[str(path)]
pd.DataFrame(rows).to_csv(out/'comparison.csv',index=False)
(out/'protocol.json').write_text(json.dumps(dict(inputs=hashes,scope='Jan-Mar calibration; saved selected checkpoints and original threshold grids only',limitation='Exploratory policy; 1% nominal duration is not an established operational requirement; excludes checkpoint reselection. No assessment access or frozen decision changes.'),indent=2)+'\n')
print('All three original rules reproduced; alternative cap selects identical rules.')
