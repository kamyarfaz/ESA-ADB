"""Calibration-only audit of saved grids; no rule replacement or validation access."""
import json
from pathlib import Path
import pandas as pd
import numpy as np
from esa_thesis.thresholds import select_rows
from esa_thesis.diagnose_channels import digest
out=Path('docs/research/tables/calibration-selection-audit-2026-10-08')
if out.exists():raise ValueError('Preserve output')
rows=[];hashes={}
for method,rootname in [('joint','inner_selection_cuda_v2'),('isolated','inner_isolation_cuda_v1'),('score_change','score_change_v1'),('telemetry','telemetry_reference_v1')]:
 root=Path('results_longrun/development')/rootname
 for fold in ('2001','2002'):
  for seed in ([None] if method=='telemetry' else [42,43,44]):
   folder=root/fold if seed is None else root/fold/f'seed{seed}'
   if method in ('joint','isolated'):
    decision=folder/'frozen_decisions.json';r=json.loads(decision.read_text())['calibration_selected'];epoch=r['epoch'];rule=r['rule'];gridpath=folder/f'epoch_{epoch}'/'calibration_candidates.csv'
   else:
    decision=folder/'frozen_rule.json';r=json.loads(decision.read_text());epoch=r.get('epoch');rule=r['rule'];gridpath=folder/'calibration_candidates.csv'
   hashes[str(decision)]=digest(decision);hashes[str(gridpath)]=digest(gridpath)
   grid=pd.read_csv(gridpath);chosen=select_rows(grid).iloc[0]
   assert all(np.isclose(chosen[k],rule[k],atol=1e-12,rtol=0) for k in ('threshold','merge_gap','min_dur','val_esa_f05'))
   # Audit counterfactuals only: never persist them as detector rules.
   ordered=grid.sort_values(['val_esa_f05','val_false_positive_seconds','threshold','merge_gap','min_dur'],ascending=[False,True,False,True,True],kind='stable')
   unrestricted=ordered.iloc[0]
   feasible=grid[(grid.val_pred_anomaly_rate<=.01)&(~grid.val_saturated)]
   more=feasible[feasible.val_TPe>chosen.val_TPe].sort_values(['val_esa_f05','val_false_positive_seconds'],ascending=[False,True])
   for label,row in [('selected',chosen),('best_without_cap',unrestricted),('best_more_recall_with_cap',None if more.empty else more.iloc[0])]:
    rows.append(dict(method=method,fold=fold,seed='deterministic' if seed is None else seed,epoch=epoch,diagnostic=label,exists=row is not None,
       **({k:row[k] for k in ['threshold','val_esa_f05','val_TPe','val_FPe','val_FNe','val_pred_anomaly_rate','val_false_positive_seconds']} if row is not None else {})))
out.mkdir(parents=True);df=pd.DataFrame(rows);df.to_csv(out/'calibration_tradeoffs.csv',index=False)
assert all(digest(f)==h for f,h in hashes.items())
(out/'verification.json').write_text(json.dumps(dict(status='complete',selected_rules_verified=20,source_hashes=hashes,scope='Saved selected-epoch calibration grids only; no validation scores loaded, no rule changes'),indent=2)+'\n')
print(df[['method','fold','seed','diagnostic','exists','val_esa_f05','val_TPe','val_FPe','val_pred_anomaly_rate']].to_string(index=False))
