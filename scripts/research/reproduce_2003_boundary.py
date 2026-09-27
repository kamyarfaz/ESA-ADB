"""Isolate a frozen threshold boundary; perturbations are diagnostic only."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from esa_thesis.diagnose_channels import aggregate,digest
from esa_thesis.evaluation.compare_scoring import paired_scores
from esa_thesis.models import MultivariateAE
from esa_thesis.protocol import evaluator_for

out=Path('outputs/calibration-boundary-2003-20260926');out.mkdir(parents=True,exist_ok=False)
r=Path('results_longrun/diagnostics/baseline_2003_epoch25_v1');source=Path('results_longrun/development/ae_baseline_2003_seed42/2003')
files=[r/'calibration_inputs.npz',r/'calibration_window_errors.npz',source/'model_checkpoint.pt',source/'frozen_rule.json']
before={str(p):digest(p) for p in files}
with np.load(files[0]) as a:x=a['scaled'];t=a['timestamps']
with np.load(files[1]) as a:cached=aggregate(a['errors'],a['starts'],len(t)).max(axis=0)
frozen=json.loads(files[3].read_text())['rule'];threshold=frozen['threshold']
torch.set_num_threads(1)
saved=torch.load(files[2],map_location='cpu',weights_only=False)
model=MultivariateAE(8);model.load_state_dict(saved['model_state_dict'])
print('Running historical paired scorer on cached calibration inputs...',flush=True)
scored,uncovered=paired_scores(model,x,device='cpu',batch_size=64)
score=scored['window_mean'];ev=evaluator_for(t)
near=np.abs(cached-threshold)<1e-6
rows=[]
for delta in [0,1e-6]:
 pred=cached>threshold+delta;starts,ends,_=ev.prediction_intervals(pred)
 rows.append(dict(diagnostic_threshold_delta=delta,metrics=ev.score(pred),intervals=[[str(pd.Timestamp(a)),str(pd.Timestamp(b))] for a,b in zip(starts,ends)]))
assert before=={str(p):digest(p) for p in files}
result=dict(inputs=before,cuda_available=torch.cuda.is_available(),uncovered=uncovered,paired_equals_diagnostic=bool(np.array_equal(score,cached)),max_score_difference=float(np.max(np.abs(score-cached))),paired_metrics=ev.score(score>threshold),threshold=threshold,adjacent_float32=float(np.nextafter(np.float32(threshold),np.float32(np.inf))),boundary_scores=np.unique(cached[near]).tolist(),boundary_samples=int(near.sum()),boundary_first=str(pd.Timestamp(t[near][0])),boundary_last=str(pd.Timestamp(t[near][-1])),diagnostic_perturbations=rows,limitation='No historical GPU score array available for direct numerical comparison. No threshold modification or failed diagnostic acceptance.')
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
