"""Bounded repeatability and threshold-sensitivity audit; no tuning or training."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from esa_thesis import config
from esa_thesis.augmentation_comparison import score
from esa_thesis.data import RobustChannelScaler
from esa_thesis.development import load_period
from esa_thesis.diagnose_channels import digest, FEATURES, TRAIN_SHA
from esa_thesis.models import MultivariateAE
from esa_thesis.thresholds import postprocess

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'results_longrun/development/inner_selection_cuda_v2'
RUN=SOURCE/'2002/seed44'
PREVIOUS=ROOT/'results_longrun/diagnostics/channel_error_precision_2002_seed44_threads2_20261017'
OUT=ROOT/'results_longrun/diagnostics/calibration_numerical_audit_20261017_v2'
if OUT.exists(): raise ValueError('Preserve existing audit')
OUT.mkdir()
plan=json.loads((SOURCE/'protocol.json').read_text())
dependencies=['augmentation_comparison.py','diagnose_channels.py','models.py','config.py','data.py','development.py','thresholds.py','runtime.py']
historical=ROOT/'results_longrun/diagnostics/calibration_numerical_audit_20261017/historical_development.py'
assert digest(historical)==plan['source_sha256']['development.py']
def loader_ast(path):
    return ast.dump(next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='load_period'))
assert loader_ast(historical)==loader_ast(ROOT/'esa_thesis/development.py')
for name in dependencies:
    if name!='development.py': assert digest(ROOT/'esa_thesis'/name)==plan['source_sha256'][name],name
for name,h in json.loads((RUN/'complete.json').read_text()).items(): assert digest(RUN/name)==h,name
assert digest(config.TRAIN_FILE)==TRAIN_SHA
print('Source artifacts, inference code and raw data hashes verified',flush=True)
decision=json.loads((RUN/'frozen_decisions.json').read_text())['calibration_selected'];rule=decision['rule']
checkpoint=RUN/f"epoch_{decision['epoch']}"/'model.pt'
saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
torch.set_num_threads(2)
raw,_,times=load_period(config.TRAIN_FILE,FEATURES,'2002-01-01','2002-07-01')
np.testing.assert_array_equal(times,np.load(SOURCE/'2002/calibration_timestamps.npy'))
scaler=RobustChannelScaler();scaler.median_=saved['scaler_median'];scaler.iqr_=saved['scaler_iqr'];x=scaler.transform(raw)
model=MultivariateAE(8).to('cuda');model.load_state_dict(saved['model_state_dict']);model.eval()
expected=np.load(RUN/f"epoch_{decision['epoch']}"/'calibration_score.npy')
predict=lambda s:postprocess(s>rule['threshold'],int(rule['merge_gap']),int(rule['min_dur']))
arrays={'historical':expected,'previous_cuda':np.load(PREVIOUS/'cuda_scores.npy'),'previous_cpu':np.load(PREVIOUS/'cpu_scores.npy')}
for i in range(2):
 arrays[f'repeat_cuda_{i+1}']=score(model,x,SimpleNamespace(device='cuda',batch_size=64))
 np.save(OUT/f'repeat_cuda_{i+1}.npy',arrays[f'repeat_cuda_{i+1}'])
rows=[]
for name,s in arrays.items():
 bad=np.flatnonzero(~np.isclose(s,expected,rtol=1e-5,atol=1e-6));changed=np.flatnonzero(predict(s)!=predict(expected))
 rows.append(dict(name=name,score_tolerance_pass=not len(bad),score_fail_samples=len(bad),mask_diff_samples=len(changed),max_abs_error=float(np.abs(s-expected).max()),
   score_fail_reference_range=[float(expected[bad].min()),float(expected[bad].max())] if len(bad) else None,
   score_fail_max_abs_error=float(np.abs(s[bad]-expected[bad]).max()) if len(bad) else 0,
   mask_changes_at_historical_threshold=bool(np.all(expected[changed]==rule['threshold'])),
   changed_values=[dict(timestamp=str(np.datetime64(int(times[j]),'ns')),reference=float(expected[j]),replay=float(s[j]),float32_ulps=float((s[j]-expected[j])/np.spacing(expected[j]))) for j in changed]))
report=dict(status='audit_only_original_gate_unchanged',rule=rule,rows=rows,
 repeat_cuda_bitwise_equal=bool(np.array_equal(arrays['repeat_cuda_1'],arrays['repeat_cuda_2'])),
 prior_cuda_bitwise_equal=bool(np.array_equal(arrays['repeat_cuda_1'],arrays['previous_cuda'])),
 source_hashes_verified_except_development_module=True,load_period_ast_identical_to_historical=True,checkpoint_sha256=digest(checkpoint),data_sha256=TRAIN_SHA,
 torch=torch.__version__,gpu=torch.cuda.get_device_name(0),threads=2,
 inference_source_sha256={n:digest(ROOT/'esa_thesis'/n) for n in dependencies},
 artifacts={p.name:digest(p) for p in OUT.glob('*.npy')})
(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['rows','inference_source_sha256','artifacts','rule']}),flush=True)
for row in rows: print(json.dumps({k:v for k,v in row.items() if k!='changed_values'}),flush=True)
