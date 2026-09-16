import ast, importlib, sys, zipfile
from pathlib import Path
import numpy as np
import torch
from esa_thesis import tracking, models, metrics, thresholds
root=Path(__file__).resolve().parents[2]
with zipfile.ZipFile(root/'archive/pre_organization_source_2026-09-15.zip') as z:
 name=next(n for n in z.namelist() if 'sweep' in n and n.endswith('.py'))
 source=z.read(name).decode()
original=ast.parse(source)
expected={n.name:ast.dump(n,include_attributes=False) for n in original.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
seen=set()
# Evaluation/training are intentionally changed; their new behavior has dedicated tests.
unchanged = ('RobustChannelScaler', 'NormalWindows', 'AllWindows', 'ForecastWindows',
             'MultivariateAE', 'MLPForecaster', 'make_pseudo_anomaly',
             'score_series', 'score_series_mlp', 'ensemble_scores', 'postprocess')
for module in ('data', 'models', 'augmentation', 'scoring', 'thresholds'):
 for node in ast.parse((root/'esa_thesis'/(module+'.py')).read_text()).body:
  if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in unchanged:
   assert ast.dump(node,include_attributes=False)==expected[node.name],(module,node.name)
   seen.add(node.name)
assert seen==set(unchanged)
sys.modules['wandb_utils']=tracking
old=type(sys)('original_pipeline')
exec(compile(source,name,'exec'),old.__dict__)
torch.set_num_threads(2)
for name in ['MultivariateAE','MLPForecaster']:
 a=getattr(old,name)(2).eval(); b=getattr(models,name)(2).eval()
 b.load_state_dict(a.state_dict())
 x=torch.randn(2,2,256,1)
 with torch.no_grad():
  torch.testing.assert_close(a(x),b(x),rtol=0,atol=0)
for seed in range(10):
 rng=np.random.default_rng(seed); y=rng.integers(0,2,200); p=rng.integers(0,2,200)
 assert old.metrics(y,p)==metrics.legacy_metrics(y,p)
 for gap in [0,3,20]:
  for duration in [1,4]:
   np.testing.assert_array_equal(old.postprocess(p,gap,duration), thresholds.postprocess(p,gap,duration))
print(f'PASS: {len(seen)} unchanged definitions; both model outputs and state dictionaries; 10 legacy metric cases; 60 postprocessing cases')
