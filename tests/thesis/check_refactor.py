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
extracted_modules = ('config', 'runtime', 'data', 'models', 'augmentation', 'metrics',
                     'scoring', 'thresholds', 'plots', 'checkpoints', 'training')
for module in extracted_modules:
 p = root / 'esa_thesis' / (module + '.py')
 for n in ast.parse(p.read_text()).body:
  if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in expected and p.name not in ('__main__.py',):
   assert ast.dump(n,include_attributes=False)==expected[n.name],(p,n.name)
   seen.add(n.name)
assert seen==set(expected),set(expected)-seen
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
 assert old.metrics(y,p)==metrics.metrics(y,p)
 for gap in [0,3,20]:
  for duration in [1,4]:
   np.testing.assert_array_equal(old.postprocess(p,gap,duration), thresholds.postprocess(p,gap,duration))
print(f'PASS: {len(seen)} unchanged definitions; both model outputs and state dictionaries; 10 metric cases; 60 postprocessing cases')
