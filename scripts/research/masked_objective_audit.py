import json
from pathlib import Path
import numpy as np
import torch
from esa_thesis import config
from esa_thesis.masked_pilot import MaskedAE,reconstruction_error,patch_mask
from esa_thesis.models import MultivariateAE
from esa_thesis.augmentation import make_pseudo_anomaly
from esa_thesis.data import RobustChannelScaler
from esa_thesis.development import load_period
from esa_thesis.diagnose_channels import FEATURES,TRAIN_SHA,digest
from esa_thesis.runtime import set_seed

torch.set_num_threads(4)
root=Path('results_longrun/development/masked_pilot_2004_2005_seed42_v2/2004')
print('Verifying training CSV',flush=True)
assert digest(config.TRAIN_FILE)==TRAIN_SHA
x,y,_=load_period(config.TRAIN_FILE,FEATURES,None,'2004-01-01')
starts=np.load(root/'training_starts.npy')
starts=np.random.default_rng(20260924).choice(starts,128,replace=False)
assert all(np.all(y[s:s+config.SEQ_LEN]==0) for s in starts)
ck=torch.load(root/'baseline/model_checkpoint.pt',map_location='cpu',weights_only=False)
sc=RobustChannelScaler();sc.median_=ck['scaler_median'];sc.iqr_=ck['scaler_iqr']
w=np.stack([sc.transform(x[s:s+config.SEQ_LEN]).T[:,:,None] for s in starts])
del x,y
inputs=torch.from_numpy(w)
set_seed(20260924);corrupted=make_pseudo_anomaly(inputs)
rows=[]
with torch.no_grad():
 for arm in ['baseline','masked']:
  for stage in ['best','last']:
   model=MaskedAE(8) if arm=='masked' else MultivariateAE(8)
   cp=torch.load(root/arm/('model_checkpoint.pt' if stage=='best' else 'last.pt'),map_location='cpu',weights_only=False)
   model.load_state_dict(cp['model_state_dict' if stage=='best' else 'model']);model.eval()
   normals=[];abnormals=[];penalties=[];unchanged=[]
   for phase in range(4):
    for b in range(0,128,16):
     a=inputs[b:b+16];c=corrupted[b:b+16]
     n=reconstruction_error(model,a,arm=='masked',phase)
     z=reconstruction_error(model,c,arm=='masked',phase)
     normals.extend(n.tolist());abnormals.extend(z.tolist())
     penalties.extend(torch.relu(config.PA_MARGIN-z/(n.mean()+1e-8)).tolist())
     hidden=patch_mask(16,phase,'cpu').repeat_interleave(16)
     unchanged.extend(((a-c)[:,:,hidden,:]==0).flatten(1).all(1).tolist())
   rows.append(dict(arm=arm,checkpoint=stage,epoch=1 if stage=='best' else cp['epoch'],normal_mse_mean=float(np.mean(normals)),pseudo_mse_mean=float(np.mean(abnormals)),pseudo_penalty_mean=float(np.mean(penalties)),pseudo_mse_median=float(np.median(abnormals)),normal_mse_median=float(np.median(normals)),fraction_pseudo_error_above_normal=float(np.mean(np.array(abnormals)>np.array(normals))),fraction_hidden_targets_unchanged=float(np.mean(unchanged))))
   print(rows[-1],flush=True)
assert digest(config.TRAIN_FILE)==TRAIN_SHA
out=dict(protocol='128 seeded nominal windows from saved 2004 training starts; identical pseudo inputs; eval mode; four mask phases; batch16 penalty denominator; not historical training losses or deployed scores; no calibration/assessment access',seed=20260924,starts=starts.tolist(),rows=rows)
p=Path('docs/research/tables/masked-objective-audit-2026-09-24/training_probe.json');p.write_text(json.dumps(out,indent=2)+'\n')
