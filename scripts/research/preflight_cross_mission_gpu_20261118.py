"""Synthetic shape/memory feasibility only; never reads research observations."""
import json
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from esa_thesis import config
from esa_thesis.models import MultivariateAE
from esa_thesis.masked_pilot import MaskedAE,reconstruction_error
from esa_thesis.short_augmentation import make_training_anomaly
from esa_thesis.calibration_audit import digest
torch.set_num_threads(2);torch.manual_seed(20261118)
rows=[]
for count,masked in ((26,False),(13,True)):
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    model=(MaskedAE(count) if masked else MultivariateAE(count)).cuda().train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    x=torch.randn(32,count,256,1,device='cuda')
    normal=reconstruction_error(model,x,masked,0);loss=normal.mean()
    if not masked:
        corrupt=make_training_anomaly(x,mode='short_no_roll')
        abnormal=reconstruction_error(model,corrupt,False,0)
        loss=loss+config.PA_LAMBDA*torch.relu(config.PA_MARGIN-abnormal/(normal.mean().detach()+1e-8)).mean()
    assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();torch.cuda.synchronize()
    rows.append(dict(channels=count,masked=masked,batch=32,peak_bytes=torch.cuda.max_memory_allocated(),loss=float(loss)))
    del model,optimizer,x,normal,loss
    if not masked:del corrupt,abnormal
    torch.cuda.empty_cache()
p=ROOT/'results_longrun/cross_mission_transfer_20261118_v1/gpu-preflight.json'
p.write_text(json.dumps(dict(passed=True,synthetic_only=True,checks=rows,script_sha256=digest(__file__),gpu=torch.cuda.get_device_name(0)),indent=2)+'\n')
print(p.read_text())
