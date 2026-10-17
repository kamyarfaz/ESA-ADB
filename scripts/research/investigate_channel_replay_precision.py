"""Diagnose the stopped replay without changing a frozen decision."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from esa_thesis import config
from esa_thesis.development import load_period
from esa_thesis.data import RobustChannelScaler
from esa_thesis.diagnose_channels import FEATURES,window_errors,aggregate,digest
from esa_thesis.models import MultivariateAE
from esa_thesis.thresholds import postprocess

ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--fold',type=int,choices=[2001,2002],default=2001)
p.add_argument('--seed',type=int,choices=[42,43,44],default=42)
p.add_argument('--threads',type=int,default=2)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
SOURCE=ROOT/f'results_longrun/development/inner_selection_cuda_v2/{a.fold}/seed{a.seed}'
OUT=a.output
if OUT.exists():raise ValueError('Preserve investigation')
OUT.mkdir(parents=True)
torch.set_num_threads(a.threads)
decision=json.loads((SOURCE/'frozen_decisions.json').read_text())['calibration_selected'];rule=decision['rule']
saved=torch.load(SOURCE/f"epoch_{decision['epoch']}"/'model.pt',map_location='cpu',weights_only=False)
raw,_,times=load_period(config.TRAIN_FILE,FEATURES,f'{a.fold}-01-01',f'{a.fold}-07-01')
scaler=RobustChannelScaler();scaler.median_=saved['scaler_median'];scaler.iqr_=saved['scaler_iqr'];x=scaler.transform(raw)
expected=np.load(SOURCE/f"epoch_{decision['epoch']}"/'calibration_score.npy')
predict=lambda s:postprocess(s>rule['threshold'],int(rule['merge_gap']),int(rule['min_dur']))
records=[]
for device in ['cuda','cpu']:
    model=MultivariateAE(8).to(device);model.load_state_dict(saved['model_state_dict']);model.eval()
    starts=np.arange(0,len(x)-config.SEQ_LEN+1,config.SCORE_STRIDE)
    errors=window_errors(model,x,starts,device,64,device)
    values=aggregate(errors,starts,len(x));scores=values.max(axis=0)
    bad=np.flatnonzero(predict(scores)!=predict(expected))
    np.save(OUT/f'{device}_scores.npy',scores)
    np.save(OUT/f'{device}_channel_scores.npy',values)
    record=dict(device=device,max_abs_error=float(np.max(np.abs(scores-expected))),scores_within_tolerance=bool(np.allclose(scores,expected,rtol=1e-5,atol=1e-6)),mismatching_samples=len(bad),threshold=rule['threshold'],
        mismatches=[dict(timestamp=str(np.datetime64(int(times[i]),'ns')),old=float(expected[i]),replayed=float(scores[i]),difference=float(scores[i]-expected[i])) for i in bad])
    records.append(record);print(json.dumps({k:v for k,v in record.items() if k!='mismatches'}),flush=True)
(OUT/'report.json').write_text(json.dumps(dict(records=records,threads=a.threads,fold=a.fold,seed=a.seed,torch=torch.__version__,gpu=torch.cuda.get_device_name(0),source_checkpoint_sha256=digest(SOURCE/f"epoch_{decision['epoch']}"/'model.pt'),unchanged_threshold=True),indent=2)+'\n')
