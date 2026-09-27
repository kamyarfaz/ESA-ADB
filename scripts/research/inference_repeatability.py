"""Two independent CPU inference passes on fixed2004 calibration segments."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import torch
from esa_thesis.models import MultivariateAE
from esa_thesis.data import RobustChannelScaler
from esa_thesis.diagnose_channels import aggregate
from esa_thesis.thresholds import postprocess


def sha(data):return hashlib.sha256(data).hexdigest()


def model_sha(model):
    h=hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):
        h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def worker(out):
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    source=Path('results_longrun/development/ae_baseline_remaining_seed42/2004')
    cache=Path('results_longrun/diagnostics/baseline_2004_epoch35_v2/calibration_inputs.npz')
    checkpoint=source/'model_checkpoint.pt';rulepath=source/'frozen_rule.json'
    initial={str(p):sha(p.read_bytes()) for p in [cache,checkpoint,rulepath]}
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    features=[f'channel_{j}' for j in range(40,48)]
    if saved['features']!=features:raise ValueError('Wrong checkpoint features')
    frozen=json.loads(rulepath.read_text());rule=frozen['rule']
    if frozen['epoch']!=35:raise ValueError('Wrong checkpoint epoch')
    scaler=RobustChannelScaler();scaler.median_=np.asarray(saved['scaler_median'],dtype=np.float32);scaler.iqr_=np.asarray(saved['scaler_iqr'],dtype=np.float32)
    model=MultivariateAE(8).eval();model.load_state_dict(saved['model_state_dict'])
    before=model_sha(model)
    with np.load(cache) as f:
        times=f['timestamps'];values=f['values'];cached_scaled=f['scaled']
    anchors=['2004-01-01 05:00','2004-01-24 08:00','2004-02-14 15:00','2004-03-01 12:00']
    raw=[];scaled=[];time_blocks=[]
    for anchor in anchors:
        center=int(np.searchsorted(times,pd.Timestamp(anchor,tz='UTC').value))
        start=max(0,min(len(times)-2048,center-1024));start=(start//32)*32
        x=values[start:start+2048];z=scaler.transform(x)
        np.testing.assert_array_equal(z,cached_scaled[start:start+2048])
        raw.append(x);scaled.append(z);time_blocks.append(times[start:start+2048])
    raw=np.stack(raw);scaled=np.stack(scaled);time_blocks=np.stack(time_blocks)
    starts=np.arange(0,2048-256+1,32)
    windows=np.stack([block[s:s+256].T[:,:,None] for block in scaled for s in starts])
    inputs_sha=sha(windows.tobytes())
    recon=[]
    with torch.inference_mode():
        for begin in range(0,len(windows),16):
            recon.append(model(torch.from_numpy(windows[begin:begin+16])).numpy())
    recon=np.concatenate(recon)
    if not np.isfinite(recon).all():raise ValueError('Nonfinite model output')
    errors=((recon-windows)**2).mean(axis=(2,3)).reshape(4,len(starts),8)
    scores=np.stack([aggregate(e,starts,2048).max(axis=0) for e in errors])
    predictions=np.stack([postprocess(s>rule['threshold'],int(rule['merge_gap']),int(rule['min_dur'])) for s in scores])
    if sha(windows.tobytes())!=inputs_sha or model_sha(model)!=before:raise ValueError('Inputs or weights changed')
    for p,h in initial.items():
        if sha(Path(p).read_bytes())!=h:raise ValueError('Source changed')
    np.savez(out/'arrays.npz',raw=raw,scaled=scaled,timestamps=time_blocks,reconstruction=recon,errors=errors,scores=scores,predictions=predictions)
    metadata=dict(device='cpu',threads=1,epoch=35,anchors=anchors,blocks=4,samples_per_block=2048,windows=len(windows),
        input_tensor_sha256=inputs_sha,model_sha256=before,source_sha256=initial,threshold=rule['threshold'],
        scope='Fixed calibration blocks; edge context differs from full calibration aggregation. No event metrics, tuning, assessment or training.')
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('Worker completed',out,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);p.add_argument('--worker',action='store_true');args=p.parse_args()
    if args.worker:return worker(args.output)
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    for name in ['pass1','pass2']:
        with (out/(name+'.log')).open('w') as log:
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker','--output',str(out/name)],stdout=log,stderr=subprocess.STDOUT,check=True)
    left=json.loads((out/'pass1/metadata.json').read_text());right=json.loads((out/'pass2/metadata.json').read_text())
    rows=[]
    with np.load(out/'pass1/arrays.npz') as a,np.load(out/'pass2/arrays.npz') as b:
        for key in a.files:
            x=a[key];y=b[key];equal=np.array_equal(x,y)
            row=dict(stage=key,exact_equal=equal,differing_elements=int(np.count_nonzero(x!=y)),sha256_a=sha(x.tobytes()),sha256_b=sha(y.tobytes()))
            if np.issubdtype(x.dtype,np.floating):row['max_absolute_difference']=float(np.max(np.abs(x-y)))
            rows.append(row)
    result=dict(metadata_equal=left==right,stages=rows,all_exact=left==right and all(r['exact_equal'] for r in rows),protocol=left)
    (out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
