"""Frozen 2002 checkpoint replay and fixed monthly nominal-window probes."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from esa_thesis import config
from esa_thesis.data import RobustChannelScaler
from esa_thesis.development import load_period
from esa_thesis.diagnose_channels import FEATURES, TRAIN_SHA, digest, window_errors, aggregate
from esa_thesis.models import MultivariateAE


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Use a new output')
    assert torch.cuda.is_available();torch.set_num_threads(2)
    roots={arm:Path('results_longrun/development')/name for arm,name in
           [('lambda1','inner_selection_cuda_v2'),('lambda0.1','inner_lambda01_cuda_v1')]}
    hashes={}
    for root in roots.values():
        plan=json.loads((root/'protocol.json').read_text())
        assert all(digest(Path('esa_thesis')/f)==h for f,h in plan['source_sha256'].items())
        hashes[str(root/'protocol.json')]=digest(root/'protocol.json')
    print('Checking prepared CSV integrity',flush=True);assert digest(config.TRAIN_FILE)==TRAIN_SHA
    print('Loading 2000–2002 only',flush=True)
    raw,y,times=load_period(config.TRAIN_FILE,FEATURES,None,'2003-01-01')
    months=pd.to_datetime(times,utc=True).strftime('%Y-%m').to_numpy()
    starts=np.arange(0,len(raw)-256+1,32);c=np.r_[0,np.cumsum(y!=0)]
    valid=starts[(c[starts+256]==c[starts])&(months[starts]==months[starts+255])]
    selected=[]
    for month in np.unique(months):
        v=valid[months[valid]==month]
        assert len(v)>0
        selected.extend(v[np.linspace(0,len(v)-1,min(128,len(v)),dtype=int)])
    selected=np.asarray(selected);sample_months=months[selected]
    val_start=np.searchsorted(times,pd.Timestamp('2002-07-01',tz='UTC').value)
    vs=np.arange(0,len(raw)-val_start-256+1,32)
    a.output.mkdir(parents=True);np.save(a.output/'sample_starts.npy',selected)
    rows=[];replays=[]
    for arm,root in roots.items():
        for seed in (42,43,44):
            folder=root/'2002'/f'seed{seed}'
            manifest=json.loads((folder/'complete.json').read_text())
            assert all(digest(folder/f)==h for f,h in manifest.items())
            d=json.loads((folder/'frozen_decisions.json').read_text())['calibration_selected']
            cp=folder/f"epoch_{d['epoch']}"/'model.pt';hashes[str(cp)]=digest(cp)
            saved=torch.load(cp,map_location='cpu',weights_only=False)
            scaler=RobustChannelScaler();scaler.median_=saved['scaler_median'];scaler.iqr_=saved['scaler_iqr']
            x=scaler.transform(raw)
            model=MultivariateAE(8).cuda();model.load_state_dict(saved['model_state_dict']);model.eval()
            errors=window_errors(model,x[val_start:],vs,'cuda',64,f'{arm}/{seed} validation replay')
            channels=aggregate(errors,vs,len(raw)-val_start)
            original=np.load(folder/'calibration_selected_validation_score.npy')
            delta=float(np.max(np.abs(channels.max(axis=0)-original)))
            np.testing.assert_allclose(channels.max(axis=0),original,rtol=1e-5,atol=1e-5)
            replays.append(dict(arm=arm,seed=seed,max_abs_difference=delta))
            # Fixed evenly spaced wholly nominal windows, independent of errors.
            probes=[];observations=[];recons=[]
            with torch.no_grad():
                for start in range(0,len(selected),64):
                    ix=selected[start:start+64,None]+np.arange(256)[None,:]
                    obs=x[ix];tensor=torch.from_numpy(obs.transpose(0,2,1)[:,:,:,None].copy()).cuda()
                    rec=model(tensor).cpu().numpy()[:,:,:,0].transpose(0,2,1)
                    probes.append(np.mean((obs-rec)**2,axis=1));observations.append(obs.mean(axis=1));recons.append(rec.mean(axis=1))
            e=np.concatenate(probes);om=np.concatenate(observations);rm=np.concatenate(recons)
            for month in np.unique(sample_months):
                keep=sample_months==month;ix=selected[keep,None]+np.arange(256)[None,:]
                unscaled=raw[ix];z=(unscaled-scaler.median_)/scaler.iqr_
                for j,ch in enumerate(FEATURES):
                    rows.append(dict(arm=arm,seed=seed,month=month,channel=ch,windows=int(keep.sum()),
                        prepared_input_median=float(np.median(unscaled[:,:,j])),
                        scaled_window_mean=float(om[keep,j].mean()),reconstruction_window_mean=float(rm[keep,j].mean()),
                        mse_median=float(np.median(e[keep,j])),mse_q99=float(np.percentile(e[keep,j],99)),
                        winner_fraction=float(np.mean(e[keep].argmax(axis=1)==j)),
                        clipping_fraction=float(np.mean(np.abs(z[:,:,j])>config.CLIP_Z)),threshold=d['rule']['threshold']))
            pd.DataFrame(rows).to_csv(a.output/'channel_monthly.csv',index=False)
            pd.DataFrame(replays).to_csv(a.output/'replay.csv',index=False)
            print(arm,seed,'complete',flush=True)
    assert digest(config.TRAIN_FILE)==TRAIN_SHA
    assert all(digest(f)==h for f,h in hashes.items())
    (a.output/'verification.json').write_text(json.dumps(dict(status='complete',train_sha256=TRAIN_SHA,
        sources=hashes,windows_per_month=128,selection='Evenly spaced globally nominal score-stride windows wholly within each month',
        limitations='Prepared inputs, not native raw telemetry; sampled window statistics, not pointwise or pooled event metrics; validation inspected for diagnosis only'),indent=2)+'\n')


if __name__=='__main__':main()
