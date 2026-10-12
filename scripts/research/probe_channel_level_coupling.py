"""Training/calibration-only input interventions on frozen 2002-fold AEs.

Synthetic probes explain model sensitivity; they are not benchmark assessments.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from esa_thesis import config
from esa_thesis.data import RobustChannelScaler
from esa_thesis.development import load_period
from esa_thesis.diagnose_channels import FEATURES, TRAIN_SHA, digest
from esa_thesis.models import MultivariateAE


def recenter_channel(windows, channel, target):
    result = windows.copy()
    result[:, :, channel] += target - windows[:, :, channel].mean(axis=1, keepdims=True)
    return result


@torch.no_grad()
def reconstruct(model, windows):
    parts=[]
    for i in range(0,len(windows),64):
        tensor=torch.from_numpy(windows[i:i+64].transpose(0,2,1)[:,:,:,None].copy()).cuda()
        parts.append(model(tensor).cpu().numpy()[:,:,:,0].transpose(0,2,1))
    return np.concatenate(parts)


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
    print('Verifying data integrity',flush=True);assert digest(config.TRAIN_FILE)==TRAIN_SHA
    print('Loading training and calibration only: before July 2002',flush=True)
    raw,y,times=load_period(config.TRAIN_FILE,FEATURES,None,'2002-07-01')
    months=pd.to_datetime(times,utc=True).strftime('%Y-%m').to_numpy()
    starts=np.arange(0,len(raw)-256+1,32);c=np.r_[0,np.cumsum(y!=0)]
    valid=starts[(c[starts+256]==c[starts])&(months[starts]==months[starts+255])]
    sample=[]
    for month in np.unique(months):
        v=valid[months[valid]==month];assert len(v)>0
        sample.extend(v[np.linspace(0,len(v)-1,min(64,len(v)),dtype=int)])
    sample=np.asarray(sample);sm=months[sample];ix=sample[:,None]+np.arange(256)[None,:]
    train=sm<'2002-01';cal=~train
    a.output.mkdir(parents=True);np.save(a.output/'sample_starts.npy',sample)
    rows=[];sweeps=[];refs=[]
    for arm,root in roots.items():
        for seed in (42,43,44):
            folder=root/'2002'/f'seed{seed}';manifest=json.loads((folder/'complete.json').read_text())
            assert all(digest(folder/f)==h for f,h in manifest.items())
            dec=json.loads((folder/'frozen_decisions.json').read_text())['calibration_selected']
            cp=folder/f"epoch_{dec['epoch']}"/'model.pt';hashes[str(cp)]=digest(cp)
            saved=torch.load(cp,map_location='cpu',weights_only=False)
            scaler=RobustChannelScaler();scaler.median_=saved['scaler_median'];scaler.iqr_=saved['scaler_iqr']
            windows=scaler.transform(raw)[ix]
            reference=np.median(windows[train].mean(axis=1),axis=0)
            refs.append(dict(arm=arm,seed=seed,**dict(zip(FEATURES,reference.tolist()))))
            model=MultivariateAE(8).cuda();model.load_state_dict(saved['model_state_dict']);model.eval()
            baseline=reconstruct(model,windows)
            for name,ch in [('original',None),('recenter_ch47',7),('recenter_ch40',0)]:
                changed=windows if ch is None else recenter_channel(windows,ch,reference[ch])
                if ch is not None:
                    others=[j for j in range(8) if j!=ch]
                    np.testing.assert_array_equal(changed[:,:,others],windows[:,:,others])
                    np.testing.assert_allclose(np.diff(changed[:,:,ch],axis=1),np.diff(windows[:,:,ch],axis=1),atol=2e-6)
                reconstruction=baseline if ch is None else reconstruct(model,changed)
                mse=np.mean((changed-reconstruction)**2,axis=1)
                output_delta=np.mean((reconstruction-baseline)**2,axis=1)
                for month in np.unique(sm):
                    keep=sm==month
                    for j,channel in enumerate(FEATURES):
                        rows.append(dict(arm=arm,seed=seed,month=month,intervention=name,channel=channel,
                            mean_mse=float(mse[keep,j].mean()),median_mse=float(np.median(mse[keep,j])),
                            output_change_mse=float(output_delta[keep,j].mean()),
                            mean_input=float(changed[keep,:,j].mean()),mean_output=float(reconstruction[keep,:,j].mean())))
            # Controlled dose sweep on the same nominal training windows.
            base=windows[train][np.linspace(0,train.sum()-1,128,dtype=int)]
            for level in (0.,1.,2.,3.,4.):
                changed=recenter_channel(base,7,reference[7]+level)
                rec=reconstruct(model,changed);mse=np.mean((changed-rec)**2,axis=1)
                sweeps.append(dict(arm=arm,seed=seed,added_level=level,
                    other_channels_mean_mse=float(mse[:,:7].mean()),ch47_mean_mse=float(mse[:,7].mean())))
            pd.DataFrame(rows).to_csv(a.output/'monthly_interventions.csv',index=False)
            pd.DataFrame(sweeps).to_csv(a.output/'training_level_sweep.csv',index=False)
            print(arm,seed,'complete',flush=True)
    pd.DataFrame(refs).to_csv(a.output/'training_references.csv',index=False)
    assert digest(config.TRAIN_FILE)==TRAIN_SHA
    assert all(digest(f)==h for f,h in hashes.items())
    (a.output/'verification.json').write_text(json.dumps(dict(status='complete',train_sha256=TRAIN_SHA,
        sources=hashes,period='2000-01-01 <= t < 2002-07-01',windows_per_month=64,
        reference='Median of per-window scaled means over fixed nominal training sample only',
        limitations='Synthetic interventions on model inputs; not real-data detection scores or operational preprocessing; no clipping after intervention; no validation or outer assessment loaded'),indent=2)+'\n')


if __name__=='__main__':main()
