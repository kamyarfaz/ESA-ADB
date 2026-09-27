"""Fixed six-window calibration inspection; no assessment or rule selection."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from esa_thesis.models import MultivariateAE
from esa_thesis.diagnose_channels import digest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('results_longrun/diagnostics/baseline_2005_epoch5_v1')
checkpoint=Path('results_longrun/development/ae_baseline_remaining_seed42/2005/model_checkpoint.pt')
out=Path('docs/research/tables/nominal-windows-2005-2026-09-26');out.mkdir(parents=True,exist_ok=False)
files=[root/'calibration_inputs.npz',root/'calibration_window_errors.npz',root/'results.json',checkpoint]
before={str(p):digest(p) for p in files}
assert json.loads(files[2].read_text())['reproduction']=='passed within 1e-6'
with np.load(files[0]) as a:x=a['scaled'];raw=a['values'];times=a['timestamps']
with np.load(files[1]) as a:errors=a['errors'];starts=a['starts'];nominal=a['nominal'];np.testing.assert_array_equal(times,a['timestamps'])
cs=np.r_[0,np.cumsum(~nominal)];clean=cs[starts+256]==cs[starts]
months=pd.DatetimeIndex(times).month.to_numpy();choices=[]
for month in [1,2,3]:
 valid=np.flatnonzero(clean&(months[starts]==month)&(months[starts+255]==month))
 scores=errors[valid].max(axis=1)
 for kind,pos in [('highest',int(np.argmax(scores))),('typical',int(np.argmin(abs(scores-np.median(scores)))))]:
  i=int(valid[pos]);choices.append(dict(month=month,kind=kind,index=i,start=int(starts[i])))
(out/'selection.json').write_text(json.dumps(dict(policy='Within each calibration month: highest max-channel MSE and nearest median, earliest tie; wholly nominal256sample windows only',windows=choices,limitations='Six selected examples, not prevalence estimates or physical explanations. No assessment access.'),indent=2)+'\n')
torch.set_num_threads(1);saved=torch.load(checkpoint,map_location='cpu',weights_only=False);model=MultivariateAE(8);model.load_state_dict(saved['model_state_dict']);model.eval()
inputs=torch.from_numpy(np.stack([x[c['start']:c['start']+256].T[:,:,None] for c in choices]))
with torch.no_grad():prediction=model(inputs).squeeze(-1).numpy();mse=(prediction-inputs.squeeze(-1).numpy())**2;mse=mse.mean(axis=2)
np.testing.assert_allclose(mse,np.stack([errors[c['index']] for c in choices]),rtol=1e-5,atol=1e-5)
rows=[]
for k,c in enumerate(choices):
 start=c['start'];v=x[start:start+256];rv=raw[start:start+256]
 for j in range(8):
  rows.append(dict(**c,channel=f'channel_{40+j}',timestamp=str(pd.Timestamp(times[start])),cached_mse=float(errors[c['index'],j]),repeated_mse=float(mse[k,j]),raw_min=float(rv[:,j].min()),raw_max=float(rv[:,j].max()),scaled_min=float(v[:,j].min()),scaled_max=float(v[:,j].max()),reconstruction_min=float(prediction[k,j].min()),reconstruction_max=float(prediction[k,j].max()),largest_scaled_step=float(np.max(abs(np.diff(v[:,j])))),unchanged_fraction=float(np.mean(np.diff(rv[:,j])==0))))
for month in [1,2,3]:
 indices=[k for k,c in enumerate(choices) if c['month']==month]
 fig,axes=plt.subplots(8,2,figsize=(13,15),constrained_layout=True)
 for col,k in enumerate(indices):
  c=choices[k];start=c['start'];elapsed=np.arange(256)*.5
  for j in range(8):
   ax=axes[j,col];ax.plot(elapsed,x[start:start+256,j],label='Observed',lw=1);ax.plot(elapsed,prediction[k,j],label='Reconstructed',lw=1,alpha=.8)
   ax.set_ylabel(f'Channel {40+j}');ax.set_title(f"{c['kind']}: MSE {mse[k,j]:.3g}",fontsize=9)
   if j==0:ax.legend(fontsize=8)
   if j==7:ax.set_xlabel('Minutes from window start')
 fig.suptitle(f'2005-{month:02d}: wholly nominal windows, training-scaled telemetry\nLeft: highest error; right: typical error. Each panel has its own y-axis scale.')
 fig.savefig(out/f'month_{month:02d}.png',dpi=110);plt.close(fig)
pd.DataFrame(rows).to_csv(out/'window_channels.csv',index=False)
assert before=={str(p):digest(p) for p in files}
(out/'verification.json').write_text(json.dumps(dict(status='complete',inputs=before,reconstruction_check='Cached per-channel window MSE reproduced rtol1e-5 atol1e-5; no threshold or event metric modified'),indent=2)+'\n')
df=pd.DataFrame(rows)
print(df.loc[df.groupby(['month','kind']).cached_mse.idxmax(),['month','kind','timestamp','channel','cached_mse','scaled_min','scaled_max','reconstruction_min','reconstruction_max','largest_scaled_step']].to_string(index=False))
