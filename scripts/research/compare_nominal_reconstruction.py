"""Compare two existing checkpoints on six previously selected nominal windows."""
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
out=Path('docs/research/tables/nominal-reconstruction-comparison-2026-09-26');out.mkdir(parents=True,exist_ok=False)
selection=Path('docs/research/tables/nominal-windows-2005-2026-09-26/selection.json')
cache=Path('results_longrun/diagnostics/baseline_2005_epoch5_v1/calibration_inputs.npz')
roots={n:Path('results_longrun/development')/d for n,d in [('pseudo','ae_baseline_remaining_seed42'),('reconstruction','ae_reconstruction_seed42')]}
files=[selection,cache]
for r in roots.values():files.extend([r/'protocol.json',r/'2005/model_checkpoint.pt',r/'2005/frozen_rule.json'])
before={str(p):digest(p) for p in files}
choices=json.loads(selection.read_text())['windows']
with np.load(cache) as a:x=a['scaled'];raw=a['values'];times=a['timestamps']
protocols={n:json.loads((r/'protocol.json').read_text()) for n,r in roots.items()}
saved={n:torch.load(r/'2005/model_checkpoint.pt',map_location='cpu',weights_only=False) for n,r in roots.items()}
for field in ['features','scaler_median','scaler_iqr']:np.testing.assert_array_equal(saved['pseudo'][field],saved['reconstruction'][field])
for field in ['epochs','batch_size','seed','precision','scoring','checkpoint_selection']:assert protocols['pseudo'][field]==protocols['reconstruction'][field]
assert protocols['pseudo']['folds']['2005']==protocols['reconstruction']['folds']['2005']
assert protocols['pseudo']['spec']['pseudo'] and not protocols['reconstruction']['spec']['pseudo']
inputs=torch.from_numpy(np.stack([x[c['start']:c['start']+256].T[:,:,None] for c in choices]))
torch.set_num_threads(1);preds={};rows=[]
for name in roots:
 model=MultivariateAE(8);model.load_state_dict(saved[name]['model_state_dict']);model.eval()
 with torch.no_grad():preds[name]=model(inputs).squeeze(-1).numpy()
 errors=(preds[name]-inputs.squeeze(-1).numpy())**2
 for k,c in enumerate(choices):
  for j in range(8):rows.append(dict(model=name,month=c['month'],kind=c['kind'],start=str(pd.Timestamp(times[c['start']])),channel=f'channel_{40+j}',mse=float(errors[k,j].mean()),observed_min=float(inputs[k,j].min()),observed_max=float(inputs[k,j].max()),reconstruction_min=float(preds[name][k,j].min()),reconstruction_max=float(preds[name][k,j].max())))
df=pd.DataFrame(rows);df.to_csv(out/'channel_errors.csv',index=False)
summary=df.groupby(['month','kind','model'],as_index=False).mse.max().pivot(index=['month','kind'],columns='model',values='mse').reset_index();summary.to_csv(out/'max_channel_error.csv',index=False)
fig,axes=plt.subplots(3,1,figsize=(11,9),constrained_layout=True)
for ax,month in zip(axes,[1,2,3]):
 k=next(k for k,c in enumerate(choices) if c['month']==month and c['kind']=='highest')
 j=int(np.argmax(np.mean((preds['pseudo'][k]-inputs[k].squeeze(-1).numpy())**2,axis=1)))
 ax.plot(np.arange(256)*.5,inputs[k,j,:,0],label='Observed',color='black',lw=1.2)
 for n in preds:ax.plot(np.arange(256)*.5,preds[n][k,j],label=n,lw=1,alpha=.8)
 ax.set_title(f"2005-{month:02d}: channel {40+j}, selected highest-error nominal window");ax.set_ylabel('Training-scaled value');ax.legend(ncol=3)
axes[-1].set_xlabel('Minutes from window start');fig.savefig(out/'dominant_channels.png',dpi=140);plt.close(fig)
assert before=={str(p):digest(p) for p in files}
metadata=dict(status='complete',inputs=before,selected_epochs={n:json.loads((r/'2005/frozen_rule.json').read_text())['epoch'] for n,r in roots.items()},matching_scalers=True,matching_features=True,source_hash_equality={k:protocols['pseudo'].get('source_sha256',{}).get(k)==protocols['reconstruction'].get('source_sha256',{}).get(k) for k in ['models.py','data.py','development.py']},limitations='Six windows selected using pseudo model errors; selection favors finding its failures. Different training objectives/random-number consumption and development source hashes; not strict causal ablation. Lower reconstruction error alone does not establish better anomaly detection. No assessment access or new training.')
(out/'protocol.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(summary.to_string(index=False));print(metadata['selected_epochs'])
