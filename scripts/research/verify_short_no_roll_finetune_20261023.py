"""Independent receipt, selection, weight, prediction and event-metric replay."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from esa_thesis.evaluation.esa import EventEvaluator
from esa_thesis.thresholds import postprocess,select_rows

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'results_longrun/short_no_roll_finetune_20261023_seed42_v1'
OLD=ROOT/'results_longrun/short_no_roll_full_20261014_seed42_v1'
OUT=ROOT/'docs/research/tables/short-no-roll-finetune-2026-10-23'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())
for n,h in read(RUN/'complete.json')['files'].items():assert sha(RUN/n)==h,n
plan=read(RUN/'protocol.json')
assert plan==read(RUN/'resume_protocol.json')
for n,h in plan['source_sha256'].items():assert sha(ROOT/'esa_thesis'/n)==h,n
assert sha(ROOT/'docs/research/short-no-roll-finetune-protocol-2026-10-23.md')==plan['protocol_sha256']
for n,h in plan['reference'].items():assert sha(OLD/n)==h,n
for n,h in plan['annotations'].items():assert sha(ROOT/n)==h,n
base=ROOT/'data/frozen/mission1-reference-v1'
assert sha(base/'manifest.json')==plan['manifest_sha']
for n,r in read(base/'manifest.json')['files'].items():assert sha(base/n)==r['sha256'],n
folder=RUN/'finetuned'
for n,h in read(folder/'trained.json')['files'].items():assert sha(folder/n)==h,n
hist=pd.read_csv(folder/'history.csv')
assert hist.epoch.tolist()==list(range(1,11)) and np.isfinite(hist.loss).all()
initial=read(OLD/'full_period/frozen_rule.json')
candidates=[dict(epoch=0,calibration_f05=initial['rule']['val_esa_f05'])]
for epoch in [5,10]:
 grid=pd.read_csv(folder/f'calibration_candidates_epoch{epoch}.csv')
 choice=select_rows(grid).iloc[0]
 val=float(hist.set_index('epoch').loc[epoch,'calibration_f05'])
 assert np.isclose(choice.val_esa_f05,val,rtol=1e-14)
 candidates.append(dict(epoch=epoch,calibration_f05=val))
selected=max(candidates,key=lambda x:x['calibration_f05'])['epoch']
frozen=read(folder/'frozen_rule.json')
assert frozen['epoch']==selected==0
assert frozen['rule']==initial['rule']
a=torch.load(OLD/'full_period/model_checkpoint.pt',map_location='cpu',weights_only=False)
b=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
assert a['features']==b['features']==[f'channel_{i}' for i in range(40,48)]
for k,v in a['model_state_dict'].items():assert torch.equal(v,b['model_state_dict'][k]),k
for k in ['scaler_median','scaler_iqr']:assert np.array_equal(a[k],b[k])
last=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
assert last['epoch']==11 and last['batch']==0 and last['best']['epoch']==0
score=np.load(folder/'testing_score.npy');pred=np.load(folder/'testing_prediction.npy')
r=frozen['rule']
assert np.array_equal(pred,postprocess(score>r['threshold'],int(r['merge_gap']),int(r['min_dur'])))
original=np.load(OLD/'full_period/assessment_prediction.npy')
assert np.array_equal(pred,original)
labels=pd.read_csv(ROOT/'data/ESA-Mission1/labels.csv').merge(pd.read_csv(ROOT/'data/ESA-Mission1/anomaly_types.csv'),on='ID',validate='many_to_one')
local=labels.copy();local.loc[~local.Channel.isin(b['features']),'Category']='Out-of-scope (neutral)'
anns={'system_all_channels':labels,'input_channels_40_47':local}
saved=pd.read_csv(RUN/'all_results.csv',dtype={'period':str}).set_index(['method','period','scope'])
assert len(saved)==32 and saved.index.is_unique
times=pd.date_range('2007-01-01','2014-01-01',freq='30s',inclusive='left').asi8
assert len(pred)==len(times)
rows=[];events=[]
for period in [str(y) for y in range(2007,2014)]+['2007-2013']:
 keep=np.ones(len(times),bool) if '-' in period else ((times>=pd.Timestamp(period+'-01-01').value)&(times<pd.Timestamp(str(int(period)+1)+'-01-01').value))
 for scope,ann in anns.items():
  ev=EventEvaluator(times[keep],ann)
  for method,p in [('original',original),('finetuned',pred)]:
   metrics=ev.score(p[keep])
   for k,v in metrics.items():
    actual=saved.loc[(method,period,scope),k]
    assert v==actual if isinstance(v,str) else np.isclose(v,actual,rtol=1e-11,atol=1e-9),(method,period,scope,k)
   rows.append(dict(method=method,period=period,scope=scope,**metrics))
  if period=='2007-2013':
   starts,ends,closed=ev.prediction_intervals(pred)
   for aid,intervals in ev.by_id.items():
    hit=any(ev._overlaps(starts,ends,closed,x,y).any() for x,y in intervals)
    events.append(dict(scope=scope,event_id=aid,original_detected=hit,selected_detected=hit,gained=False,lost=False))
OUT.mkdir(parents=True,exist_ok=True)
pd.DataFrame(rows).to_csv(OUT/'verified_results.csv',index=False)
pd.DataFrame(events).to_csv(OUT/'event_ids.csv',index=False)
pd.DataFrame(candidates).to_csv(OUT/'calibration_comparison.csv',index=False)
(OUT/'verification.json').write_text(json.dumps(dict(verified=True,metric_rows=32,selected_continuation_epoch=0,complete_receipt_sha256=sha(RUN/'complete.json'),checks=['Frozen data/source/protocol/reference and artifact hashes','Ten-epoch history and exact final checkpoint state','Calibration-only grid selection at epochs0,5,10','Selected weights and scaler exactly equal original','Prediction replay and exact original prediction equality','32 corrected annual/pooled metric rows recomputed','All event IDs; no gained or lost IDs'],limitations=['Single seed; reused benchmark','Unselected checkpoints not assessed']),indent=2)+'\n')
print('Verified all32 rows, original weights/predictions retained',flush=True)
