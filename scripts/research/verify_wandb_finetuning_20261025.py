"""Independent frozen receipt, calibration replay, selection and assessment audit."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch,wandb
from esa_thesis import config
from esa_thesis.evaluation.esa import EventEvaluator
from esa_thesis.thresholds import postprocess,select_rows,sweep_thresholds
from esa_thesis.development import QUANTILES
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'results_longrun/wandb_finetuning_20261024_seed42_v1'
OUT=ROOT/'docs/research/tables/wandb-finetuning-2026-10-25'
METHODS=['short_no_roll','pretrain_finetune','expanded_short_no_roll']
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def same(a,b):
 if isinstance(a,(str,bool)):assert a==b,(a,b)
 else:assert np.isclose(a,b,rtol=1e-11,atol=1e-9),(a,b)
def source(m):return ROOT/('results_longrun/short_no_roll_full_20261014_seed42_v1/full_period' if m=='short_no_roll' else 'results_longrun/combinations_full_20261021_seed42_v1/'+m)
def ids(ev,p):
 s,e,c=ev.prediction_intervals(p)
 return {i for i,xs in ev.by_id.items() if any(ev._overlaps(s,e,c,a,b).any() for a,b in xs)}
receipt=read(RUN/'complete.json')
for n,h in receipt['files'].items():assert sha(RUN/n)==h,n
plan=read(RUN/'protocol.json');assert plan==read(RUN/'resume_protocol.json')
for n,h in plan['source_sha256'].items():assert sha(ROOT/'esa_thesis'/n)==h,n
for group in ['reference','annotations']:
 for n,h in plan[group].items():assert sha(ROOT/n)==h,n
assert plan['settings']=={k:str(v) for k,v in vars(config).items() if k.isupper()}
assert plan['runtime']==dict(torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,wandb=wandb.__version__,cuda=torch.version.cuda)
assert sha(ROOT/'docs/research/wandb-finetuning-protocol-2026-10-24.md')==plan['research_protocol_sha256']
base=ROOT/'data/frozen/mission1-reference-v1'
assert sha(base/'manifest.json')==plan['manifest_sha']
for n,r in read(base/'manifest.json')['files'].items():assert sha(base/n)==r['sha256'],n
print('Frozen data, runtime, source and all receipt hashes passed',flush=True)
labels=pd.read_csv(ROOT/'data/ESA-Mission1/labels.csv').merge(pd.read_csv(ROOT/'data/ESA-Mission1/anomaly_types.csv'),on='ID',validate='many_to_one')
local=labels.copy();local.loc[~local.Channel.isin([f'channel_{i}' for i in range(40,48)]),'Category']='Out-of-scope (neutral)'
anns={'system_all_channels':labels,'input_channels_40_47':local}
ct=pd.date_range('2006-01-01','2007-01-01',freq='30s',inclusive='left').asi8
ce=EventEvaluator(ct,labels)
selections=read(RUN/'selection_complete.json');cal=[]
for method in METHODS:
 original=read(source(method)/'frozen_rule.json')['rule'];choices=[]
 for i,lr in enumerate([3e-6,1e-5,3e-5],1):
  f=RUN/f'{method}_lr{i}';hist=pd.read_csv(f/'history.csv').set_index('epoch')
  assert hist.index.tolist()==list(range(1,11)) and np.isfinite(hist.loss).all()
  for n,h in read(f/'trained.json')['files'].items():assert sha(f/n)==h
  candidates=[(0,original)]
  for ep in [5,10]:
   score=np.load(f/f'calibration_score_epoch{ep}.npy');assert len(score)==len(ct)
   grid=sweep_thresholds(score,evaluator=ce,threshold_values=np.percentile(score,QUANTILES),merge_gaps=[0,16,64],min_durs=[1,8,32])
   saved=pd.read_csv(f/f'calibration_candidates_epoch{ep}.csv')
   pd.testing.assert_frame_equal(grid,saved,check_dtype=False,rtol=1e-11,atol=1e-9)
   rule=select_rows(grid).iloc[0].to_dict();same(rule['val_esa_f05'],hist.loc[ep,'calibration_f05'])
   pred=postprocess(score>rule['threshold'],int(rule['merge_gap']),int(rule['min_dur']))
   for scope,ann in anns.items():
    metrics=EventEvaluator(ct,ann).score(pred)
    for k in ['EW_F_0.50','EW_precision','EW_recall','nominal_false_alarm_rate','TPe']:same(metrics[k],hist.loc[ep,'calibration/'+scope+'/'+k])
   candidates.append((ep,rule))
  best=max(candidates,key=lambda x:x[1]['val_esa_f05']);fr=read(f/'frozen_rule.json');assert fr['epoch']==best[0]
  for k,v in best[1].items():same(v,fr['rule'][k])
  chosen=torch.load(f/'model.pt',map_location='cpu',weights_only=False)
  ref=(source(method)/('model_checkpoint.pt' if method=='short_no_roll' else 'model.pt')) if best[0]==0 else f/f'candidate_epoch{best[0]}.pt'
  model=torch.load(ref,map_location='cpu',weights_only=False)
  assert chosen['features']==model['features']
  for k,v in model['model_state_dict'].items():assert torch.equal(v,chosen['model_state_dict'][k])
  for k in ['scaler_median','scaler_iqr']:assert np.array_equal(model[k],chosen[k])
  last=torch.load(f/'last.pt',map_location='cpu',weights_only=False);assert last['epoch']==11 and last['batch']==0 and last['best']['epoch']==best[0]
  choices.append((best[1]['val_esa_f05'],best[0],lr,f.name))
  for ep,r in candidates:cal.append(dict(method=method,learning_rate=lr,epoch=ep,calibration_system_f05=r['val_esa_f05'],run_selected=ep==best[0]))
  print(f.name,'calibration grid replay and weights verified',flush=True)
 winner=sorted(choices,key=lambda x:(-x[0],x[1],x[2]))[0];s=selections[method]
 assert (s['rule']['val_esa_f05'],s['epoch'],s['lr'],s['folder'])==winner
 assert sha(RUN/s['folder']/'model.pt')==s['model_sha256']
 assert max((RUN/x[3]/'trained.json').stat().st_mtime for x in choices)<(RUN/'selection_complete.json').stat().st_mtime
saved=pd.read_csv(RUN/'all_results.csv',dtype={'period':str}).set_index(['method','variant','period','scope']);assert len(saved)==96 and saved.index.is_unique
rows=[];events=[]
times=pd.date_range('2007-01-01','2014-01-01',freq='30s',inclusive='left').asi8
for method,s in selections.items():
 f=RUN/s['folder'];r=s['rule'];score=np.load(f/'testing_score.npy');pred=np.load(f/'testing_prediction.npy')
 assert np.array_equal(pred,postprocess(score>r['threshold'],int(r['merge_gap']),int(r['min_dur'])))
 assert (f/'testing_score.npy').stat().st_mtime>(RUN/'selection_complete.json').stat().st_mtime
 original=np.load(source(method)/('assessment_prediction.npy' if method=='short_no_roll' else 'testing_prediction.npy'))
 if s['epoch']==0:assert np.array_equal(pred,original)
 prior=pd.read_csv(source(method).parent/'all_results.csv',dtype={'period':str})
 if method!='short_no_roll':prior=prior[prior.method==method]
 prior=prior.set_index(['period','scope'])
 for period in [str(y) for y in range(2007,2014)]+['2007-2013']:
  keep=np.ones(len(times),bool) if '-' in period else ((times>=pd.Timestamp(period+'-01-01').value)&(times<pd.Timestamp(str(int(period)+1)+'-01-01').value))
  for scope,ann in anns.items():
   ev=EventEvaluator(times[keep],ann)
   for variant,p in [('original',original),('selected',pred)]:
    metrics=ev.score(p[keep])
    for k,v in metrics.items():
     same(v,saved.loc[(method,variant,period,scope),k])
     if variant=='original':same(v,prior.loc[(period,scope),k])
    rows.append(dict(method=method,variant=variant,period=period,scope=scope,**metrics))
   a,b=ids(ev,original[keep]),ids(ev,pred[keep])
   for aid in ev.by_id:events.append(dict(method=method,period=period,scope=scope,event_id=aid,original_detected=aid in a,selected_detected=aid in b,gained=aid in b-a,lost=aid in a-b))
 print(method,'all assessment metrics verified',flush=True)
OUT.mkdir(parents=True,exist_ok=True)
pd.DataFrame(rows).to_csv(OUT/'verified_results.csv',index=False)
pd.DataFrame(events).to_csv(OUT/'event_ids.csv',index=False)
pd.DataFrame(cal).to_csv(OUT/'calibration_comparison.csv',index=False)
(OUT/'verification.json').write_text(json.dumps(dict(verified=True,metric_rows=96,calibration_grids_replayed=18,complete_receipt_sha256=sha(RUN/'complete.json'),checks=['All completion/training hashes; frozen dataset/reference/source/runtime/settings','Nine ten-epoch histories and final resumable states','18 full calibration grids replayed from saved scores and raw annotations','Within-run and cross-rate calibration-only selection; weight/scaler equality','Assessment after selection freeze; exact binary prediction replay','96 corrected annual/pooled rows; originals match prior results','Annual and pooled detected/missed/gained/lost event IDs'],limitations=['Single seed; reused benchmark; short-no-roll LR1e-5 repeated','Replay uses stored scores; no independent neural inference rerun']),indent=2)+'\n')
print('VERIFIED',flush=True)
