"""Frozen score-evidence postprocessing; no training and no assessment selection."""
import fcntl,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from esa_thesis import config
from esa_thesis.calibration_audit import digest
from esa_thesis.full_short_no_roll import verify_frozen,report_predictions
from esa_thesis.full_similarity_mosad import write_json,save_npy
from esa_thesis.masked_pilot import assessment_scopes
from esa_thesis.thresholds import postprocess
from esa_thesis.development_recovery import check_protocol
R=Path(__file__).resolve().parents[2]
SRC=R/'results_longrun/wandb_finetuning_20261024_seed42_v1'
CAL=R/'results_longrun/temporal_postprocessing_20261028_v1'
OUT=R/'results_longrun/alarm_evidence_20261028_v1'
DURS=(1,32,48,64,96,128,256,512)
FAMILIES=('strength_duration','hysteresis','duration')
def read(p):return json.loads(p.read_text())
def ids(ev,p):
 s,e,c=ev.prediction_intervals(p)
 return {i for i,xs in ev.by_id.items() if any(ev._overlaps(s,e,c,a,b).any() for a,b in xs)}
def segments(p):
 d=np.diff(np.r_[0,p,0].astype(np.int8));return list(zip(np.flatnonzero(d==1),np.flatnonzero(d==-1)))
def candidates(family):
 out=[dict(family=family,original=True)]
 if family=='strength_duration':out += [dict(family=family,original=False,duration=d,peak=q) for d in DURS for q in (1.1,1.25,1.5,2,3,5)]
 elif family=='hysteresis':out += [dict(family=family,original=False,low=l,high=h) for l in (.5,.75,1) for h in (1,1.1,1.25,1.5,2)]
 elif family=='duration':out += [dict(family=family,original=False,duration=d) for d in DURS]
 else:raise ValueError(family)
 return [dict(index=i,**r) for i,r in enumerate(out)]
def transform(score,threshold,rule,baseline_rule):
 if not np.isfinite(score).all() or threshold<=0:raise ValueError('Invalid score/threshold')
 if rule['original']:return postprocess(score>threshold,int(baseline_rule['merge_gap']),int(baseline_rule['min_dur'])).astype(bool)
 family=rule['family'];p=score>threshold*(rule['low'] if family=='hysteresis' else 1)
 for a,b in segments(p):
  if family=='hysteresis':remove=not np.any(score[a:b]>threshold*rule['high'])
  elif family=='duration':remove=b-a<rule['duration']
  else:remove=(b-a<rule['duration'] and np.max(score[a:b])/threshold<rule['peak'])
  if remove:p[a:b]=False
 return p

def choose(rows):
 eligible=[r for r in rows if r['feasible']]
 if not eligible:raise ValueError('No feasible fallback')
 return min(eligible,key=lambda r:(-r['local_f05'],r['false_seconds'],not r['rule']['original'],r['rule']['index']))
def verify_inputs():
 verify_frozen(R)
 prior=read(SRC/'protocol.json')
 for n,h in prior['source_sha256'].items():assert digest(R/'esa_thesis'/n)==h,n
 for group in ['annotations','reference']:
  for n,h in prior[group].items():assert digest(R/n)==h,n
 for folder in (SRC,CAL):
  for n,h in read(folder/'complete.json')['files'].items():assert digest(folder/n)==h,n
 return prior

def main():
 lock=open(R/'.alarm-evidence.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 prior=verify_inputs();torch.set_num_threads(2)
 plan=dict(source_receipt=digest(SRC/'complete.json'),calibration_receipt=digest(CAL/'complete.json'),selection=digest(SRC/'selection_complete.json'),runner=digest(Path(__file__)),protocol=digest(R/'docs/research/alarm-evidence-protocol-2026-10-28.md'),source=prior['source_sha256'],runtime=dict(torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__),settings={k:str(v) for k,v in vars(config).items() if k.isupper()},candidates={f:candidates(f) for f in FAMILIES})
 check_protocol(OUT,plan,(OUT/'protocol.json').exists())
 if (OUT/'complete.json').exists():
  for n,h in read(OUT/'complete.json')['files'].items():assert digest(OUT/n)==h
  print('Already complete');return
 ct=pd.date_range('2006-01-01','2007-01-01',freq='30s',inclusive='left').asi8
 evs=assessment_scopes(ct);local=evs['input_channels_40_47'];frozen={};diagnostics=[]
 for method,s in read(SRC/'selection_complete.json').items():
  f=OUT/method;f.mkdir(exist_ok=True);score=np.load(CAL/method/'calibration_score.npy');rule=s['rule'];threshold=rule['threshold']
  baseline=transform(score,threshold,candidates('duration')[0],rule);target=ids(local,baseline)
  assert baseline.mean()<=.01
  for a,b in segments(baseline):
   p=np.zeros(len(score),bool);p[a:b]=True;metrics=local.score(p)
   diagnostics.append(dict(method=method,start_sample=int(a),end_sample=int(b),duration_minutes=(b-a)/2,peak_ratio=float(score[a:b].max()/threshold),mean_excess_ratio=float(np.maximum(score[a:b]/threshold-1,0).mean()),local_ids=','.join(sorted(ids(local,p))),nominal_false_seconds=metrics['false_positive_seconds']))
  choices={}
  for family in FAMILIES:
   rows=[]
   for c in candidates(family):
    p=transform(score,threshold,c,rule);a=local.score(p);b=evs['system_all_channels'].score(p)
    rows.append(dict(rule=c,local_f05=a['EW_F_0.50'],system_f05=b['EW_F_0.50'],false_seconds=a['false_positive_seconds'],local_detected=a['TPe'],alarm_fraction=float(p.mean()),segments=a['num_pred_events'],feasible=bool(target.issubset(ids(local,p)) and p.mean()<=.01)))
   write_json(f/(family+'_candidates.json'),rows);choices[family]=choose(rows)
   print(method,family,'selected',choices[family],flush=True)
  frozen[method]=dict(source_folder=s['folder'],threshold=threshold,baseline_rule=rule,calibration_sha256=digest(CAL/method/'calibration_score.npy'),choices=choices)
 pd.DataFrame(diagnostics).to_csv(OUT/'calibration_segments.csv',index=False)
 if (OUT/'selection_complete.json').exists():assert read(OUT/'selection_complete.json')==frozen
 else:write_json(OUT/'selection_complete.json',frozen)
 print('All nine selections frozen; opening assessment scores',flush=True)
 times=pd.date_range('2007-01-01','2014-01-01',freq='30s',inclusive='left').asi8;records=[];events=[];diags=[]
 for method,s in frozen.items():
  score=np.load(SRC/s['source_folder']/'testing_score.npy');original=np.load(SRC/s['source_folder']/'testing_prediction.npy').astype(bool)
  assert np.array_equal(original,transform(score,s['threshold'],candidates('duration')[0],s['baseline_rule']))
  preds={'original':original}
  for family,c in s['choices'].items():
   preds[family]=transform(score,s['threshold'],c['rule'],s['baseline_rule']);save_npy(OUT/method/(family+'_prediction.npy'),preds[family])
  for variant,p in preds.items():
   annual,pooled=report_predictions(times,p);frame=pd.concat([annual,pooled]);frame.insert(0,'variant',variant);frame.insert(0,'method',method);records.append(frame)
   diags.append(dict(method=method,variant=variant,segments=len(segments(p)),alarm_samples=int(p.sum()),sample_grid_alarm_seconds=int(p.sum())*30))
  for period in [str(y) for y in range(2007,2014)]+['2007-2013']:
   keep=np.ones(len(times),bool) if '-' in period else ((times>=pd.Timestamp(period+'-01-01').value)&(times<pd.Timestamp(str(int(period)+1)+'-01-01').value))
   for scope,ev in assessment_scopes(times[keep]).items():
    baseids=ids(ev,original[keep])
    for family in FAMILIES:
     selected=ids(ev,preds[family][keep])
     for aid in ev.by_id:events.append(dict(method=method,family=family,period=period,scope=scope,event_id=aid,original_detected=aid in baseids,selected_detected=aid in selected,gained=aid in selected-baseids,lost=aid in baseids-selected))
  print(method,'assessment complete',flush=True)
 pd.concat(records).to_csv(OUT/'all_results.csv',index=False);pd.DataFrame(events).to_csv(OUT/'event_ids.csv',index=False);pd.DataFrame(diags).to_csv(OUT/'segment_diagnostics.csv',index=False)
 verify_inputs()
 write_json(OUT/'complete.json',dict(needs_independent_verification=True,files={str(p.relative_to(OUT)):digest(p) for p in OUT.rglob('*') if p.is_file() and p.name!='complete.json'}))
 print('Complete; independent verification required',flush=True)
if __name__=='__main__':main()
