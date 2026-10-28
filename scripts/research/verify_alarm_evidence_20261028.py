"""Independent segment implementation, calibration selection and metric replay."""
import json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from esa_thesis.evaluation.esa import EventEvaluator
R=Path(__file__).resolve().parents[2];RUN=R/'results_longrun/alarm_evidence_20261028_v1';SRC=R/'results_longrun/wandb_finetuning_20261024_seed42_v1';CAL=R/'results_longrun/temporal_postprocessing_20261028_v1';OUT=R/'docs/research/tables/alarm-evidence-2026-10-28'
FAMILIES=('strength_duration','hysteresis','duration')
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def same(a,b):
 if isinstance(a,(str,bool)):assert a==b,(a,b)
 else:assert np.isclose(a,b,rtol=1e-11,atol=1e-9),(a,b)
def ids(ev,p):
 s,e,c=ev.prediction_intervals(p)
 return {aid for aid,xs in ev.by_id.items() if any(ev._overlaps(s,e,c,a,b).any() for a,b in xs)}
def replay(scores,t,r):
 # All fixed baseline rules are gap=0/duration=1 (asserted below).
 low=r.get('low',1) if not r['original'] else 1
 idx=np.flatnonzero(scores>t*low);out=np.zeros(len(scores),bool)
 for group in np.split(idx,np.flatnonzero(np.diff(idx)>1)+1):
  if not len(group):continue
  a,b=int(group[0]),int(group[-1])+1;keep=True
  if not r['original']:
   if r['family']=='hysteresis':keep=float(scores[a:b].max())>t*r['high']
   elif r['family']=='duration':keep=(b-a)>=r['duration']
   else:keep=(b-a)>=r['duration'] or float(scores[a:b].max())/t>=r['peak']
  if keep:out[a:b]=True
 return out

def main():
 receipt=read(RUN/'complete.json');plan=read(RUN/'protocol.json');assert plan==read(RUN/'resume_protocol.json')
 for n,h in receipt['files'].items():assert sha(RUN/n)==h,n
 assert sha(R/'scripts/research/run_alarm_evidence_20261028.py')==plan['runner']
 assert sha(R/'docs/research/alarm-evidence-protocol-2026-10-28.md')==plan['protocol']
 assert sha(SRC/'complete.json')==plan['source_receipt'] and sha(CAL/'complete.json')==plan['calibration_receipt'] and sha(SRC/'selection_complete.json')==plan['selection']
 for n,h in plan['source'].items():assert sha(R/'esa_thesis'/n)==h
 prior=read(SRC/'protocol.json');base=R/'data/frozen/mission1-reference-v1';assert sha(base/'manifest.json')==prior['manifest_sha']
 for n,x in read(base/'manifest.json')['files'].items():assert sha(base/n)==x['sha256']
 for group in ['annotations','reference']:
  for n,h in prior[group].items():assert sha(R/n)==h
 for folder in (SRC,CAL):
  for n,h in read(folder/'complete.json')['files'].items():assert sha(folder/n)==h,n
 labels=pd.read_csv(R/'data/ESA-Mission1/labels.csv').merge(pd.read_csv(R/'data/ESA-Mission1/anomaly_types.csv'),on='ID',validate='many_to_one')
 local=labels.copy();local.loc[~local.Channel.isin([f'channel_{i}' for i in range(40,48)]),'Category']='Out-of-scope (neutral)'
 anns={'system_all_channels':labels,'input_channels_40_47':local}
 ct=pd.date_range('2006-01-01','2007-01-01',freq='30s',inclusive='left').asi8;times=pd.date_range('2007-01-01','2014-01-01',freq='30s',inclusive='left').asi8
 freeze=read(RUN/'selection_complete.json');saved=pd.read_csv(RUN/'all_results.csv',dtype={'period':str}).set_index(['method','variant','period','scope']);assert len(saved)==192 and saved.index.is_unique
 old=pd.read_csv(SRC/'all_results.csv',dtype={'period':str});old=old[old.variant=='selected'].set_index(['method','period','scope'])
 rows=[];events=[];diags=[];count=0
 for method,s in freeze.items():
  assert int(s['baseline_rule']['merge_gap'])==0 and int(s['baseline_rule']['min_dur'])==1
  score=np.load(CAL/method/'calibration_score.npy');assert sha(CAL/method/'calibration_score.npy')==s['calibration_sha256']
  ev=EventEvaluator(ct,local);system=EventEvaluator(ct,labels);target=ids(ev,score>s['threshold'])
  for family in FAMILIES:
   grid=read(RUN/method/(family+'_candidates.json'));assert [g['rule'] for g in grid]==plan['candidates'][family]
   assert len(grid)=={'strength_duration':49,'hysteresis':16,'duration':9}[family]
   for g in grid:
    p=replay(score,s['threshold'],g['rule']);a=ev.score(p);b=system.score(p)
    for k,v in dict(local_f05=a['EW_F_0.50'],system_f05=b['EW_F_0.50'],false_seconds=a['false_positive_seconds'],local_detected=a['TPe'],alarm_fraction=float(p.mean()),segments=a['num_pred_events']).items():same(g[k],v)
    assert g['feasible']==bool(target.issubset(ids(ev,p)) and p.mean()<=.01);count+=1
   winner=sorted([g for g in grid if g['feasible']],key=lambda g:(-g['local_f05'],g['false_seconds'],not g['rule']['original'],g['rule']['index']))[0]
   assert winner==s['choices'][family]
  score=np.load(SRC/s['source_folder']/'testing_score.npy');original=np.load(SRC/s['source_folder']/'testing_prediction.npy').astype(bool);assert np.array_equal(original,score>s['threshold'])
  preds={'original':original}
  for family in FAMILIES:
   p=np.load(RUN/method/(family+'_prediction.npy'));assert np.array_equal(p,replay(score,s['threshold'],s['choices'][family]['rule']))
   assert (RUN/'selection_complete.json').stat().st_mtime<(RUN/method/(family+'_prediction.npy')).stat().st_mtime
   preds[family]=p
  for v,p in preds.items():diags.append(dict(method=method,variant=v,segments=int(np.sum(np.diff(np.r_[0,p,0].astype(np.int8))==1)),alarm_samples=int(p.sum()),sample_grid_alarm_seconds=int(p.sum())*30))
  for period in [str(y) for y in range(2007,2014)]+['2007-2013']:
   keep=np.ones(len(times),bool) if '-' in period else ((times>=pd.Timestamp(period+'-01-01').value)&(times<pd.Timestamp(str(int(period)+1)+'-01-01').value))
   for scope,ann in anns.items():
    e=EventEvaluator(times[keep],ann);baseids=ids(e,original[keep])
    for v,p in preds.items():
     metrics=e.score(p[keep])
     for k,value in metrics.items():
      same(value,saved.loc[(method,v,period,scope),k])
      if v=='original':same(value,old.loc[(method,period,scope),k])
     rows.append(dict(method=method,variant=v,period=period,scope=scope,**metrics))
     if v!='original':
      selected=ids(e,p[keep])
      for aid in e.by_id:events.append(dict(method=method,family=v,period=period,scope=scope,event_id=aid,original_detected=aid in baseids,selected_detected=aid in selected,gained=aid in selected-baseids,lost=aid in baseids-selected))
  print(method,'verified',flush=True)
 assert count==222
 pd.testing.assert_frame_equal(pd.DataFrame(events),pd.read_csv(RUN/'event_ids.csv',dtype={'period':str}))
 pd.testing.assert_frame_equal(pd.DataFrame(diags),pd.read_csv(RUN/'segment_diagnostics.csv'))
 OUT.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(OUT/'verified_results.csv',index=False);pd.DataFrame(events).to_csv(OUT/'event_ids.csv',index=False);pd.DataFrame(diags).to_csv(OUT/'segment_diagnostics.csv',index=False)
 (OUT/'verification.json').write_text(json.dumps(dict(verified=True,calibration_candidates=count,metric_rows=len(rows),complete_receipt_sha256=sha(RUN/'complete.json'),checks=['Frozen data/source/prior receipts','Separately implemented transforms','All 222 calibration metrics and nine selections','192 assessment metrics and unchanged control parity','Exact prediction and event-ID replay','Assessment artifacts after joint freeze'],limitations=['Single seed; reused benchmark; saved-score replay; offline processing']),indent=2)+'\n')
 print('VERIFIED',flush=True)
if __name__=='__main__':main()
