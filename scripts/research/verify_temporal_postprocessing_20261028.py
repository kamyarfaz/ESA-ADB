"""Independent temporal grid, selection, predictions and metric audit."""
import json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from esa_thesis.evaluation.esa import EventEvaluator
from esa_thesis.thresholds import postprocess
R=Path(__file__).resolve().parents[2];RUN=R/'results_longrun/temporal_postprocessing_20261028_v1';SRC=R/'results_longrun/wandb_finetuning_20261024_seed42_v1';OUT=R/'docs/research/tables/temporal-postprocessing-2026-10-28'
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
def main():
 receipt=read(RUN/'complete.json')
 for n,h in receipt['files'].items():assert sha(RUN/n)==h,n
 plan=read(RUN/'protocol.json');assert plan==read(RUN/'resume_protocol.json')
 assert sha(R/'scripts/research/run_temporal_postprocessing_20261028.py')==plan['runner']
 assert sha(R/'docs/research/temporal-postprocessing-protocol-2026-10-28.md')==plan['protocol']
 assert sha(SRC/'complete.json')==plan['source_receipt'] and sha(SRC/'selection_complete.json')==plan['selection']
 for n,h in plan['source'].items():assert sha(R/'esa_thesis'/n)==h
 base=R/'data/frozen/mission1-reference-v1';prior=read(SRC/'protocol.json')
 assert sha(base/'manifest.json')==prior['manifest_sha']
 for n,x in read(base/'manifest.json')['files'].items():assert sha(base/n)==x['sha256']
 for group in ['annotations','reference']:
  for n,h in prior[group].items():assert sha(R/n)==h
 for n,h in read(SRC/'complete.json')['files'].items():assert sha(SRC/n)==h
 labels=pd.read_csv(R/'data/ESA-Mission1/labels.csv').merge(pd.read_csv(R/'data/ESA-Mission1/anomaly_types.csv'),on='ID',validate='many_to_one')
 local=labels.copy();local.loc[~local.Channel.isin([f'channel_{i}' for i in range(40,48)]),'Category']='Out-of-scope (neutral)'
 anns={'system_all_channels':labels,'input_channels_40_47':local}
 ct=pd.date_range('2006-01-01','2007-01-01',freq='30s',inclusive='left').asi8;times=pd.date_range('2007-01-01','2014-01-01',freq='30s',inclusive='left').asi8
 freeze=read(RUN/'selection_complete.json');saved=pd.read_csv(RUN/'all_results.csv',dtype={'period':str}).set_index(['method','variant','period','scope']);assert len(saved)==96 and saved.index.is_unique
 old=pd.read_csv(SRC/'all_results.csv',dtype={'period':str});old=old[old.variant=='selected'].set_index(['method','period','scope'])
 rows=[];events=[];diagnostics=[]
 for method,s in freeze.items():
  f=RUN/method;src=SRC/s['source_folder'];rule=s['baseline_rule'];score=np.load(f/'calibration_score.npy');assert len(score)==len(ct)
  assert sha(f/'calibration_score.npy')==s['calibration_sha256'] and sha(f/'calibration_candidates.csv')==s['grid_sha256']
  ev=EventEvaluator(ct,local);system=EventEvaluator(ct,labels);baseline=postprocess(score>s['threshold'],int(rule['merge_gap']),int(rule['min_dur']));target=ids(ev,baseline)
  grid=pd.read_csv(f/'calibration_candidates.csv');assert len(grid)==48 and set(zip(grid.gap,grid.duration))=={(a,b) for a in plan['gaps'] for b in plan['durations']}
  for _,g in grid.iterrows():
   p=postprocess(score>s['threshold'],int(g.gap),int(g.duration));a=ev.score(p);b=system.score(p)
   for x,y in [(g.local_f05,a['EW_F_0.50']),(g.system_f05,b['EW_F_0.50']),(g.false_seconds,a['false_positive_seconds']),(g.local_detected,a['TPe']),(g.alarm_fraction,p.mean()),(g.segments,a['num_pred_events'])]:same(x,y)
   assert bool(g.feasible)==(target.issubset(ids(ev,p)) and p.mean()<=.01)
   assert bool(g.original)==(g.gap==rule['merge_gap'] and g.duration==rule['min_dur'])
  winner=grid[grid.feasible].sort_values(['local_f05','false_seconds','original','gap','duration'],ascending=[False,True,False,True,True],kind='stable').iloc[0]
  for k,v in s['chosen'].items():
   # write_json serializes NumPy booleans through default=str.
   if k in ['feasible','original'] and isinstance(v,str):
    assert v in ['True','False'];v=v=='True'
   same(v,winner[k])
  score=np.load(src/'testing_score.npy');original=np.load(src/'testing_prediction.npy');chosen=s['chosen'];selected=np.load(f/'selected_prediction.npy')
  assert np.array_equal(original,postprocess(score>s['threshold'],int(rule['merge_gap']),int(rule['min_dur'])))
  assert np.array_equal(selected,postprocess(score>s['threshold'],chosen['gap'],chosen['duration']))
  assert (RUN/'selection_complete.json').stat().st_mtime<(f/'selected_prediction.npy').stat().st_mtime
  for v,p in [('original',original),('selected',selected)]:diagnostics.append(dict(method=method,variant=v,segments=int(np.sum(np.diff(np.r_[0,p,0].astype(np.int8))==1)),alarm_samples=int(p.sum()),sample_grid_alarm_seconds=int(p.sum())*30))
  for period in [str(y) for y in range(2007,2014)]+['2007-2013']:
   keep=np.ones(len(times),bool) if '-' in period else ((times>=pd.Timestamp(period+'-01-01').value)&(times<pd.Timestamp(str(int(period)+1)+'-01-01').value))
   for scope,ann in anns.items():
    e=EventEvaluator(times[keep],ann)
    for v,p in [('original',original),('selected',selected)]:
     metrics=e.score(p[keep])
     for k,value in metrics.items():
      same(value,saved.loc[(method,v,period,scope),k])
      if v=='original':same(value,old.loc[(method,period,scope),k])
     rows.append(dict(method=method,variant=v,period=period,scope=scope,**metrics))
    a,b=ids(e,original[keep]),ids(e,selected[keep])
    for aid in e.by_id:events.append(dict(method=method,period=period,scope=scope,event_id=aid,original_detected=aid in a,selected_detected=aid in b,gained=aid in b-a,lost=aid in a-b))
  print(method,'verified',flush=True)
 pd.testing.assert_frame_equal(pd.DataFrame(events),pd.read_csv(RUN/'event_ids.csv',dtype={'period':str}))
 pd.testing.assert_frame_equal(pd.DataFrame(diagnostics),pd.read_csv(RUN/'segment_diagnostics.csv'))
 OUT.mkdir(parents=True,exist_ok=True)
 pd.DataFrame(rows).to_csv(OUT/'verified_results.csv',index=False);pd.DataFrame(events).to_csv(OUT/'event_ids.csv',index=False);pd.DataFrame(diagnostics).to_csv(OUT/'segment_diagnostics.csv',index=False)
 (OUT/'verification.json').write_text(json.dumps(dict(verified=True,metric_rows=96,calibration_candidates=144,complete_receipt_sha256=sha(RUN/'complete.json'),checks=['Frozen datasets/source/protocol/reference receipts','All 144 calibration candidates and selection replay','Exact original/selected prediction replay','96 annual/pooled corrected metrics and prior control parity','Event IDs and segment/duration diagnostics','Assessment prediction artifacts created after freeze'],limitations=['Single seed; reused benchmark; offline lookahead','Saved-score replay; not a new independent inference replication']),indent=2)+'\n')
 print('VERIFIED',flush=True)
if __name__=='__main__':main()
