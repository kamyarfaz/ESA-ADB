"""Frozen temporal post-processing; calibration first, assessment after freeze."""
import fcntl,json,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
from esa_thesis import config
from esa_thesis.calibration_audit import digest
from esa_thesis.full_short_no_roll import verify_frozen,report_predictions
from esa_thesis.full_similarity_mosad import write_json,save_npy,verify_receipt
from esa_thesis.full_combinations import score_model
from esa_thesis.masked_pilot import assessment_scopes
from esa_thesis.thresholds import postprocess
from esa_thesis.development_recovery import check_protocol
R=Path(__file__).resolve().parents[2];SRC=R/'results_longrun/wandb_finetuning_20261024_seed42_v1';OUT=R/'results_longrun/temporal_postprocessing_20261028_v1'
GAPS=(0,1,2,4,8,16,32,64);DURS=(1,2,4,8,16,32)
def read(p):return json.loads(p.read_text())
def ids(ev,p):
 s,e,c=ev.prediction_intervals(p)
 return {i for i,xs in ev.by_id.items() if any(ev._overlaps(s,e,c,a,b).any() for a,b in xs)}
def choose(rows):
 eligible=[r for r in rows if r['feasible']]
 if not eligible:raise ValueError('No eligible fallback')
 return sorted(eligible,key=lambda r:(-r['local_f05'],r['false_seconds'],not r['original'],r['gap'],r['duration']))[0]
def main():
 lock=open(R/'.temporal-postprocessing.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 torch.set_num_threads(2)
 verify_frozen(R)
 original_plan=read(SRC/'protocol.json')
 for n,h in original_plan['source_sha256'].items():
  if digest(R/'esa_thesis'/n)!=h:raise ValueError('Source changed '+n)
 for group in ['annotations','reference']:
  for n,h in original_plan[group].items():
   if digest(R/n)!=h:raise ValueError('Frozen reference changed '+n)
 receipt=read(SRC/'complete.json')
 for n,h in receipt['files'].items():
  if digest(SRC/n)!=h:raise ValueError('Prior artifact changed '+n)
 selections=read(SRC/'selection_complete.json')
 plan=dict(source_receipt=digest(SRC/'complete.json'),selection=digest(SRC/'selection_complete.json'),runner=digest(Path(__file__)),protocol=digest(R/'docs/research/temporal-postprocessing-protocol-2026-10-28.md'),source=original_plan['source_sha256'],runtime=dict(torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,cuda=torch.version.cuda),settings={k:str(v) for k,v in vars(config).items() if k.isupper()},gaps=GAPS,durations=DURS)
 check_protocol(OUT,plan,(OUT/'protocol.json').exists())
 if (OUT/'complete.json').exists():
  assert verify_receipt(OUT,'complete.json');print('Already complete');return
 ct=pd.date_range('2006-01-01','2007-01-01',freq='30s',inclusive='left').asi8;evs=assessment_scopes(ct);frozen={}
 for method,s in selections.items():
  f=OUT/method;f.mkdir(exist_ok=True);src=SRC/s['folder'];rule=s['rule']
  # Only calibration features are opened in this phase.
  score=score_model(src/'model.pt',f,'calibration')
  base=postprocess(score>rule['threshold'],int(rule['merge_gap']),int(rule['min_dur']));local=evs['input_channels_40_47'];detected=ids(local,base);rows=[]
  for gap in GAPS:
   for dur in DURS:
    p=postprocess(score>rule['threshold'],gap,dur);a=local.score(p);b=evs['system_all_channels'].score(p)
    rows.append(dict(gap=gap,duration=dur,local_f05=a['EW_F_0.50'],system_f05=b['EW_F_0.50'],false_seconds=a['false_positive_seconds'],local_detected=a['TPe'],alarm_fraction=float(p.mean()),segments=a['num_pred_events'],original=gap==rule['merge_gap'] and dur==rule['min_dur'],feasible=detected.issubset(ids(local,p)) and p.mean()<=.01))
  grid=pd.DataFrame(rows);grid.to_csv(f/'calibration_candidates.csv',index=False);winner=choose(rows)
  frozen[method]=dict(source_folder=s['folder'],threshold=rule['threshold'],baseline_rule=rule,chosen=winner,grid_sha256=digest(f/'calibration_candidates.csv'),calibration_sha256=digest(f/'calibration_score.npy'))
  print(method,'calibration selected',winner,flush=True)
 if (OUT/'selection_complete.json').exists():assert read(OUT/'selection_complete.json')==frozen
 else:write_json(OUT/'selection_complete.json',frozen)
 print('All selections frozen; assessing',flush=True)
 times=pd.date_range('2007-01-01','2014-01-01',freq='30s',inclusive='left').asi8;records=[];events=[];diagnostics=[]
 for method,s in frozen.items():
  src=SRC/s['source_folder'];f=OUT/method;score=np.load(src/'testing_score.npy');original=np.load(src/'testing_prediction.npy');r=s['baseline_rule']
  assert np.array_equal(original,postprocess(score>s['threshold'],int(r['merge_gap']),int(r['min_dur'])))
  c=s['chosen'];selected=postprocess(score>s['threshold'],c['gap'],c['duration']);save_npy(f/'selected_prediction.npy',selected)
  for variant,p in [('original',original),('selected',selected)]:
   annual,pooled=report_predictions(times,p);frame=pd.concat([annual,pooled]);frame.insert(0,'variant',variant);frame.insert(0,'method',method);records.append(frame)
   diagnostics.append(dict(method=method,variant=variant,segments=int(np.sum(np.diff(np.r_[0,p,0].astype(np.int8))==1)),alarm_samples=int(p.sum()),sample_grid_alarm_seconds=int(p.sum())*30))
  for period in [str(y) for y in range(2007,2014)]+['2007-2013']:
   keep=np.ones(len(times),bool) if '-' in period else ((times>=pd.Timestamp(period+'-01-01').value)&(times<pd.Timestamp(str(int(period)+1)+'-01-01').value))
   for scope,ev in assessment_scopes(times[keep]).items():
    a,b=ids(ev,original[keep]),ids(ev,selected[keep])
    for aid in ev.by_id:events.append(dict(method=method,period=period,scope=scope,event_id=aid,original_detected=aid in a,selected_detected=aid in b,gained=aid in b-a,lost=aid in a-b))
  print(method,'assessment complete',flush=True)
 pd.concat(records).to_csv(OUT/'all_results.csv',index=False);pd.DataFrame(events).to_csv(OUT/'event_ids.csv',index=False);pd.DataFrame(diagnostics).to_csv(OUT/'segment_diagnostics.csv',index=False)
 verify_frozen(R)
 write_json(OUT/'complete.json',dict(needs_independent_verification=True,files={str(p.relative_to(OUT)):digest(p) for p in OUT.rglob('*') if p.is_file() and p.name!='complete.json'}))
 print('Completed; independent verification and publication required',flush=True)
if __name__=='__main__':main()
