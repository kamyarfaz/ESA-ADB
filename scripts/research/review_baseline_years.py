from pathlib import Path
import json
import numpy as np,pandas as pd
from esa_thesis.masked_pilot import assessment_scopes
from esa_thesis.thresholds import postprocess
from esa_thesis.protocol import evaluator_for
out=Path('docs/research/tables/baseline-years-2026-09-25');out.mkdir(exist_ok=True)
summary=[];alarms=[];months=[];events=[];histories=[]
for year in (2003,2004,2005):
 root=Path('results_longrun/development')/('ae_baseline_2003_seed42' if year==2003 else 'ae_baseline_remaining_seed42')/str(year)
 frozen=json.loads((root/'frozen_rule.json').read_text());rule=frozen['rule'];history=pd.read_csv(root/'history.csv');history['year']=year;histories.append(history)
 assert frozen['epoch']==int(history.loc[history.calibration_f05.idxmax(),'epoch'])
 t=pd.date_range(f'{year}-04-01',f'{year+1}-01-01',freq='30s',inclusive='left').asi8
 score=np.load(root/'assessment_score.npy');pred=np.load(root/'assessment_prediction.npy')
 np.testing.assert_array_equal(pred,postprocess(score>rule['threshold'],int(rule['merge_gap']),int(rule['min_dur'])))
 scopes=assessment_scopes(t);ev=scopes['system_all_channels'];metrics=ev.score(pred)
 saved=json.loads((root/'results.json').read_text())
 for key in ['EW_F_0.50','TPe','FPe','FNe','false_positive_seconds']:assert np.isclose(metrics[key],saved[key],rtol=0,atol=1e-9)
 cal=evaluator_for(pd.date_range(f'{year}-01-01',f'{year}-04-01',freq='30s',inclusive='left').asi8)
 assert len(cal.by_id)==rule['val_num_true_events']
 summary.append(dict(year=year,epoch=frozen['epoch'],calibration_f05=rule['val_esa_f05'],calibration_events=len(cal.by_id),calibration_TP=rule['val_TPe'],calibration_FP=rule['val_FPe'],threshold=rule['threshold'],merge_gap=rule['merge_gap'],min_dur=rule['min_dur'],local_f05=scopes['input_channels_40_47'].score(pred)['EW_F_0.50'],**metrics))
 a,b,c=ev.prediction_intervals(pred)
 for i,(start,end) in enumerate(zip(a,b)):
  ids=[eid for eid,ints in ev.by_id.items() if any(ev._overlaps(a[i:i+1],b[i:i+1],c[i:i+1],lo,hi)[0] for lo,hi in ints)]
  touch=any(ev._overlaps(a[i:i+1],b[i:i+1],c[i:i+1],lo,hi)[0] for lo,hi in ev.all_intervals)
  alarms.append(dict(year=year,start=pd.Timestamp(start,tz='UTC'),end=pd.Timestamp(end,tz='UTC'),duration_minutes=(end-start)/60e9,false_positive=not touch,detected_ids=';'.join(ids)))
 for eid,ints in ev.by_id.items():
  hit=any(ev._overlaps(a,b,c,lo,hi).any() for lo,hi in ints)
  events.append(dict(year=year,event_id=eid,detected=hit,annotated_on_inputs=eid in scopes['input_channels_40_47'].by_id))
 nominal=np.ones(len(t),dtype=bool)
 for lo,hi in ev.all_intervals:nominal[(t<=hi)&(t+30_000_000_000>lo)]=False
 calendar=pd.DatetimeIndex(t).month
 for month in range(4,13):
  mask=nominal&(calendar==month);v=score[mask]/rule['threshold']
  months.append(dict(year=year,month=month,nominal_bins=int(mask.sum()),median_score_to_threshold=float(np.median(v)),q99_score_to_threshold=float(np.quantile(v,.99)),max_score_to_threshold=float(v.max()),fraction_nominal_bins_alarm=float(pred[mask].mean())))
pd.DataFrame(summary).to_csv(out/'summary.csv',index=False);pd.DataFrame(alarms).to_csv(out/'alarm_intervals.csv',index=False);pd.DataFrame(events).to_csv(out/'events.csv',index=False);pd.DataFrame(months).to_csv(out/'monthly_nominal_scores.csv',index=False);pd.concat(histories).to_csv(out/'histories.csv',index=False)
print(pd.DataFrame(summary).to_string(index=False))
print('FALSE ALARMS');print(pd.DataFrame(alarms).query('false_positive').to_string(index=False))
print('2004 NOMINAL SCORES');print(pd.DataFrame(months).query('year==2004').to_string(index=False))
print('MISSES');print(pd.DataFrame(events).query('not detected').to_string(index=False))
