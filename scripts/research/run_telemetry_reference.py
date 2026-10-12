"""Fixed direct-telemetry pilot; frozen calibration before real validation."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from esa_thesis import config
from esa_thesis.data import RobustChannelScaler
from esa_thesis.development import load_period,QUANTILES
from esa_thesis.diagnose_channels import FEATURES,TRAIN_SHA,digest
from esa_thesis.telemetry_reference import deviations,fit_scale
from esa_thesis.masked_pilot import assessment_scopes
from esa_thesis.thresholds import sweep_thresholds,select_rows,postprocess
root=Path('results_longrun/development/telemetry_reference_v1')
if root.exists():raise ValueError('Preserve output')
control=Path('results_longrun/development/inner_selection_cuda_v2')
original=json.loads((control/'protocol.json').read_text())
assert all(digest(p)==h for p,h in original['annotations'].items())
print('Golden CSV verification',flush=True);assert digest(config.TRAIN_FILE)==TRAIN_SHA
root.mkdir(parents=True)
(root/'protocol.json').write_text(json.dumps(dict(train_sha256=TRAIN_SHA,annotations=original['annotations'],plan_sha256=digest('docs/research/telemetry-reference-plan-2026-10-08.md'),implementation_sha256=digest('esa_thesis/telemetry_reference.py'),runner_sha256=digest(__file__)),indent=2)+'\n')
rows=[];events=[]
for fold in ('2001','2002'):
 out=root/fold;out.mkdir();print(fold,'loading training/calibration only',flush=True)
 raw,y,t=load_period(config.TRAIN_FILE,FEATURES,None,fold+'-07-01')
 split=np.searchsorted(t,pd.Timestamp(fold+'-01-01',tz='UTC').value)
 scaler=RobustChannelScaler().fit(raw[:split],y[:split]);x=scaler.transform(raw)
 for seed in (42,43,44):
  f=control/fold/f'seed{seed}';m=json.loads((f/'complete.json').read_text());assert all(digest(f/n)==h for n,h in m.items())
  pre=json.loads((f/'preflight.json').read_text());np.testing.assert_array_equal(scaler.median_,pre['scaler_median']);np.testing.assert_array_equal(scaler.iqr_,pre['scaler_iqr'])
 errors,state=deviations(x);scale=fit_scale(errors[:split],y[:split]==0)
 # Independent EWM verification on all training/calibration samples.
 xx=x.astype(np.float64);means=pd.DataFrame(xx).ewm(halflife=2880,adjust=False).mean().to_numpy()
 np.testing.assert_allclose(errors,np.abs(xx-np.vstack([xx[0],means[:-1]])),atol=1e-9,rtol=0)
 values=(errors[split:]/scale).max(axis=1);ct=t[split:]
 grid=sweep_thresholds(values,evaluator=assessment_scopes(ct)['system_all_channels'],threshold_values=np.percentile(values,QUANTILES),merge_gaps=[0,16,64],min_durs=[1,8,32])
 rule=select_rows(grid).iloc[0].to_dict();grid.to_csv(out/'calibration_candidates.csv',index=False)
 (out/'frozen_rule.json').write_text(json.dumps(dict(rule=rule,state=state.tolist(),scale=scale.tolist(),median=scaler.median_.tolist(),iqr=scaler.iqr_.tolist()),indent=2)+'\n')
 np.save(out/'calibration_score.npy',values);np.save(out/'calibration_timestamps.npy',ct)
 print(fold,'rule frozen; loading real validation',flush=True)
 del raw,x,errors,xx,means
 raw,_,vt=load_period(config.TRAIN_FILE,FEATURES,fold+'-07-01',str(int(fold)+1)+'-01-01')
 vx=scaler.transform(raw);err,final=deviations(vx,state)
 # Include previous state as first EWM observation to independently check continuation.
 joined=np.vstack([state,vx.astype(np.float64)])
 means=pd.DataFrame(joined).ewm(halflife=2880,adjust=False).mean().to_numpy()
 np.testing.assert_allclose(err,np.abs(joined[1:]-means[:-1]),atol=1e-9,rtol=0)
 score=(err/scale).max(axis=1);pred=postprocess(score>rule['threshold'],int(rule['merge_gap']),int(rule['min_dur']))
 np.save(out/'validation_score.npy',score);np.save(out/'validation_prediction.npy',pred);np.save(out/'validation_timestamps.npy',vt)
 for scope,ev in assessment_scopes(vt).items():
  rows.append(dict(fold=fold,arm='telemetry_reference',seed='deterministic',scope=scope,calibration_f05=rule['val_esa_f05'],**ev.score(pred)))
  a,b,c=ev.prediction_intervals(pred)
  for aid,intervals in ev.by_id.items():events.append(dict(fold=fold,scope=scope,event_id=aid,detected=bool(any(ev._overlaps(a,b,c,lo,hi).any() for lo,hi in intervals))))
 for seed in (42,43,44):
  f=control/fold/f'seed{seed}';np.testing.assert_array_equal(vt,np.load(f/'validation_timestamps.npy'))
  saved=pd.read_csv(f/'results.csv');saved=saved[saved.policy=='calibration_selected'].copy();saved['arm']='joint';saved['fold']=fold;saved['seed']=str(seed)
  rows.extend(saved.to_dict('records'))
 print(fold,'complete',flush=True)
frame=pd.DataFrame(rows);frame.to_csv(root/'results.csv',index=False);pd.DataFrame(events).to_csv(root/'events.csv',index=False)
summary=frame.groupby(['fold','arm','scope']).agg(f05=('EW_F_0.50','mean'),false_events=('FPe','mean'),recall=('EW_recall','mean'),nominal_alarm_fraction=('nominal_false_alarm_rate','mean')).reset_index();summary.to_csv(root/'summary.csv',index=False)
s=summary[summary.scope=='system_all_channels'].set_index(['fold','arm'])
passed=all(s.loc[(f,'telemetry_reference'),'f05']>s.loc[(f,'joint'),'f05'] and s.loc[(f,'telemetry_reference'),'false_events']<=s.loc[(f,'joint'),'false_events'] for f in ('2001','2002'))
print('Final integrity check',flush=True);assert digest(config.TRAIN_FILE)==TRAIN_SHA
assert all(digest(p)==h for p,h in original['annotations'].items())
(root/'verification.json').write_text(json.dumps(dict(status='complete',advancement_gate_passed=bool(passed),promoted=False,independent_check='Pandas EWM vs SciPy all train/cal/validation deviations within1e-9',deterministic_runs=2),indent=2)+'\n')
print(summary.to_string(index=False),flush=True);print('Gate',passed,flush=True)
