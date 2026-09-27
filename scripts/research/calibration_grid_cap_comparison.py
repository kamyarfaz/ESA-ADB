"""Fixed-checkpoint 2x2 calibration diagnostic; never loads assessment data."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from esa_thesis import config
from esa_thesis.development import QUANTILES
from esa_thesis.diagnose_channels import digest
from esa_thesis.protocol import evaluator_for
from esa_thesis.thresholds import sweep_thresholds, postprocess

SOURCES={2003:('ae_baseline_2003_seed42','baseline_2003_epoch25_gpu_v1'),2004:('ae_baseline_remaining_seed42','baseline_2004_epoch35_v2'),2005:('ae_baseline_remaining_seed42','baseline_2005_epoch5_v1')}

def choose(frame,policy):
    column='val_pred_anomaly_rate' if policy=='total_alarm' else 'val_nominal_false_alarm_rate'
    eligible=frame[(frame[column]<=.01)&(~frame.val_saturated)]
    if eligible.empty:raise ValueError('No eligible candidate')
    return eligible.sort_values(['val_esa_f05','val_false_positive_seconds','threshold','merge_gap','min_dur'],ascending=[False,True,False,True,True],kind='stable').iloc[0]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--plan-only',action='store_true');a=p.parse_args()
    plan=dict(scope='January-March2003/2004/2005 calibration only',checkpoints='Frozen historical epochs25/35/5; no checkpoint reselection',original_quantiles=QUANTILES,dense_quantiles=sorted(set(QUANTILES+[round(90+i*.1,2) for i in range(101)])),merge_gaps=[0,16,64],min_durs=[1,8,32],policies=['total_alarm','nominal_false_alarm'],cap=.01,saturation_guard=config.SATURATION_RATE,limitations='Exploratory, motivated by previously inspected2005 calibration. Not independent validation. Alternative cap not an agreed operational requirement. No assessment or frozen decision changes.')
    if a.plan_only:
        a.output.mkdir(parents=True,exist_ok=False)
        (a.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');print('Plan recorded before computation.');return
    if json.loads((a.output/'plan.json').read_text())!=plan:raise ValueError('Plan changed')
    if (a.output/'comparison.csv').exists():raise ValueError('Preserve completed output')
    rows=[];hashes={};blocked=[]
    for year,(run,diagnostic) in SOURCES.items():
        source=Path('results_longrun/development')/run/str(year)
        diag=Path('results_longrun/diagnostics')/diagnostic
        if not (diag/'results.json').exists():
            blocked.append(dict(year=year,reason='No completed verified diagnostic; omitted, not accepted'))
            print(year,'BLOCKED: diagnostic incomplete',flush=True)
            continue
        files=[source/'frozen_rule.json',source/'calibration_candidates.csv',diag/'calibration_channel_scores.npz',diag/'results.json',config.ANNOTATIONS_FILE,config.ANOMALY_TYPES_FILE]
        before={str(f):digest(f) for f in files};hashes[str(year)]=before
        rule=json.loads(files[0].read_text())['rule']
        with np.load(files[2]) as cache:
            times=cache['timestamps'];score=cache['scores'].max(axis=0);saved=cache['prediction']
        np.testing.assert_array_equal(times,pd.date_range(f'{year}-01-01',f'{year}-04-01',freq='30s',inclusive='left').asi8)
        prediction=postprocess(score>rule['threshold'],int(rule['merge_gap']),int(rule['min_dur']))
        np.testing.assert_array_equal(prediction,saved)
        ev=evaluator_for(times)
        for k,v in ev.score(prediction).items():
            if isinstance(v,(int,float)):assert np.isclose(v,rule['val_'+k],rtol=0,atol=1e-6),(year,k)
        original=pd.read_csv(files[1])
        for key in ['threshold','merge_gap','min_dur','val_esa_f05']:
            assert np.isclose(choose(original,'total_alarm')[key],rule[key],rtol=0,atol=1e-9)
        # Retain exact historical thresholds, avoiding CPU/GPU percentile-boundary drift.
        dense_values=np.unique(np.r_[original.threshold.to_numpy(),np.percentile(score,plan['dense_quantiles'])])
        dense=sweep_thresholds(score,evaluator=ev,threshold_values=dense_values,merge_gaps=plan['merge_gaps'],min_durs=plan['min_durs'])
        dense.to_csv(a.output/f'{year}_dense_candidates.csv',index=False)
        for grid,frame in [('original',original),('dense',dense)]:
            for policy in plan['policies']:
                selected=choose(frame,policy)
                rows.append(dict(year=year,grid=grid,policy=policy,**selected.to_dict()))
        assert before=={str(f):digest(f) for f in files}
        print(year,'completed',flush=True)
    pd.DataFrame(rows).to_csv(a.output/('comparison_partial.csv' if blocked else 'comparison.csv'),index=False)
    (a.output/'verification.json').write_text(json.dumps(dict(inputs=hashes,status='partial' if blocked else 'complete',blocked=blocked,reproduction='Included folds reproduced frozen predictions and calibration metrics'),indent=2)+'\n')
    print(pd.DataFrame(rows)[['year','grid','policy','val_esa_f05','val_TPe','val_FPe','val_FNe','val_pred_anomaly_rate','val_nominal_false_alarm_rate']].to_string(index=False))

if __name__=='__main__':main()
