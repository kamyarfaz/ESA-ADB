"""Execute the predeclared six-run calibration diagnostic from verified scores."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from esa_thesis.calibration_audit import verify_recorded,digest,load_annotations
from esa_thesis.evaluation.esa import EventEvaluator
from esa_thesis.nominal_quantile import nominal_threshold,advancement
from esa_thesis.thresholds import postprocess

FEATURES=[f'channel_{i}' for i in range(40,48)]

def equal_metrics(actual,saved,prefix=''):
    for k,v in actual.items():
        old=saved[prefix+k]
        if isinstance(v,str):
            if old!=v:raise ValueError('Metric version mismatch')
        elif not np.isclose(v,old,rtol=1e-11,atol=1e-9):
            raise ValueError(f'Metric reproduction failed: {k}: {v} != {old}')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Use a new output directory')
    root=Path(__file__).resolve().parents[2];base=root/'results_longrun/development/inner_selection_cuda_v2'
    labelpaths=[root/'data/ESA-Mission1'/n for n in ('labels.csv','anomaly_types.csv')]
    deps=list(labelpaths)
    for year in (2001,2002):
        deps.append(base/str(year)/'calibration_timestamps.npy')
        for seed in (42,43,44):
            f=base/str(year)/f'seed{seed}'
            deps += [f/'frozen_decisions.json',f/'epoch_5/calibration_score.npy']
    before=verify_recorded(root,deps);labels=load_annotations(*labelpaths)
    decisions=[];calrows=[]
    for year in (2001,2002):
        times=np.load(base/str(year)/'calibration_timestamps.npy',allow_pickle=False)
        if times[0]!=pd.Timestamp(f'{year}-01-01',tz='UTC').value or times[-1]!=pd.Timestamp(f'{year}-07-01',tz='UTC').value-30*10**9:raise ValueError('Calibration period changed')
        ev=EventEvaluator(times,labels)
        for seed in (42,43,44):
            f=base/str(year)/f'seed{seed}';control=json.loads((f/'frozen_decisions.json').read_text())['fixed_epoch5']
            if control['epoch']!=5:raise ValueError('Wrong checkpoint')
            s=np.load(f/'epoch_5/calibration_score.npy',allow_pickle=False)
            threshold,n=nominal_threshold(times,s,labels);rule=control['rule']
            for arm,value in [('control',rule['threshold']),('nominal_q999',threshold)]:
                d=dict(fold=year,seed=seed,arm=arm,epoch=5,threshold=value,merge_gap=int(rule['merge_gap']),min_dur=int(rule['min_dur']),nominal_samples=n)
                decisions.append(d);pred=postprocess(s>value,d['merge_gap'],d['min_dur']);metrics=ev.score(pred)
                if arm=='control':
                    equal_metrics(metrics,rule,'val_')
                    if not np.isclose(pred.mean(),rule['val_pred_anomaly_rate'],atol=1e-12,rtol=0):raise ValueError('Calibration occupancy mismatch')
                calrows.append(dict(**d,total_occupancy=float(pred.mean()),**metrics))
    if verify_recorded(root,deps)!=before:raise ValueError('Inputs changed')
    a.output.mkdir(parents=True)
    frozen=a.output/'frozen_decisions.json';frozen.write_text(json.dumps(decisions,indent=2)+'\n');frozen.chmod(0o444)
    seal=digest(frozen);(a.output/'frozen_decisions.sha256').write_text(seal+'\n')
    pd.DataFrame(calrows).to_csv(a.output/'calibration.csv',index=False)
    print('All six candidate/control pairs frozen before reading validation arrays',flush=True)
    valdeps=[]
    for y in (2001,2002):
        for seed in (42,43,44):
            f=base/str(y)/f'seed{seed}'
            valdeps += [f/n for n in ('validation_timestamps.npy','fixed_epoch5_validation_score.npy','fixed_epoch5_validation_prediction.npy','results.csv')]
    valhash=verify_recorded(root,valdeps);rows=[]
    for y in (2001,2002):
        for seed in (42,43,44):
            f=base/str(y)/f'seed{seed}';times=np.load(f/'validation_timestamps.npy',allow_pickle=False);s=np.load(f/'fixed_epoch5_validation_score.npy',allow_pickle=False)
            if times[0]!=pd.Timestamp(f'{y}-07-01',tz='UTC').value or times[-1]!=pd.Timestamp(f'{y+1}-01-01',tz='UTC').value-30*10**9:raise ValueError('Validation period changed')
            if s.shape!=times.shape or not np.isfinite(s).all():raise ValueError('Invalid validation scores')
            local=labels.copy();local.loc[~local.Channel.isin(FEATURES),'Category']='Out-of-scope (neutral)'
            evs={'system_all_channels':EventEvaluator(times,labels),'input_channels_40_47':EventEvaluator(times,local)}
            saved=pd.read_csv(f/'results.csv')
            for d in decisions:
                if d['fold']!=y or d['seed']!=seed:continue
                pred=postprocess(s>d['threshold'],d['merge_gap'],d['min_dur'])
                if d['arm']=='control' and not np.array_equal(pred,np.load(f/'fixed_epoch5_validation_prediction.npy',allow_pickle=False)):raise ValueError('Control predictions differ')
                np.save(a.output/f'{y}_seed{seed}_{d["arm"]}_prediction.npy',pred)
                for scope,ev in evs.items():
                    metrics=ev.score(pred)
                    if d['arm']=='control':equal_metrics(metrics,saved[(saved.policy=='fixed_epoch5')&(saved.scope==scope)].iloc[0])
                    rows.append(dict(**d,scope=scope,total_occupancy=float(pred.mean()),**metrics))
    if digest(frozen)!=seal or verify_recorded(root,valdeps)!=valhash:raise ValueError('Frozen inputs changed')
    frame=pd.DataFrame(rows);frame.to_csv(a.output/'results.csv',index=False)
    summary=frame.groupby(['fold','arm','scope']).agg(f05_mean=('EW_F_0.50','mean'),f05_sd=('EW_F_0.50','std'),fp_mean=('FPe','mean'),false_seconds_mean=('false_positive_seconds','mean'),recall_mean=('EW_recall','mean'),occupancy_mean=('total_occupancy','mean')).reset_index()
    summary.to_csv(a.output/'summary.csv',index=False)
    decision=advancement(summary);decision.update(input_sha256={**before,**valhash},frozen_decisions_sha256=seal,source_sha256={str(p.relative_to(root)):digest(p) for p in [Path(__file__),root/'esa_thesis/nominal_quantile.py',root/'esa_thesis/evaluation/esa.py',root/'esa_thesis/thresholds.py',root/'docs/research/nominal-quantile-plan-2026-10-13.md']},limits='Reused development periods; fixed epoch 5 diagnostic, not replacement primary policy; no unseen generalization claim')
    (a.output/'verification_and_decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    print(summary.to_string(index=False));print('ADVANCE:',decision['advance'])
if __name__=='__main__':main()
