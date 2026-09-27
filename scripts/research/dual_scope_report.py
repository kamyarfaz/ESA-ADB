"""Report fixed channels40–47 predictions at input-channel and system event scopes."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from esa_thesis import config
from esa_thesis.diagnose_channels import FEATURES, digest
from esa_thesis.evaluation.esa import EventEvaluator


def scope_evaluator(times, annotations, features=None):
    labels=annotations.copy()
    if features is not None:
        # Out-of-scope annotations remain neutral, not nominal false-alarm targets.
        labels.loc[~labels.Channel.isin(features),'Category']='Out-of-scope (neutral)'
    return EventEvaluator(times,labels)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',required=True,type=Path)
    p.add_argument('--diagnostic',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    if a.output.exists():p.error('Use a new output directory')
    source=json.loads((a.baseline/'protocol.json').read_text())
    if source['spec']['features']!=FEATURES:raise ValueError('Expected channels40–47 baseline')
    diag=json.loads((a.diagnostic/'protocol.json').read_text())
    if diag['checkpoint_sha256']!=digest(a.baseline/'full_period/model_checkpoint.pt'):raise ValueError('Diagnostic/checkpoint mismatch')
    labels=pd.read_csv(config.ANNOTATIONS_FILE).merge(pd.read_csv(config.ANOMALY_TYPES_FILE),on='ID',validate='many_to_one')
    rows=[];events=[];hashes={}
    for period,path in [('calibration_2006',a.diagnostic/'calibration_channel_scores.npz'),('assessment_2007_2013',a.baseline/'full_period/assessment_prediction.npy')]:
        hashes[str(path)]=digest(path)
        if path.suffix=='.npz':
            with np.load(path) as f:times=f['timestamps'];prediction=f['prediction']
            expected=json.loads((a.diagnostic/'results.json').read_text())['calibration']
        else:
            prediction=np.load(path);times=pd.date_range('2007-01-01','2014-01-01',freq='30s',inclusive='left').asi8
            expected=json.loads((a.baseline/'full_period/results.json').read_text())
        for name,features in [('system_all_channels',None),('input_channels_40_47',FEATURES)]:
            ev=scope_evaluator(times,labels,features);result=ev.score(prediction)
            if features is None:
                for key in ['EW_F_0.50','TPe','FPe','FNe']:
                    if not np.isclose(result[key],expected[key],atol=1e-12,rtol=0):raise ValueError('System score differs from saved baseline')
            rows.append(dict(period=period,scope=name,**result))
            starts,ends,closed=ev.prediction_intervals(prediction)
            for aid,intervals in ev.by_id.items():
                detected=any(ev._overlaps(starts,ends,closed,lo,hi).any() for lo,hi in intervals)
                events.append(dict(period=period,scope=name,event_id=aid,detected=bool(detected)))
        if digest(path)!=hashes[str(path)]:raise ValueError('Predictions changed during reporting')
    a.output.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(a.output/'results.csv',index=False);pd.DataFrame(events).to_csv(a.output/'events.csv',index=False)
    (a.output/'protocol.json').write_text(json.dumps(dict(features=FEATURES,prediction_sha256=hashes,annotations={str(f):digest(f) for f in [config.ANNOTATIONS_FILE,config.ANOMALY_TYPES_FILE]},scope='Input scope detects raw event-ID unions restricted to input-channel intervals. System scope uses all channel intervals. Non-selected annotations remain neutral in both. Same predictions and threshold; no tuning.',limitations='Descriptive reuse of exposed assessment, not independent validation. A single global prediction does not measure per-channel localization. Supervisor said41–47, existing model uses40–47; no silent retraining or input changes.',source_sha256=digest(Path(__file__))),indent=2))
    print(pd.DataFrame(rows)[['period','scope','EW_F_0.50','TPe','FPe','FNe','num_true_events']].to_string(index=False))

if __name__=='__main__':main()
