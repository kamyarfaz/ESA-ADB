import csv, hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path('/home/k_faz/projects/ESA-ADB')
sys.path.insert(0,str(ROOT))
from esa_thesis.calibration_audit import verify_recorded,load_annotations
from esa_thesis.evaluation.esa import EventEvaluator

base=ROOT/'results_longrun/long_baseline/ch40_47_train2000_2005_seed42'
paths=[base/'full_period/assessment_prediction.npy',base/'full_period/results.json',base/'yearly_results.csv',base/'protocol.json',ROOT/'data/ESA-Mission1/labels.csv',ROOT/'data/ESA-Mission1/anomaly_types.csv']
hashes=verify_recorded(ROOT,paths)
protocol=json.loads((base/'protocol.json').read_text())
assert protocol['train']==['2000-01-01','2006-01-01'] and protocol['calibration']==['2006-01-01','2007-01-01'] and protocol['assessment']==['2007-01-01','2014-01-01']
assert protocol['seed']==42 and protocol['epochs']==40 and protocol['max_train_windows']==0
labels=load_annotations(paths[-2],paths[-1]);pred=np.load(paths[0],allow_pickle=False)
times=pd.date_range('2007-01-01','2014-01-01',freq='30s',inclusive='left').asi8
saved=json.loads(paths[1].read_text());annual=pd.read_csv(paths[2]).set_index('year')
rows=[]
for period in ['2007-2013']+list(range(2007,2014)):
    mask=np.ones(len(times),dtype=bool) if isinstance(period,str) else ((times>=pd.Timestamp(f'{period}-01-01').value)&(times<pd.Timestamp(f'{period+1}-01-01').value))
    m=EventEvaluator(times[mask],labels).score(pred[mask]);old=saved if isinstance(period,str) else annual.loc[period]
    for key in ('EW_F_0.50','EW_precision','EW_recall','nominal_false_alarm_rate'):
        assert np.isclose(m[key],old[key],rtol=1e-11,atol=1e-10),(period,key)
    assert np.isclose(m['EW_recall'],m['TPe']/(m['TPe']+m['FNe']))
    assert np.isclose(m['nominal_false_alarm_rate'],m['false_positive_seconds']/m['nominal_seconds'])
    rows.append(dict(period=str(period),precision=m['EW_precision'],tpr=m['EW_recall'],fpr=m['nominal_false_alarm_rate'],f05=m['EW_F_0.50']))
out=Path(__file__).parent
(out/'data.json').write_text(json.dumps(dict(rows=rows,sha256=hashes,baseline_verified=True,candidate_status='In progress',fpr='Nominal false-alarm duration / nominal duration; not pointwise FP/(FP+TN)',tpr='Detected target event IDs / all target event IDs (event recall)'),indent=2))
print('Replayed baseline pooled and all seven annual system scores; rate definitions verified.')
