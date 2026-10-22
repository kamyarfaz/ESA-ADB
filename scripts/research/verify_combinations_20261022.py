"""Replay frozen combination predictions and corrected metrics from raw annotations."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from esa_thesis.evaluation.esa import EventEvaluator
from esa_thesis.thresholds import select_rows

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'results_longrun/combinations_full_20261021_seed42_v1'
REF = ROOT / 'results_longrun/short_no_roll_full_20261014_seed42_v1'
OUT = ROOT / 'docs/research/tables/combinations-2026-10-22'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def read(path):
    return json.loads(path.read_text())

receipt = read(RUN/'complete.json')
for name, digest in receipt['files'].items():
    assert sha(RUN/name) == digest, name
p = read(RUN/'protocol.json')
assert p == read(RUN/'resume_protocol.json')
for name, digest in p['annotations'].items():
    assert sha(ROOT/name) == digest, name
for name, digest in p['source_sha256'].items():
    assert sha(ROOT/'esa_thesis'/name) == digest, name
for name, digest in p['reference'].items():
    assert sha(REF/name) == digest, name
base = ROOT/'data/frozen/mission1-reference-v1'
assert sha(base/'manifest.json') == p['manifest_sha']
for name, record in read(base/'manifest.json')['files'].items():
    assert sha(base/name) == record['sha256'], name
print('Frozen data, source, reference and artifact hashes verified', flush=True)
selected = {}
for method in p['arms']:
    folder = RUN/method
    for name, digest in read(folder/'trained.json')['files'].items():
        assert sha(folder/name) == digest
    history = pd.read_csv(folder/'history.csv')
    assert history.epoch.tolist() == list(range(1,41))
    assert np.isfinite(history.loss).all()
    eligible = history.dropna(subset=['calibration_f05'])
    assert eligible.epoch.tolist() == list(range(25 if method == 'pretrain_finetune' else 5,41,5))
    frozen = read(folder/'frozen_rule.json')
    assert int(eligible.loc[eligible.calibration_f05.idxmax(),'epoch']) == frozen['epoch']
    chosen = select_rows(pd.read_csv(folder/'calibration_candidates.csv')).iloc[0]
    for key in ['threshold','merge_gap','min_dur','val_esa_f05']:
        assert np.isclose(chosen[key],frozen['rule'][key],rtol=1e-12)
    selected[method] = frozen['epoch']
fusion = read(RUN/'fusion/frozen_rule.json')
grid = pd.read_csv(RUN/'fusion/calibration_candidates.csv')
choices = [(float(select_rows(g).iloc[0].val_esa_f05),float(w)) for w,g in grid.groupby('masked_weight',sort=True)]
assert max(choices,key=lambda x:(x[0],-x[1]))[1] == fusion['masked_weight']
label_cols = [c for c in pd.read_csv(base/'calibration.csv',nrows=0).columns if c.startswith('is_anomaly_')]
nominal = np.concatenate([~(frame.to_numpy()>0).any(axis=1) for frame in pd.read_csv(base/'calibration.csv',usecols=label_cols,chunksize=25000)])
for method, key in [('control','scale_control'),('masked','scale_masked')]:
    a = np.load(RUN/method/'calibration_score.npy')
    assert np.isclose(np.percentile(a[nominal],99),fusion[key],rtol=1e-12)
times = pd.date_range('2007-01-01','2014-01-01',freq='30s',inclusive='left').asi8
labels = pd.read_csv(ROOT/'data/ESA-Mission1/labels.csv').merge(pd.read_csv(ROOT/'data/ESA-Mission1/anomaly_types.csv'),on='ID',validate='many_to_one')
local = labels.copy()
local.loc[~local.Channel.isin([f'channel_{i}' for i in range(40,48)]),'Category'] = 'Out-of-scope (neutral)'
saved = pd.read_csv(RUN/'all_results.csv',dtype={'period':str}).set_index(['method','period','scope'])
assert len(saved)==80 and saved.index.is_unique
rows = []
for method in ['control',*p['arms'],'fusion']:
    folder = RUN/method
    rule = read((REF/'full_period' if method=='control' else folder)/'frozen_rule.json')['rule']
    score = np.load(folder/'testing_score.npy',allow_pickle=False)
    pred = np.load(folder/'testing_prediction.npy',allow_pickle=False)
    assert len(score)==len(pred)==len(times)==7364160 and np.isfinite(score).all()
    assert rule['merge_gap']==0 and rule['min_dur']==1
    assert np.array_equal(pred,score>rule['threshold'])
    if method=='control':
        assert np.array_equal(pred,np.load(REF/'full_period/assessment_prediction.npy'))
    if method=='fusion':
        a=np.load(RUN/'control/testing_score.npy'); b=np.load(RUN/'masked/testing_score.npy')
        w=fusion['masked_weight']
        replay=(1-w)*a/fusion['scale_control']+w*b/fusion['scale_masked']
        assert np.array_equal(score,replay)
    for period in [str(y) for y in range(2007,2014)]+['2007-2013']:
        mask=np.ones(len(times),dtype=bool) if '-' in period else ((times>=pd.Timestamp(f'{period}-01-01').value)&(times<pd.Timestamp(f'{int(period)+1}-01-01').value))
        for scope, annotations in [('system_all_channels',labels),('input_channels_40_47',local)]:
            metrics=EventEvaluator(times[mask],annotations).score(pred[mask])
            old=saved.loc[(method,period,scope)]
            for key,value in metrics.items():
                assert value==old[key] if isinstance(value,str) else np.isclose(value,old[key],rtol=1e-11,atol=1e-10), (method,period,scope,key)
            rows.append(dict(method=method,period=period,scope=scope,**metrics))
    print(method+' prediction and all 16 metric rows verified',flush=True)
OUT.mkdir(parents=True,exist_ok=True)
pd.DataFrame(rows).to_csv(OUT/'verified_results.csv',index=False)
(OUT/'verification.json').write_text(json.dumps(dict(verified=True,selected_epochs=selected,samples_per_method=len(times),metric_rows=len(rows),run_receipt_sha256=sha(RUN/'complete.json'),verified_results_sha256=sha(OUT/'verified_results.csv'),checks=['Frozen data/source/reference and all run artifact hashes','Complete finite 40-epoch histories and earliest calibration-best checkpoint','Saved calibration-grid rule and fusion-weight selection','Nominal calibration fusion normalization','Exact threshold prediction replay for all methods','Exact control-reference prediction and fusion-score replay','80 corrected annual/pooled metric rows independently recomputed from raw annotations'],limitations=['Prediction replay uses saved model scores; no second neural inference run','Single seed and previously inspected benchmark; no independent validation']),indent=2)+'\n')
