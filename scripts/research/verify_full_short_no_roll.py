"""Independently replay the completed fixed-period prediction and metric files."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from esa_thesis.evaluation.esa import EventEvaluator

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'results_longrun/short_no_roll_full_20261014_seed42_v1'
OUT = ROOT / 'docs/research/tables/full-short-no-roll-2026-10-16'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

complete = json.loads((RUN/'complete.json').read_text())
for name, expected in complete['artifacts'].items():
    assert sha(RUN/name) == expected, name
protocol = json.loads((RUN/'protocol.json').read_text())
for name, expected in protocol['annotations'].items():
    assert sha(ROOT/name) == expected, name
for name, expected in protocol['source_sha256'].items():
    assert sha(ROOT/'esa_thesis'/name) == expected, name
base = ROOT/'data/frozen/mission1-reference-v1'
assert sha(base/'manifest.json') == protocol['manifest_sha256']
for name, rec in json.loads((base/'manifest.json').read_text())['files'].items():
    assert sha(base/name) == rec['sha256'], name
history = pd.read_csv(RUN/'full_period/history.csv')
assert history.epoch.tolist() == list(range(1,41))
rule = json.loads((RUN/'full_period/frozen_rule.json').read_text())
eligible = history.dropna(subset=['calibration_f05'])
assert int(eligible.loc[eligible.calibration_f05.idxmax(),'epoch']) == rule['epoch'] == complete['selected_epoch']
# This frozen run selects no gap filling and minimum duration one: direct threshold replay.
assert rule['rule']['merge_gap'] == 0 and rule['rule']['min_dur'] == 1
scores = np.load(RUN/'full_period/assessment_score.npy', allow_pickle=False)
pred = np.load(RUN/'full_period/assessment_prediction.npy', allow_pickle=False)
assert np.isfinite(scores).all() and np.array_equal(pred, scores > rule['rule']['threshold'])
times = pd.date_range('2007-01-01','2014-01-01',freq='30s',inclusive='left').asi8
assert len(pred) == len(times) == 7364160
labels = pd.read_csv(ROOT/'data/ESA-Mission1/labels.csv').merge(pd.read_csv(ROOT/'data/ESA-Mission1/anomaly_types.csv'),on='ID',validate='many_to_one')
local = labels.copy()
local.loc[~local.Channel.isin([f'channel_{i}' for i in range(40,48)]),'Category'] = 'Out-of-scope (neutral)'
saved = pd.read_csv(RUN/'all_results.csv',dtype={'period':str}).set_index(['period','scope'])
rows = []
for period in [str(y) for y in range(2007,2014)]+['2007-2013']:
    mask = np.ones(len(times),dtype=bool) if '-' in period else ((times >= pd.Timestamp(f'{period}-01-01').value) & (times < pd.Timestamp(f'{int(period)+1}-01-01').value))
    for scope, annotations in [('system_all_channels',labels),('input_channels_40_47',local)]:
        metrics = EventEvaluator(times[mask],annotations).score(pred[mask])
        old = saved.loc[(period,scope)]
        for key, value in metrics.items():
            if isinstance(value,str): assert value == old[key], (period,scope,key)
            else: assert np.isclose(value,old[key],rtol=1e-11,atol=1e-10), (period,scope,key)
        rows.append(dict(period=period,scope=scope,**metrics))
OUT.mkdir(parents=True,exist_ok=True)
pd.DataFrame(rows).to_csv(OUT/'verified_results.csv',index=False)
(OUT/'verification.json').write_text(json.dumps(dict(verified=True,samples=len(pred),epochs=40,selected_epoch=rule['epoch'],artifact_hashes=complete['artifacts'],checks=['All recorded run artifacts and package source hashes','All frozen data hashes','Earliest maximum calibration checkpoint','Independent direct-threshold prediction replay','All 16 annual/pooled metric rows replayed from raw annotations']),indent=2)+'\n')
data_path = ROOT/'outputs/thesis-system-20261016/build/data.json'
data = json.loads(data_path.read_text())
system = {r['period']:r for r in rows if r['scope']=='system_all_channels'}
data['candidate_rows'] = [dict(period=p,precision=system[p]['EW_precision'],tpr=system[p]['EW_recall'],fpr=system[p]['nominal_false_alarm_rate'],f05=system[p]['EW_F_0.50']) for p in ['2007-2013']+[str(y) for y in range(2007,2014)]]
data['candidate_status'] = 'Complete'
data['candidate_verified'] = True
data_path.write_text(json.dumps(data,indent=2)+'\n')
print(pd.DataFrame(rows)[['period','scope','EW_F_0.50']].to_string(index=False))
