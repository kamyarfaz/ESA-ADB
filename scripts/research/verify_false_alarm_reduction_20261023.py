"""Independent replay and event-ID accounting for completed alarm refinements."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from esa_thesis.evaluation.esa import EventEvaluator
from esa_thesis.thresholds import postprocess

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'results_longrun/false_alarm_reduction_20261023_seed42_v1'
SOURCE=ROOT/'results_longrun/combinations_full_20261021_seed42_v1'
OUT=ROOT/'docs/research/tables/false-alarm-reduction-2026-10-23'
METHODS=('pretrain_finetune','expanded_short_no_roll')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def read(path):return json.loads(path.read_text())

def ids(ev,pred):
    starts,ends,closed=ev.prediction_intervals(pred)
    return {aid for aid,intervals in ev.by_id.items() if any(ev._overlaps(starts,ends,closed,a,b).any() for a,b in intervals)}

receipt=read(RUN/'complete.json')
for name,digest in receipt['files'].items():assert sha(RUN/name)==digest,name
protocol=read(RUN/'protocol.json')
for name,digest in protocol['source_sha256'].items():assert sha(ROOT/name)==digest,name
assert sha(ROOT/'docs/research/false-alarm-reduction-protocol-2026-10-23.md')==protocol['research_protocol_sha256']
assert sha(SOURCE/'complete.json')==protocol['source_receipt']
prior=read(SOURCE/'protocol.json')
for name,digest in prior['annotations'].items():assert sha(ROOT/name)==digest
base=ROOT/'data/frozen/mission1-reference-v1'
assert sha(base/'manifest.json')==prior['manifest_sha']
for name,record in read(base/'manifest.json')['files'].items():assert sha(base/name)==record['sha256'],name
freeze=read(RUN/'selection_complete.json')
assert sha(RUN/'research_review.json')==freeze['research_review_sha256']
for method,digest in freeze['rules'].items():assert sha(RUN/method/'frozen_rule.json')==digest
labels=pd.read_csv(ROOT/'data/ESA-Mission1/labels.csv').merge(pd.read_csv(ROOT/'data/ESA-Mission1/anomaly_types.csv'),on='ID',validate='many_to_one')
local=labels.copy();local.loc[~local.Channel.isin([f'channel_{i}' for i in range(40,48)]),'Category']='Out-of-scope (neutral)'
annotations={'system_all_channels':labels,'input_channels_40_47':local}
saved=pd.read_csv(RUN/'all_results.csv',dtype={'period':str}).set_index(['method','variant','period','scope'])
old=pd.read_csv(SOURCE/'all_results.csv',dtype={'period':str}).set_index(['method','period','scope'])
assert len(saved)==64 and saved.index.is_unique
times=pd.date_range('2007-01-01','2014-01-01',freq='30s',inclusive='left').asi8
cal_times=pd.date_range('2006-01-01','2007-01-01',freq='30s',inclusive='left').asi8
reference=np.load(SOURCE/'control/testing_prediction.npy')
rows=[];comparison=[]
for method in METHODS:
    folder=RUN/method;frozen=read(folder/'frozen_rule.json');candidate=frozen['candidate']
    grid=pd.read_csv(folder/'calibration_candidates.csv')
    eligible=grid[grid.feasible].sort_values(['system_f05','false_alarm_seconds','original','candidate'],ascending=[False,True,False,True],kind='stable')
    assert int(eligible.iloc[0].candidate)==candidate['candidate']
    assert sha(folder/'calibration_candidates.csv')==frozen['calibration_candidates_sha256']
    cal=np.load(folder/'calibration_per_channel.npy')
    cal_raw=cal.max(axis=0)
    cal_score=cal_raw if candidate['family']=='raw' else (cal/np.maximum(np.asarray(frozen['normalization_q99'],np.float32)[:,None],1e-8)).max(axis=0)
    cal_base=postprocess(cal_raw>frozen['baseline']['threshold'],int(frozen['baseline']['merge_gap']),int(frozen['baseline']['min_dur']))
    cal_pred=postprocess(cal_score>candidate['threshold'],int(candidate['merge_gap']),int(candidate['min_dur']))
    assert cal_pred.mean()<=.01
    for scope,ann in annotations.items():
        ev=EventEvaluator(cal_times,ann)
        assert ids(ev,cal_base).issubset(ids(ev,cal_pred))
        a,b=ev.score(cal_base),ev.score(cal_pred)
        assert b['FPe']<=a['FPe'] and b['false_positive_seconds']<=a['false_positive_seconds']+1e-8
    del cal,cal_score,cal_raw
    per=np.load(folder/'testing_per_channel.npy');raw=per.max(axis=0)
    assert np.array_equal(raw,np.load(SOURCE/method/'testing_score.npy'))
    predictions={}
    for variant in ('original','refined'):
        r=frozen['baseline'] if variant=='original' else candidate
        score=raw if variant=='original' or r['family']=='raw' else (per/np.maximum(np.asarray(frozen['normalization_q99'],np.float32)[:,None],1e-8)).max(axis=0)
        replay=postprocess(score>r['threshold'],int(r['merge_gap']),int(r['min_dur']))
        pred=np.load(folder/(variant+'_prediction.npy'))
        assert np.array_equal(replay,pred) and len(pred)==len(times)
        if variant=='original':assert np.array_equal(pred,np.load(SOURCE/method/'testing_prediction.npy'))
        predictions[variant]=pred
        for period in [str(y) for y in range(2007,2014)]+['2007-2013']:
            keep=np.ones(len(times),bool) if '-' in period else ((times>=pd.Timestamp(f'{period}-01-01').value)&(times<pd.Timestamp(f'{int(period)+1}-01-01').value))
            for scope,ann in annotations.items():
                ev=EventEvaluator(times[keep],ann);metrics=ev.score(pred[keep]);record=saved.loc[(method,variant,period,scope)]
                for key,value in metrics.items():
                    assert value==record[key] if isinstance(value,str) else np.isclose(value,record[key],rtol=1e-11,atol=1e-9),(method,variant,period,scope,key)
                    if variant=='original':
                        previous=old.loc[(method,period,scope)]
                        assert value==previous[key] if isinstance(value,str) else np.isclose(value,previous[key],rtol=1e-11,atol=1e-9)
                rows.append(dict(method=method,variant=variant,period=period,scope=scope,**metrics))
    for scope,ann in annotations.items():
        ev=EventEvaluator(times,ann)
        before=ids(ev,predictions['original']);after=ids(ev,predictions['refined']);ref=ids(ev,reference)
        for aid in sorted(ev.by_id):comparison.append(dict(method=method,scope=scope,event_id=aid,short_no_roll=aid in ref,original=aid in before,refined=aid in after,lost=aid in before and aid not in after,gained=aid not in before and aid in after))
    print(method+' verified',flush=True)
OUT.mkdir(parents=True,exist_ok=True)
pd.DataFrame(rows).to_csv(OUT/'verified_results.csv',index=False)
pd.DataFrame(comparison).to_csv(OUT/'event_comparison.csv',index=False)
write=dict(verified=True,metric_rows=64,samples_per_prediction=len(times),complete_receipt_sha256=sha(RUN/'complete.json'),checks=['All artifact/source/frozen-data/reference fingerprints','Frozen rule hashes and candidate selection','Exact calibration baseline detected-ID retention and alarm constraints','New per-channel inference exactly reproduces original aggregate scores','Original and refined prediction replay','All 64 annual/pooled metric rows independently recomputed','Original metrics equal prior completed run','Event IDs gained/lost and short-no-roll comparison'],limitations=['One seed; reused assessment benchmark','Calibration retention does not guarantee assessment retention'])
(OUT/'verification.json').write_text(json.dumps(write,indent=2)+'\n')
