"""Descriptive event coverage audit; never select inputs or rules on test outcomes."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from esa_thesis.calibration_audit import digest, verify_recorded, load_annotations
from esa_thesis.evaluation.esa import EventEvaluator, CATEGORIES

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'results_longrun/long_baseline/ch40_47_train2000_2005_seed42'
NEW = ROOT/'results_longrun/short_no_roll_full_20261014_seed42_v1'
OUT = ROOT/'docs/research/tables/full-period-missed-events-2026-10-16'
INPUTS = {f'channel_{i}' for i in range(40,48)}

def detected_ids(evaluator, prediction):
    starts, ends, closed = evaluator.prediction_intervals(prediction)
    found = {aid for aid, intervals in evaluator.by_id.items()
             if any(np.any((starts <= hi) & ((ends > lo) | ((ends == lo) & closed)))
                    for lo, hi in intervals)}
    assert len(found) == evaluator.score(prediction)['TPe']
    return found

def main():
    if OUT.exists():
        raise ValueError('Preserve existing audit directory')
    annotation_paths = [ROOT/'data/ESA-Mission1/labels.csv',ROOT/'data/ESA-Mission1/anomaly_types.csv']
    hashes = verify_recorded(ROOT, annotation_paths + [BASE/'full_period/assessment_prediction.npy',BASE/'full_period/results.json',BASE/'yearly_results.csv'])
    completion = json.loads((NEW/'complete.json').read_text())
    for name in ['full_period/assessment_prediction.npy','all_results.csv']:
        actual = digest(NEW/name)
        assert actual == completion['artifacts'][name]
        hashes[str((NEW/name).relative_to(ROOT))] = actual
    labels = load_annotations(*annotation_paths)
    times = pd.date_range('2007-01-01','2014-01-01',freq='30s',inclusive='left').asi8
    preds = {name:np.load(folder/'full_period/assessment_prediction.npy',allow_pickle=False)
             for name,folder in [('baseline',BASE),('short_no_roll',NEW)]}
    assert all(p.shape == times.shape for p in preds.values())
    old_annual = pd.read_csv(BASE/'yearly_results.csv').set_index('year')
    old_pooled = json.loads((BASE/'full_period/results.json').read_text())
    new_saved = pd.read_csv(NEW/'all_results.csv',dtype={'period':str}).set_index(['period','scope'])
    rows, summary = [], []
    for period in ['2007-2013']+[str(y) for y in range(2007,2014)]:
        mask = np.ones(len(times),bool) if period=='2007-2013' else ((times>=pd.Timestamp(f'{period}-01-01').value)&(times<pd.Timestamp(f'{int(period)+1}-01-01').value))
        t = times[mask]
        ev = EventEvaluator(t,labels)
        part = labels[(labels.StartTime.astype('int64')<=t[-1]) & (labels.EndTime.astype('int64')>=t[0]) & labels.Category.isin(CATEGORIES)]
        local_labels = labels.copy()
        local_labels.loc[~local_labels.Channel.isin(INPUTS),'Category']='Out-of-scope (neutral)'
        local_ev = EventEvaluator(t,local_labels)
        detections = {name:detected_ids(ev,p[mask]) for name,p in preds.items()}
        local_detections = {name:detected_ids(local_ev,p[mask]) for name,p in preds.items()}
        for name,p in preds.items():
            metric = ev.score(p[mask])
            saved = (old_pooled if period=='2007-2013' else old_annual.loc[int(period)]) if name=='baseline' else new_saved.loc[(period,'system_all_channels')]
            for k in ['TPe','FNe','EW_F_0.50','EW_precision','EW_recall']:
                assert np.isclose(metric[k],saved[k],rtol=1e-11,atol=1e-10),(period,name,k)
        for aid, intervals in ev.by_id.items():
            group = part[part.ID.astype(str)==aid]
            channels = set(group.Channel)
            b,n = aid in detections['baseline'], aid in detections['short_no_roll']
            outcome = 'both_detected' if b and n else 'baseline_only' if b else 'short_no_roll_only' if n else 'both_missed'
            rows.append(dict(period=period,event_id=aid,category=group.Category.iloc[0],
                start_utc=str(pd.Timestamp(intervals[0,0],tz='UTC')),end_utc=str(pd.Timestamp(intervals[-1,1],tz='UTC')),
                annotation_channels=';'.join(sorted(channels)),input_annotation_channels=';'.join(sorted(channels&INPUTS)),
                has_input_annotation=bool(channels&INPUTS),baseline_detected=b,short_no_roll_detected=n,
                baseline_detected_on_input_interval=aid in local_detections['baseline'],
                short_no_roll_detected_on_input_interval=aid in local_detections['short_no_roll'],outcome=outcome))
        frame = pd.DataFrame([r for r in rows if r['period']==period])
        for coverage in ['all','input_annotated','outside_inputs_only']:
            subset = frame if coverage=='all' else frame[frame.has_input_annotation == (coverage=='input_annotated')]
            summary.append(dict(period=period,coverage=coverage,events=len(subset),
                baseline_detected=int(subset.baseline_detected.sum()),short_no_roll_detected=int(subset.short_no_roll_detected.sum()),
                **{name:int((subset.outcome==name).sum()) for name in ['both_detected','baseline_only','short_no_roll_only','both_missed']}))
    frame = pd.DataFrame(rows)
    totals = pd.DataFrame(summary)
    pooled = frame[frame.period=='2007-2013']
    assert len(pooled)==91 and pooled.baseline_detected.sum()==54 and pooled.short_no_roll_detected.sum()==48
    channel_rows=[]
    for channel in sorted(labels.Channel.unique()):
        subset=pooled[pooled.annotation_channels.str.split(';').apply(lambda x:channel in x)]
        channel_rows.append(dict(channel=channel,is_model_input=channel in INPUTS,annotated_event_ids=len(subset),
            baseline_missed=int((~subset.baseline_detected).sum()),short_no_roll_missed=int((~subset.short_no_roll_detected).sum())))
    OUT.mkdir(parents=True)
    frame.to_csv(OUT/'event_outcomes.csv',index=False)
    totals.to_csv(OUT/'coverage_summary.csv',index=False)
    pd.DataFrame(channel_rows).to_csv(OUT/'annotation_channel_counts.csv',index=False)
    (OUT/'verification.json').write_text(json.dumps(dict(verified=True,dependency_sha256=hashes,
        checks=['Pooled and all annual detections reconcile with saved corrected metrics','Per-ID detection counts reconcile with evaluator TPe','Annotation coverage calculated within each reporting period'],
        limits=['Descriptive reused-test audit only; no input selection or threshold tuning','Annotations are not proof of model observability or causal detection','Channel counts overlap and must not be summed','Annual event IDs can recur across years']),indent=2)+'\n')
    print(totals.to_string(index=False))
    print('\nChanged pooled events:\n'+pooled[pooled.outcome.isin(['baseline_only','short_no_roll_only'])].to_string(index=False))

if __name__=='__main__':
    main()
