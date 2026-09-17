"""Read-only diagnostics of frozen development predictions (no threshold tuning)."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from esa_thesis import config
from esa_thesis.development import FOLDS
from esa_thesis.protocol import evaluator_for
from esa_thesis.paths import PROJECT_ROOT


def analyze(run, fold, output, features):
    _, start, end = FOLDS[fold]
    times = pd.date_range(start, end, freq='30s', inclusive='left', tz='UTC').asi8
    score = np.load(run/'assessment_score.npy')
    pred = np.load(run/'assessment_prediction.npy')
    if len(score) != len(times):
        raise ValueError('Saved score does not match declared assessment')
    evaluator = evaluator_for(times)
    actual = evaluator.score(pred)
    saved = json.loads((run/'results.json').read_text())
    for key in ['EW_F_0.50', 'TPe', 'FPe', 'FNe']:
        if not np.isclose(actual[key], saved[key], atol=1e-12, rtol=0):
            raise ValueError(f'Saved metric mismatch: {key}')
    starts, ends, closed = evaluator.prediction_intervals(pred)
    labels = evaluator.annotations.copy()
    labels.StartTime = pd.to_datetime(labels.StartTime, utc=True).astype('int64')
    labels.EndTime = pd.to_datetime(labels.EndTime, utc=True).astype('int64')
    labels = labels[(labels.EndTime >= times[0]) & (labels.StartTime <= times[-1])]
    events = []
    matched = np.zeros(len(starts), dtype=bool)
    for aid, intervals in evaluator.by_id.items():
        hit = np.zeros(len(starts), dtype=bool)
        for lo, hi in intervals:
            hit |= evaluator._overlaps(starts, ends, closed, lo, hi)
        matched |= hit
        rows = labels[labels.ID.astype(str) == aid]
        channels = sorted(set(rows.Channel))
        events.append({'fold': fold, 'ID': aid, 'detected': bool(hit.any()),
                       'channels': ';'.join(channels), 'input_channels': ';'.join(sorted(set(channels)&set(features))),
                       'start': str(pd.Timestamp(intervals[0, 0], tz='UTC')),
                       'end': str(pd.Timestamp(intervals[-1, 1], tz='UTC'))})
    touches = np.zeros(len(starts), dtype=bool)
    for lo, hi in evaluator.all_intervals:
        touches |= evaluator._overlaps(starts, ends, closed, lo, hi)
    false = []
    for i in np.flatnonzero(~matched & ~touches):
        mask = (times >= starts[i]) & (times <= ends[i] if closed[i] else times < ends[i])
        false.append({'fold': fold, 'start': str(pd.Timestamp(starts[i], tz='UTC')),
                      'end': str(pd.Timestamp(ends[i], tz='UTC')), 'seconds': (ends[i]-starts[i])/1e9,
                      'peak_score': float(score[mask].max())})
    assert len(false) == actual['FPe']
    grid = pd.read_csv(run/'calibration_candidates.csv')
    raw = grid[(grid.merge_gap == 0) & (grid.min_dur == 1)].drop_duplicates('threshold')
    shift = raw[['threshold', 'val_pred_anomaly_rate', 'val_nominal_false_alarm_rate']].copy()
    shift['assessment_nominal_false_alarm_rate'] = [evaluator.score(score > t)['nominal_false_alarm_rate'] for t in shift.threshold]
    shift['assessment_exceedance_rate'] = [(score > t).mean() for t in shift.threshold]
    shift['fold'] = fold
    shift.to_csv(output/f'{fold}_score_tails.csv', index=False)
    rule = json.loads((run/'frozen_rule.json').read_text())['rule']
    summary = {'fold': fold, **actual, 'threshold': rule['threshold'],
               'calibration_f05': rule['val_esa_f05'], 'calibration_fp': rule['val_FPe'],
               'calibration_alarm_rate': rule['val_pred_anomaly_rate'],
               'assessment_alarm_rate': float(pred.mean()),
               'missed_with_input_channel': sum(not e['detected'] and bool(e['input_channels']) for e in events),
               'missed_without_input_channel': sum(not e['detected'] and not e['input_channels'] for e in events)}
    return summary, events, false


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    specs = next(s for s in config.RUN_SPECS if s['run']=='pa_center_ch08_40_47')
    summaries, events, false = [], [], []
    for fold in FOLDS:
        parent = 'ae_baseline_2003_seed42' if fold=='2003' else 'ae_baseline_remaining_seed42'
        run = PROJECT_ROOT/'results_longrun/development'/parent/fold
        summary, es, fs = analyze(run, fold, args.output, specs['features'])
        summaries.append(summary); events.extend(es); false.extend(fs)
    for name, rows in [('summary', summaries), ('events', events), ('false_alarms', false)]:
        pd.DataFrame(rows).to_csv(args.output/f'{name}.csv', index=False)
    print(pd.DataFrame(summaries).to_string(index=False))
    print(pd.DataFrame(events).to_string(index=False))
    print(pd.DataFrame(false).to_string(index=False))


if __name__ == '__main__':
    main()
