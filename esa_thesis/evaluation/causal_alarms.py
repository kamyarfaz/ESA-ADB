"""Calibration-only causal k-of-w alarm confirmation on frozen forecast scores."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ..development import QUANTILES
from ..forecasting import CONTEXT
from ..protocol import evaluator_for

RULES = ((1, 1), (2, 2), (3, 3), (2, 3), (3, 5))  # (required hits, past window)


def confirm(score, threshold, hits, window, covered):
    """At t, use only t and the preceding w-1 samples; never backdate alarms."""
    score = np.asarray(score)
    covered = np.asarray(covered, dtype=bool)
    if score.ndim != 1 or score.shape != covered.shape or not np.isfinite(score).all():
        raise ValueError('Invalid score/coverage')
    if not 1 <= hits <= window:
        raise ValueError('Require 1 <= hits <= window')
    raw = covered & (score > threshold)
    cumulative = np.r_[0, np.cumsum(raw, dtype=np.int64)]
    end = np.arange(1, len(score)+1)
    count = cumulative[end]-cumulative[np.maximum(0, end-window)]
    return covered & (count >= hits)


def select_rule(scores, covered, evaluator):
    thresholds = np.unique([*np.percentile(scores[covered], QUANTILES), scores.max()])
    rows = []
    for threshold in thresholds:
        for hits, window in RULES:
            pred = confirm(scores, threshold, hits, window, covered)
            rows.append({'threshold': float(threshold), 'hits': hits, 'window': window,
                         'rate': float(pred.mean()), **evaluator.score(pred)})
    grid = pd.DataFrame(rows)
    eligible = grid[grid.rate <= .01]
    chosen = eligible.sort_values(['EW_F_0.50', 'false_positive_seconds', 'window', 'hits', 'threshold'],
                                 ascending=[False, True, True, True, False], kind='stable').iloc[0].to_dict()
    return chosen, grid


def alarm_details(evaluator, prediction):
    starts, ends, closed = evaluator.prediction_intervals(prediction)
    rows = []
    for start, end, shut in zip(starts, ends, closed):
        matched = []
        for aid, intervals in evaluator.by_id.items():
            if any(evaluator._overlaps(np.array([start]), np.array([end]), np.array([shut]), lo, hi)[0] for lo, hi in intervals):
                matched.append(aid)
        touches = any(evaluator._overlaps(np.array([start]), np.array([end]), np.array([shut]), lo, hi)[0]
                      for lo, hi in evaluator.all_intervals)
        rows.append({'start': str(pd.Timestamp(start, tz='UTC')), 'end': str(pd.Timestamp(end, tz='UTC')),
                     'seconds': (end-start)/1e9, 'kind': 'detected_event' if matched else ('neutral' if touches else 'false_alarm'),
                     'event_ids': ';'.join(matched)})
    return rows


def event_delays(evaluator, prediction):
    starts, ends, closed = evaluator.prediction_intervals(prediction)
    rows = []
    for aid, intervals in evaluator.by_id.items():
        first = None
        for lo, hi in intervals:
            hit = evaluator._overlaps(starts, ends, closed, lo, hi)
            if hit.any():
                value = int(np.maximum(starts[hit], lo).min())
                first = value if first is None else min(first, value)
        rows.append({'ID': aid, 'detected': first is not None,
                     'delay_seconds': None if first is None else (first-int(intervals[0, 0]))/1e9,
                     'first_overlap': None if first is None else str(pd.Timestamp(first, tz='UTC'))})
    return rows


def run(source, fold, output):
    protocol = json.loads((source/'protocol.json').read_text())
    if protocol['version'] != 'causal_forecast_development_v1' or protocol['context'] != CONTEXT:
        raise ValueError('Unsupported source protocol')
    a, b, c = protocol['folds'][fold]
    times = pd.date_range(a, b, freq='30s', inclusive='left', tz='UTC').asi8
    evaluator = evaluator_for(times)
    output.mkdir(parents=True, exist_ok=False)
    plan = {'source': str(source.resolve()), 'source_protocol_sha256': hashlib.sha256((source/'protocol.json').read_bytes()).hexdigest(),
            'fold': fold, 'rules': RULES, 'quantiles': QUANTILES, 'cap': .01,
            'selection': 'calibration F0.5; ties: nominal alarm seconds, shorter window, fewer hits, higher threshold',
            'checkpoint_policy': 'existing checkpoints fixed; no retraining or epoch reselection',
            'delay': 'first predicted interval overlap minus earliest clipped event-ID onset; ongoing alarms give zero; misses remain missing'}
    (output/'protocol.json').write_text(json.dumps(plan, indent=2))
    frozen = {}
    summaries, alarms, delays = [], [], []
    # Freeze ALL models before loading any assessment predictions or scores.
    for name in protocol['models']:
        directory = source/fold/name
        scores = np.load(directory/'calibration_score.npy')
        if scores.shape != times.shape:
            raise ValueError('Calibration timestamps do not align')
        covered = np.arange(len(scores)) >= CONTEXT
        old = json.loads((directory/'frozen_rule.json').read_text())['rule']
        baseline = confirm(scores, old['threshold'], 1, 1, covered)
        if not np.isclose(evaluator.score(baseline)['EW_F_0.50'], old['val_esa_f05'], atol=1e-12, rtol=0):
            raise ValueError('Original calibration metric mismatch')
        selected, grid = select_rule(scores, covered, evaluator)
        grid.to_csv(output/f'{name}_calibration_candidates.csv', index=False)
        frozen[name] = {'selected': selected, 'baseline_threshold': old['threshold'],
                        'calibration_score_sha256': hashlib.sha256((directory/'calibration_score.npy').read_bytes()).hexdigest()}
    (output/'frozen_rules.json').write_text(json.dumps(frozen, indent=2))
    times = pd.date_range(b, c, freq='30s', inclusive='left', tz='UTC').asi8
    evaluator = evaluator_for(times)
    for name, rules in frozen.items():
        directory = source/fold/name
        scores = np.load(directory/'assessment_score.npy')
        covered = np.load(directory/'assessment_covered.npy')
        if scores.shape != times.shape or not np.array_equal(covered, np.arange(len(scores)) >= CONTEXT):
            raise ValueError('Assessment coverage/timestamps mismatch')
        selected = rules['selected']
        for kind, threshold, hits, window in [('baseline', rules['baseline_threshold'], 1, 1),
                                            ('confirmed', selected['threshold'], int(selected['hits']), int(selected['window']))]:
            pred = confirm(scores, threshold, hits, window, covered)
            if kind == 'baseline' and not np.array_equal(pred, np.load(directory/'assessment_prediction.npy')):
                raise ValueError('Baseline mask mismatch')
            summaries.append({'model': name, 'rule': kind, 'threshold': threshold, 'hits': hits, 'window': window,
                              **evaluator.score(pred)})
            alarms.extend({'model': name, 'rule': kind, **row} for row in alarm_details(evaluator, pred))
            delays.extend({'model': name, 'rule': kind, **row} for row in event_delays(evaluator, pred))
            np.save(output/f'{name}_{kind}_prediction.npy', pred)
    for filename, rows in [('results', summaries), ('alarm_intervals', alarms), ('event_delays', delays)]:
        pd.DataFrame(rows).to_csv(output/f'{filename}.csv', index=False)
    print(pd.DataFrame(summaries)[['model', 'rule', 'hits', 'window', 'EW_F_0.50', 'TPe', 'FPe', 'FNe']].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--fold', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.fold, args.output)


if __name__ == '__main__': main()
