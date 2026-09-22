"""Audit frozen 2004 AE alarms against raw events without threshold tuning."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from esa_thesis import config
from esa_thesis.protocol import evaluator_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    run = root/'results_longrun/development/ae_baseline_remaining_seed42/2004'
    times = pd.date_range('2004-04-01', '2005-01-01', freq='30s', inclusive='left', tz='UTC').asi8
    evaluator = evaluator_for(times)
    prediction = np.load(run/'assessment_prediction.npy')
    scores = np.load(run/'assessment_score.npy')
    saved = json.loads((run/'results.json').read_text())
    metrics = evaluator.score(prediction)
    for key, value in metrics.items():
        if isinstance(value, (float, int)):
            np.testing.assert_allclose(value, saved[key], rtol=0, atol=1e-12)
    rule = json.loads((run/'frozen_rule.json').read_text())['rule']
    np.testing.assert_array_equal(prediction, scores > rule['threshold'])
    starts, ends, closed = evaluator.prediction_intervals(prediction)
    attribution = pd.read_csv(root/'results_longrun/development/channel_diagnostics_2026-09-21/alarm_channel_errors.csv')
    alarms = attribution[(attribution.fold == 2004) & (attribution.variant == 'baseline') & attribution.dominant].copy()
    assert len(alarms) == len(starts)
    annotations = pd.read_csv(config.ANNOTATIONS_FILE)
    rows = []
    for aid, intervals in evaluator.by_id.items():
        hits = np.zeros(len(starts), dtype=bool)
        for lo, hi in intervals:
            hits |= evaluator._overlaps(starts, ends, closed, lo, hi)
        channels = sorted(annotations.loc[annotations.ID.astype(str) == aid, 'Channel'].unique())
        rows.append({'id': aid, 'detected': bool(hits.any()), 'channels': ','.join(channels),
                     'input_channels': ','.join(c for c in channels if c in [f'channel_{i}' for i in range(40,48)]),
                     'alarm_count': int(hits.sum())})
    pd.DataFrame(rows).to_csv(args.output/'events.csv', index=False)
    # Distance to any annotation, including categories neutral for false alarms.
    distances = []
    for start, end in zip(starts, ends):
        intervals = evaluator.all_intervals
        distances.append(float(np.maximum(0, np.maximum(intervals[:,0]-end, start-intervals[:,1])).min()/1e9))
    np.testing.assert_array_equal(pd.to_datetime(alarms.start, utc=True).astype('int64'), starts)
    alarms['nearest_annotation_seconds'] = distances
    alarms['peak_over_threshold'] = alarms.peak_score / rule['threshold']
    alarms.to_csv(args.output/'alarms.csv', index=False)
    (args.output/'metrics.json').write_text(json.dumps({'metrics': metrics, 'frozen_rule': rule,
        'scope': 'Descriptive development audit; no model or decision changes; no test CSV accessed'}, indent=2))
    print(alarms.to_string(index=False))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
