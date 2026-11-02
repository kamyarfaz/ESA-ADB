"""Independently replay descriptive calibration blocks; no new selection."""
import json
import numpy as np
import pandas as pd
from esa_thesis.evaluation.esa import EventEvaluator
from scripts.research.target_boundary_common import ROOT, SOURCE, CALIBRATION, OUTPUT, TABLE, TARGETS
from scripts.research.verify_target_boundary_20261101 import read, pooled, coarse_replay, boundary_replay, ids, same, sha


def main():
    frozen = read(OUTPUT / 'selection_complete.json')
    selections = read(SOURCE / 'selection_complete.json')
    source = OUTPUT / 'calibration_stability.csv'
    stored = pd.read_csv(source).fillna('').set_index(['arm', 'variant', 'block', 'scope'])
    assert len(stored) == 252 and stored.index.is_unique
    times = pd.date_range('2006-01-01', '2007-01-01', freq='30s', inclusive='left').asi8
    annotations = pd.read_csv(ROOT / 'data/ESA-Mission1/labels.csv').merge(
        pd.read_csv(ROOT / 'data/ESA-Mission1/anomaly_types.csv'), on='ID', validate='many_to_one')
    local = annotations.copy()
    local.loc[~local.Channel.isin(TARGETS), 'Category'] = 'Out-of-scope (neutral)'
    rows = 0
    for arm, spec in frozen.items():
        coarse, fine = pooled(OUTPUT / 'scores' / spec['method'] / 'calibration', spec['effective_pool'])
        predictions = {'base': coarse > spec['threshold']}
        predictions['boundary'] = boundary_replay(predictions['base'], fine, spec['boundary']['rule'])
        for family, choice in spec['choices'].items():
            predictions[family] = boundary_replay(coarse_replay(coarse, spec['threshold'], choice['rule']),
                                                  fine, spec['boundary']['rule'])
        predictions['original'] = np.load(CALIBRATION / spec['method'] / 'calibration_score.npy') > selections[spec['method']]['rule']['threshold']
        # Generate six calendar blocks independently of the runner's block list.
        for prefix, months in [('H', 6), ('Q', 3)]:
            for number, month in enumerate(range(1, 13, months), 1):
                start = pd.Timestamp(year=2006, month=month, day=1)
                end = start + pd.DateOffset(months=months)
                keep = (times >= start.value) & (times < end.value)
                for scope, labels in [('system_all_channels', annotations), ('input_channels_40_47', local)]:
                    ev = EventEvaluator(times[keep], labels)
                    for variant, pred in predictions.items():
                        metrics = ev.score(pred[keep])
                        metrics['detected_ids'] = ','.join(ids(ev, pred[keep]))
                        same(metrics, stored.loc[(arm, variant, prefix + str(number), scope)].to_dict())
                        rows += 1
    assert rows == 252
    receipt = {'verified': True, 'rows': rows, 'table_sha256': sha(source),
               'checks': ['Independent transforms and block generation', 'All block metric cells and detected IDs'],
               'limitation': 'Descriptive replay only; blocks are not independent validation folds.'}
    (TABLE / 'stability-verification.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('Verified all252 descriptive stability rows.')


if __name__ == '__main__':
    main()
