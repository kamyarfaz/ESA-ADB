import unittest
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from esa_thesis.calibration_audit import (
    detection_ids, neutralize, period_annotations, temporal_components,
    union_seconds, without_targets, digest, verify_recorded,
)
from esa_thesis.evaluation.esa import EventEvaluator
from scripts.research.audit_calibration_influence import pick


def frame(rows):
    df = pd.DataFrame(rows, columns=['ID', 'StartTime', 'EndTime', 'Category'])
    for col in ('StartTime', 'EndTime'):
        df[col] = pd.to_datetime(df[col], unit='s', utc=True)
    return df


class CalibrationAuditTests(unittest.TestCase):
    def test_components_do_not_bridge_disconnected_intervals_or_neutral_gaps(self):
        df = frame([('a', 1, 2, 'Anomaly'), ('a', 9, 10, 'Anomaly'),
                    ('b', 4, 5, 'Rare Event'), ('gap', 0, 12, 'Communication Gap'),
                    ('c', 10, 11, 'Anomaly')])
        self.assertEqual(temporal_components(df), [['a', 'c'], ['b']])
        self.assertEqual(union_seconds(df[df.Category!='Communication Gap']), 4)

    def test_transitive_overlap_is_grouped(self):
        df = frame([('a', 1, 3, 'Anomaly'), ('b', 3, 5, 'Anomaly'),
                    ('c', 5, 8, 'Anomaly')])
        self.assertEqual(temporal_components(df), [['a', 'b', 'c']])

    def test_half_open_period_and_closed_annotation_boundary(self):
        df = frame([('before', 0, 2, 'Anomaly'), ('carry', 1, 3, 'Anomaly'),
                    ('after', 6, 7, 'Anomaly'), ('point', 3, 3, 'Anomaly')])
        p = period_annotations(df, pd.Timestamp(3, unit='s', tz='UTC'),
                               pd.Timestamp(6, unit='s', tz='UTC'))
        self.assertEqual(set(p.ID), {'carry', 'point'})
        self.assertTrue(p.loc[p.ID=='carry', 'carried_in'].iloc[0])
        self.assertEqual(union_seconds(p), 0.)

    def test_withholding_matches_evaluator_on_random_predictions(self):
        times = pd.to_datetime(np.arange(31), unit='s', utc=True).asi8
        df = frame([('a', 3, 5, 'Anomaly'), ('b', 4, 8, 'Rare Event'),
                    ('c', 17, 19, 'Anomaly'), ('gap', 23, 25, 'Communication Gap')])
        ev = EventEvaluator(times, df)
        for ids in temporal_components(df):
            reduced = EventEvaluator(times, neutralize(df, ids))
            self.assertEqual(ev.nominal_ns, reduced.nominal_ns)
            for seed in range(25):
                pred = np.random.default_rng(seed).integers(0, 2, len(times))
                actual = without_targets(ev.score(pred), detection_ids(ev, pred), ids, ev.by_id)
                expected = reduced.score(pred)
                for key, value in actual.items():
                    self.assertAlmostEqual(value, expected[key], places=12)

    def test_no_remaining_targets_is_uninformative(self):
        self.assertIsNone(without_targets({}, [], ['a'], ['a']))
        with self.assertRaises(ValueError):
            neutralize(frame([('a', 0, 1, 'Anomaly')]), ['missing'])

    def test_selection_preserves_cap_and_earliest_epoch_tie(self):
        rows = pd.DataFrame([
            dict(epoch=e, threshold=t, merge_gap=0, min_dur=1,
                 val_esa_f05=f, val_pred_anomaly_rate=r, val_saturated=False,
                 val_false_positive_seconds=0.)
            for e in (1, 2) for t, f, r in [(1., .9, .1), (2., .5, .005), (3., 0., 0.)]])
        selected = pick(rows)
        self.assertEqual(selected['epoch'], 1)
        self.assertEqual(selected['threshold'], 2.)
        rows.loc[rows.threshold==2., 'val_esa_f05'] = 0.
        selected = pick(rows)
        self.assertEqual(selected['threshold'], 3.)
        self.assertEqual(selected['val_esa_f05'], 0.)

    def test_dependency_gate_rejects_changed_or_unrecorded_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'migration').mkdir()
            source = root/'scores.npy'
            source.write_bytes(b'original')
            record = {'path':'scores.npy', 'size':source.stat().st_size, 'sha256':digest(source)}
            (root/'migration/manifest.json').write_text(json.dumps({'files':[record]}))
            self.assertEqual(verify_recorded(root, [source])['scores.npy'], record['sha256'])
            source.write_bytes(b'changed!')
            with self.assertRaises(ValueError):
                verify_recorded(root, [source])
            new = root/'new.npy'
            new.write_bytes(b'new')
            with self.assertRaises(ValueError):
                verify_recorded(root, [new])


if __name__ == '__main__':
    unittest.main()
