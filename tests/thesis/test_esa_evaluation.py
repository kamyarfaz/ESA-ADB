"""Run with: PYTHONPATH=. python tests/thesis/test_esa_evaluation.py"""
import contextlib
import hashlib
import importlib
import io
import sys
import types
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from esa_thesis.evaluation.esa import EventEvaluator

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_HASH = '89458fcaf3642633c804ecd4285b9a39d80d9b694785f6e9c6d62827af3ab940'


def reference_class():
    source = ROOT / 'timeeval/metrics/ESA_ADB_metrics.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == REFERENCE_HASH
    package = types.ModuleType('_esa_reference')
    package.__path__ = [str(source.parent)]
    sys.modules[package.__name__] = package
    return importlib.import_module('_esa_reference.ESA_ADB_metrics').ESAScores


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = reference_class()

    def labels(self):
        return pd.DataFrame([
            ('a', 3, 5, 'Anomaly'), ('a', 8, 9, 'Anomaly'),
            ('b', 4, 7, 'Rare Event'), ('c', 15, 15, 'Anomaly'),
            ('gap', 20, 23, 'Communication Gap')],
            columns=['ID', 'StartTime', 'EndTime', 'Category']).assign(
                StartTime=lambda d: pd.to_datetime(d.StartTime, unit='s'),
                EndTime=lambda d: pd.to_datetime(d.EndTime, unit='s'))

    def assert_parity(self, timestamps, labels, pred, categories=('Anomaly', 'Rare Event')):
        evaluator = EventEvaluator(timestamps, labels, categories)
        vector = np.array(list(zip(pd.to_datetime(timestamps), pred)), dtype=object)
        with contextlib.redirect_stdout(io.StringIO()):
            ref = self.reference(betas=.5, select_labels={'Category': list(categories)},
                                 full_range=(pd.Timestamp(timestamps[0]), pd.Timestamp(timestamps[-1]))).score(labels, vector)
        actual = evaluator.score(pred)
        for key in ('EW_F_0.50', 'EW_precision', 'EW_recall', 'alarming_precision'):
            self.assertAlmostEqual(actual[key], ref[key], places=12, msg=key)

    def test_reference_random_predictions_regular_and_irregular_time(self):
        labels = self.labels()
        for seed in range(32):
            rng = np.random.default_rng(seed)
            times = pd.to_datetime(np.arange(31), unit='s').asi8
            if seed % 2:
                times = np.r_[times[0], times[1:-1:2], times[-1]]
            pred = (rng.random(len(times)) < .3).astype(np.int8)
            pred[0] = 1  # ensure at least one prediction for reference affiliation code
            self.assert_parity(times, labels, pred)

    def test_all_positive_zero_score(self):
        times = pd.to_datetime(np.arange(31), unit='s').asi8
        self.assert_parity(times, self.labels(), np.ones(31))
        self.assertEqual(EventEvaluator(times, self.labels()).score(np.ones(31))['EW_F_0.50'], 0)

    def test_no_predictions_and_no_events(self):
        times = pd.to_datetime(np.arange(31), unit='s').asi8
        ev = EventEvaluator(times, self.labels())
        self.assertEqual(ev.score(np.zeros(31))['FNe'], 3)
        empty = EventEvaluator(times, self.labels().iloc[:0])
        self.assertEqual(empty.score(np.ones(31))['FPe'], 1)
        self.assertEqual(empty.score(np.zeros(31))['EW_F_0.50'], 0)

    def test_half_open_boundary_and_last_point(self):
        times = pd.to_datetime(np.arange(31), unit='s').asi8
        labels = pd.DataFrame({'ID':['a'], 'StartTime':[pd.Timestamp(3, unit='s')],
                               'EndTime':[pd.Timestamp(3, unit='s')], 'Category':['Anomaly']})
        pred = np.zeros(31); pred[2] = 1
        self.assertEqual(EventEvaluator(times, labels).score(pred)['TPe'], 0)
        pred[3] = 1
        self.assertEqual(EventEvaluator(times, labels).score(pred)['TPe'], 1)
        labels[['StartTime','EndTime']] = pd.Timestamp(30, unit='s')
        pred[:] = 0; pred[-1] = 1
        self.assert_parity(times, labels, pred)

    def test_category_neutrality(self):
        times = pd.to_datetime(np.arange(31), unit='s').asi8
        pred = np.zeros(31); pred[20:23] = 1
        score = EventEvaluator(times, self.labels()).score(pred)
        self.assertEqual(score['FPe'], 0)
        self.assertEqual(score['false_positive_seconds'], 0)
        self.assert_parity(times, self.labels(), pred)

    def test_clipping_and_invalid_inputs(self):
        times = pd.to_datetime(np.arange(4, 10), unit='s').asi8
        ev = EventEvaluator(times, self.labels())
        self.assertEqual(len(ev.by_id), 2)
        with self.assertRaises(ValueError): ev.score(np.zeros(3))
        with self.assertRaises(ValueError): ev.score(np.full(6, .5))
        with self.assertRaises(ValueError): EventEvaluator(times[::-1], self.labels())

    def test_perfect_and_zero_nominal_policy(self):
        times = pd.to_datetime(np.arange(31), unit='s').asi8
        labels = self.labels().iloc[[0]].copy()
        pred = np.zeros(31); pred[3:5] = 1
        self.assertEqual(EventEvaluator(times, labels).score(pred)['EW_F_0.50'], 1)
        labels.StartTime = pd.Timestamp(times[0]); labels.EndTime = pd.Timestamp(times[-1])
        self.assertEqual(EventEvaluator(times, labels).score(np.ones(31))['EW_F_0.50'], 1)


if __name__ == '__main__':
    unittest.main()
