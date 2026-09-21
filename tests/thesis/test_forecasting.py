"""Causality, coverage, nominal targets and full frozen-rule workflow."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from esa_thesis import config
from esa_thesis.development import load_period
from esa_thesis.forecasting import CONTEXT, HORIZON, make_model, NominalForecastWindows, forecast_scores
from esa_thesis.forecast_development import MODELS, run_model


class ForecastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_persistence_alignment_and_partial_tail(self):
        values = np.arange(CONTEXT+HORIZON+3, dtype=np.float32)[:, None]
        score, covered = forecast_scores(make_model('persistence'), values)
        np.testing.assert_array_equal(score[:CONTEXT], 0)
        np.testing.assert_array_equal(covered[:CONTEXT], False)
        self.assertTrue(covered[CONTEXT:].all())
        np.testing.assert_array_equal(score[CONTEXT:CONTEXT+HORIZON], np.arange(1, HORIZON+1)**2)
        np.testing.assert_array_equal(score[-3:], [1, 4, 9])

    def test_future_values_cannot_change_past_scores_for_any_model(self):
        values = np.random.default_rng(12).normal(size=(CONTEXT+2*HORIZON+5, 2)).astype(np.float32)
        boundary = CONTEXT+HORIZON+2
        changed = values.copy(); changed[boundary:] += 50
        for name in MODELS:
            with self.subTest(model=name):
                model = make_model(name)
                first, _ = forecast_scores(model, values, batch_size=2)
                second, _ = forecast_scores(model, changed, batch_size=2)
                prefix, _ = forecast_scores(model, values[:boundary], batch_size=2)
                np.testing.assert_allclose(first[:boundary], second[:boundary], rtol=1e-6, atol=1e-6)
                np.testing.assert_allclose(first[:boundary], prefix, rtol=1e-5, atol=1e-5)

    def test_targets_must_also_be_nominal_and_no_fallback(self):
        values = np.zeros((300, 1), dtype=np.float32)
        labels = np.zeros(300); labels[280] = 1
        dataset = NominalForecastWindows(values, labels)
        np.testing.assert_array_equal(dataset.starts, [0])
        context, target = dataset[0]
        self.assertEqual(tuple(context.shape), (1, CONTEXT, 1))
        self.assertEqual(tuple(target.shape), (1, HORIZON, 1))
        with self.assertRaisesRegex(ValueError, 'No complete nominal'):
            NominalForecastWindows(values, np.ones(300))

    def test_residual_translation_equivariance_and_matched_initialization(self):
        torch.manual_seed(42)
        control = make_model('transformer')
        torch.manual_seed(42)
        model = make_model('transformer_residual').eval()
        for name, parameter in control.state_dict().items():
            torch.testing.assert_close(parameter, model.state_dict()[name], rtol=0, atol=0)
        x = torch.randn(2, 3, CONTEXT, 1)
        offset = torch.tensor([2., -3., 4.]).reshape(1, 3, 1, 1)
        with torch.no_grad():
            torch.testing.assert_close(model(x + offset), model(x) + offset, rtol=1e-5, atol=2e-6)

    def test_residual_zero_head_persists_and_detects_target_jump(self):
        model = make_model('transformer_residual')
        with torch.no_grad():
            model.head.weight.zero_()
            model.head.bias.zero_()
        values = np.full((CONTEXT + HORIZON, 2), 3., dtype=np.float32)
        values[CONTEXT:, 0] = 5.
        score, covered = forecast_scores(model, values)
        np.testing.assert_array_equal(score[covered], 4.)

    def test_all_models_freeze_before_assessment_without_test_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            times = pd.date_range('2000-01-01', periods=961, freq='30s')
            frame = pd.DataFrame({'timestamp': times, 'channel_1': np.sin(np.arange(961)/17), 'is_anomaly_channel_1': 0})
            frame.to_csv(root/'train.csv', index=False)
            pd.DataFrame({'ID': ['v', 'a'], 'Channel': ['channel_1']*2,
                          'StartTime': [times[600], times[920]], 'EndTime': [times[610], times[930]]}).to_csv(root/'labels.csv', index=False)
            pd.DataFrame({'ID': ['v', 'a'], 'Category': ['Anomaly']*2}).to_csv(root/'types.csv', index=False)
            boundaries = tuple(str(times[i]) for i in (320, 640, 960))
            for name in MODELS:
                output = root/name; output.mkdir()
                def checked_load(path, features, start, end):
                    self.assertEqual(path, root/'train.csv')
                    if start == boundaries[1]:
                        self.assertTrue((output/'frozen_rule.json').exists())
                    return load_period(path, features, start, end)
                with patch.multiple(config, TRAIN_FILE=root/'train.csv', TEST_FILE=root/'FORBIDDEN',
                                    ANNOTATIONS_FILE=root/'labels.csv', ANOMALY_TYPES_FILE=root/'types.csv'), \
                     patch('esa_thesis.forecast_development.load_period', side_effect=checked_load), \
                     contextlib.redirect_stdout(io.StringIO()):
                    result = run_model(name, ['channel_1'], boundaries, output, epochs=1,
                                       device='cpu', batch_size=8, seed=42)
                self.assertEqual(result['uncovered_assessment_samples'], CONTEXT)
                prediction = np.load(output/'assessment_prediction.npy')
                self.assertFalse(prediction[:CONTEXT].any())
                self.assertEqual(len(prediction), 320)
                frozen = json.loads((output/'frozen_rule.json').read_text())
                self.assertEqual(frozen['rule']['merge_gap'], 0)
                self.assertEqual(frozen['rule']['min_dur'], 1)
                self.assertEqual(result['epoch'], 0 if name == 'persistence' else 1)


if __name__ == '__main__': unittest.main()
