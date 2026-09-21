"""Chronological isolation and complete development-run checks."""
import contextlib
import io
import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from esa_thesis import config
from esa_thesis.development import FOLDS, load_period, run_fold, main, development_spec


class DevelopmentTests(unittest.TestCase):
    def test_channel_choice_is_explicit_and_preserves_default(self):
        baseline = development_spec('baseline')
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'not_created'
            for choice in ('baseline', 'expanded'):
                stream = io.StringIO()
                with patch.object(sys, 'argv', ['develop', '--channel-set', choice,
                                               '--output', str(output), '--dry-run']), \
                     contextlib.redirect_stdout(stream):
                    main()
                plan = json.loads(stream.getvalue())
                expected = baseline['features'] + (['channel_14', 'channel_21', 'channel_29'] if choice == 'expanded' else [])
                self.assertEqual(plan['spec']['features'], expected)
                self.assertTrue(plan['spec']['pseudo'])
                self.assertFalse(output.exists())
        self.assertEqual(development_spec('baseline'), baseline)

    def test_reconstruction_cli_and_resume_objective_guard(self):
        from esa_thesis.development_recovery import check_protocol
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'out'
            stream = io.StringIO()
            with patch.object(sys, 'argv', ['develop', '--objective', 'reconstruction',
                                           '--output', str(output), '--dry-run']), \
                 contextlib.redirect_stdout(stream):
                main()
            plan = json.loads(stream.getvalue())
            self.assertFalse(plan['spec']['pseudo'])
            self.assertEqual(plan['spec']['features'], development_spec('baseline')['features'])
            self.assertFalse(output.exists())
            check_protocol(output, plan, False)
            changed = {**plan, 'spec': development_spec('baseline')}
            with self.assertRaisesRegex(ValueError, 'protocol differs'):
                check_protocol(output, changed, True)

    def test_predeclared_folds_do_not_touch_final_validation(self):
        for train_end, cal_end, assess_end in FOLDS.values():
            self.assertLess(train_end, cal_end)
            self.assertLess(cal_end, assess_end)
            self.assertLess(assess_end, '2006-10-01')

    def test_period_boundaries_and_nonfinite_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'input.csv'
            times = pd.date_range('2000-01-01', periods=960, freq='30s')
            frame = pd.DataFrame({'timestamp': times, 'channel_1': np.arange(960), 'is_anomaly_channel_1': 0})
            frame.to_csv(path, index=False)
            x, _, observed = load_period(path, ['channel_1'], str(times[320]), str(times[640]))
            self.assertEqual(len(x), 320)
            self.assertEqual(x[0, 0], 320)
            self.assertEqual(x[-1, 0], 639)
            frame.loc[321, 'channel_1'] = np.nan
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'Invalid features'):
                load_period(path, ['channel_1'], str(times[320]), str(times[640]))

    def test_training_freezes_before_assessment_without_test_file(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            times = pd.date_range('2000-01-01', periods=961, freq='30s')
            frame = pd.DataFrame({'timestamp': times, 'channel_1': np.sin(np.arange(961)/17), 'is_anomaly_channel_1': 0})
            frame.to_csv(root/'train.csv', index=False)
            pd.DataFrame({'ID': ['v', 'a'], 'Channel': ['channel_1']*2,
                          'StartTime': [times[400], times[700]], 'EndTime': [times[410], times[710]]}).to_csv(root/'labels.csv', index=False)
            pd.DataFrame({'ID': ['v', 'a'], 'Category': ['Anomaly']*2}).to_csv(root/'types.csv', index=False)
            output = root/'out'; output.mkdir()
            boundaries = tuple(str(times[i]) for i in (320, 640, 960))
            real_load = load_period
            def checked_load(path, features, start, end):
                if start == boundaries[1]:
                    self.assertTrue((output/'frozen_rule.json').is_file())
                return real_load(path, features, start, end)
            with patch.multiple(config, TRAIN_FILE=root/'train.csv', TEST_FILE=root/'DOES_NOT_EXIST',
                                ANNOTATIONS_FILE=root/'labels.csv', ANOMALY_TYPES_FILE=root/'types.csv'), \
                 patch('esa_thesis.development.load_period', side_effect=checked_load):
                result = run_fold({'features': ['channel_1'], 'pseudo': True}, boundaries, output,
                                  epochs=1, device='cpu', batch_size=8, seed=42)
                output = root/'reconstruction'; output.mkdir()
                with patch('esa_thesis.development.make_pseudo_anomaly',
                           side_effect=AssertionError('Reconstruction must not generate pseudo anomalies')):
                    result = run_fold({'features': ['channel_1'], 'pseudo': False}, boundaries, output,
                                      epochs=1, device='cpu', batch_size=8, seed=42)
            self.assertEqual(result['epoch'], 1)
            self.assertEqual(len(np.load(output/'assessment_prediction.npy')), 320)
            self.assertEqual(json.loads((output/'frozen_rule.json').read_text())['epoch'], 1)


if __name__ == '__main__':
    unittest.main()
