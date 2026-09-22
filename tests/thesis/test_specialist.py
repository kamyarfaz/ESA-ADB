import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from esa_thesis.combine_specialist import select_combination, run_fold
from esa_thesis.development import development_spec
from esa_thesis.evaluation.esa import EventEvaluator


class SpecialistTests(unittest.TestCase):
    def evaluator(self):
        times = pd.date_range('2003-01-01', periods=10000, freq='30s', tz='UTC')
        labels = pd.DataFrame({'ID':['a','b'], 'StartTime':[times[100],times[1000]],
                               'EndTime':[times[110],times[1010]], 'Category':['Anomaly']*2})
        return EventEvaluator(times.asi8, labels)

    def test_specialist_channels_and_default_unchanged(self):
        self.assertEqual(development_spec('specialist')['features'], ['channel_14','channel_21','channel_29'])
        self.assertTrue(development_spec('specialist')['pseudo'])
        self.assertEqual(development_spec('baseline')['features'], [f'channel_{i}' for i in range(40,48)])

    def test_extra_event_enabled_but_false_alarm_branch_rejected(self):
        evaluator = self.evaluator()
        baseline = np.zeros(10000, dtype=bool); baseline[100:110] = True
        scores = np.zeros(10000);scores[1000:1010] = 10
        selected, grid = select_combination(baseline, scores, evaluator)
        self.assertTrue(selected['enabled'])
        self.assertEqual(selected['TPe'], 2)
        scores[5000:5010] = 10
        selected, grid = select_combination(baseline, scores, evaluator)
        self.assertFalse(selected['enabled'])
        self.assertTrue((grid.loc[grid.eligible,'FPe'] == 0).all())
        self.assertTrue((grid.loc[grid.eligible,'prediction_rate'] <= .01).all())

    def test_ties_disable_and_invalid_inputs_fail(self):
        evaluator = self.evaluator();baseline=np.zeros(10000,dtype=bool)
        selected, _ = select_combination(baseline,np.zeros(10000),evaluator)
        self.assertFalse(selected['enabled'])
        with self.assertRaises(ValueError):
            select_combination(baseline,np.full(10000,np.nan),evaluator)
        with self.assertRaises(ValueError):
            select_combination(np.ones(10000,dtype=bool),np.zeros(10000),evaluator)

    def test_assessment_arrays_only_read_after_rule_frozen(self):
        evaluator=self.evaluator()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);output=root/'combined'
            rule={'threshold':1.,'merge_gap':0,'min_dur':1}
            for name in ['base','specialist']:
                run=root/name;run.mkdir()
                (run/'model_checkpoint.pt').write_bytes(b'provenance')
                (run/'frozen_rule.json').write_text(json.dumps({'rule':rule}))
                (run/'results.json').write_text(json.dumps(evaluator.score(np.zeros(10000,dtype=bool))))
            def load_array(path):
                self.assertTrue((output/'frozen_rule.json').is_file())
                return np.zeros(10000,dtype=bool if 'prediction' in str(path) else float)
            with patch('esa_thesis.combine_specialist.validate_run'), \
                 patch('esa_thesis.combine_specialist.calibration_scores',return_value=(np.zeros(10000),evaluator.timestamps)), \
                 patch('esa_thesis.combine_specialist.evaluator_for',return_value=evaluator), \
                 patch('esa_thesis.combine_specialist.pd.date_range',return_value=pd.DatetimeIndex(evaluator.timestamps)), \
                 patch('esa_thesis.combine_specialist.np.load',side_effect=load_array):
                rows=run_fold('2003',root/'base',root/'specialist',output,device='cpu',batch_size=64)
            self.assertEqual(len(rows),3)
            self.assertEqual(rows[-1]['FPe'],0)


if __name__=='__main__':unittest.main()
