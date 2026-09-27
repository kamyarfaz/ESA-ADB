import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from esa_thesis import config
from esa_thesis.data import NormalWindows
from esa_thesis.development import load_period, run_fold
from esa_thesis.long_baseline import main, verify_inputs


class LongBaselineTests(unittest.TestCase):
    def test_uncapped_windows_preserve_all_eligible_starts(self):
        x = np.zeros((1024,1), dtype=np.float32)
        y = np.zeros(1024, dtype=np.int8); y[400] = 1
        full = NormalWindows(x,y,max_windows=0)
        capped = NormalWindows(x,y,max_windows=2)
        expected = [s for s in range(0,1024-256+1,16) if not y[s:s+256].any()]
        np.testing.assert_array_equal(full.starts,expected)
        self.assertEqual(len(capped),2)
        self.assertTrue(set(capped.starts).issubset(full.starts))

    def test_plan_is_fixed_and_dry_run_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'new'; stream=io.StringIO()
            with patch.object(sys,'argv',['long-baseline','--output',str(out),'--dry-run']), contextlib.redirect_stdout(stream):
                main()
            plan=json.loads(stream.getvalue())
            self.assertEqual(plan['train'],['2000-01-01','2006-01-01'])
            self.assertEqual(plan['calibration'],['2006-01-01','2007-01-01'])
            self.assertEqual(plan['assessment'],['2007-01-01','2014-01-01'])
            self.assertEqual(plan['max_train_windows'],0)
            self.assertFalse(out.exists())

    def test_wrong_checksum_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            f=Path(directory)/'bad';f.write_text('bad')
            with patch.object(config,'TRAIN_FILE',f), self.assertRaisesRegex(ValueError,'checksum differs'):
                verify_inputs()

    def test_separate_test_source_is_read_only_after_freezing(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'out';out.mkdir()
            times=pd.date_range('2000-01-01',periods=961,freq='30s')
            frame=pd.DataFrame(dict(timestamp=times,channel_1=np.sin(np.arange(961)/17),is_anomaly_channel_1=0))
            frame.iloc[:640].to_csv(root/'train.csv',index=False)
            frame.iloc[640:].to_csv(root/'test.csv',index=False)
            pd.DataFrame(dict(ID=['v','a'],Channel=['channel_1']*2,StartTime=[times[400],times[700]],EndTime=[times[410],times[710]])).to_csv(root/'labels.csv',index=False)
            pd.DataFrame(dict(ID=['v','a'],Category=['Anomaly']*2)).to_csv(root/'types.csv',index=False)
            seen=[]
            def checked(path,features,start,end):
                seen.append(path)
                if path==root/'test.csv':
                    self.assertTrue((out/'frozen_rule.json').exists())
                return load_period(path,features,start,end)
            with patch.multiple(config,TRAIN_FILE=root/'train.csv',ANNOTATIONS_FILE=root/'labels.csv',ANOMALY_TYPES_FILE=root/'types.csv'), patch('esa_thesis.development.load_period',side_effect=checked):
                run_fold(dict(features=['channel_1'],pseudo=False),tuple(str(times[i]) for i in (320,640,960)),out,epochs=1,device='cpu',batch_size=8,seed=42,assessment_path=root/'test.csv',max_windows=0)
            self.assertEqual(seen,[root/'train.csv',root/'train.csv',root/'test.csv'])
            self.assertEqual(len(np.load(out/'assessment_prediction.npy')),320)
