"""Scientific preparation boundaries for the approved cadence amendment."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from esa_thesis import cross_mission_study as old
from esa_thesis import cross_mission_study_v2 as new
from esa_thesis.mission2_study import resample as original_resample


class CadenceRevisionTests(unittest.TestCase):
    def test_only_two_mission1_channels_change(self):
        for mission in (1, 2):
            for n in range(1, 101):
                self.assertEqual(new.maximum_age_seconds(mission, f'channel_{n}'),
                                 900 if mission == 1 and n in (62, 63) else 600)

    def test_finite_values_age_boundaries_and_split_containment(self):
        rt = np.array([-60, 30, 2700, 3600, 5000], np.int64) * 10**9
        rv = np.array([999, 2, np.nan, 7, 999], np.float32)
        q = np.array([0, 30, 630, 631, 930, 931, 2700, 3000, 3600, 4200, 4501], np.int64) * 10**9
        for ch in ('channel_62', 'channel_63'):
            x, av, age, _ = new.resample(rt, rv, q, 0, 5000*10**9, 1, ch)
            np.testing.assert_array_equal(av, [False, True, True, True, True, False, False, False, True, True, False])
            self.assertTrue(np.isnan(x[0]))
            self.assertEqual(x[4], 2)
            self.assertEqual(age[4], 900*10**9)
            self.assertTrue(np.isnan(x[6]))

    def test_all_unamended_paths_equal_original_sampler(self):
        rt = np.array([-100, 10, 190, 240, 1300, 2000], np.int64)*10**9
        rv = np.array([3, 4, np.inf, 8, 9, 12], np.float32)
        q = np.arange(0, 2000, 30, dtype=np.int64)*10**9
        expected = original_resample(rt, rv, q, 0, 2000*10**9)
        for mission, ch in [(1, 'channel_61'), (1, 'channel_40'), (2, 'channel_62'), (2, 'channel_63'), (2, 'channel_58')]:
            actual = new.resample(rt, rv, q, 0, 2000*10**9, mission, ch)
            for a, b in zip(actual, expected):
                np.testing.assert_array_equal(a, b)

    def test_fifteen_minute_stream_supports_contiguous_windows(self):
        q = np.arange(600, dtype=np.int64)*30*10**9
        rt = np.arange(0, 18000, 900, dtype=np.int64)*10**9
        rv = np.arange(len(rt), dtype=np.float32)
        old_av = original_resample(rt, rv, q, 0, 18000*10**9)[1]
        new_av = new.resample(rt, rv, q, 0, 18000*10**9, 1, 'channel_62')[1]
        self.assertFalse(any(old_av[i:i+256].all() for i in range(0, 345, 16)))
        self.assertTrue(new_av.all())

    def test_same_values_in_every_split_and_held_anomaly_excluded_only_in_training(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); raw = root/'data/ESA-Mission1'; (raw/'channels').mkdir(parents=True)
            t = pd.date_range('2000-01-01', periods=6, freq='900s')
            pd.DataFrame({'value':np.arange(6, dtype=np.float32)}, index=t).to_pickle(raw/'channels/channel_62.zip', compression='zip')
            pd.DataFrame(dict(ID=['a'], Channel=['channel_62'], StartTime=[str(t[1])], EndTime=[str(t[1])])).to_csv(raw/'labels.csv', index=False)
            pd.DataFrame(dict(ID=['a'], Category=['Anomaly'])).to_csv(raw/'anomaly_types.csv', index=False)
            bounds=(str(t[0]), str(t[-1]+pd.Timedelta(seconds=900)))
            splits={1:{s:bounds for s in ('train','calibration','assessment')}}
            with patch.object(new,'ROOT',root), patch.object(new,'PREP',root/'prepared'), patch.object(new,'SPLITS',splits):
                for split in splits[1]:new.prepare_split('m1',1,split,['channel_62'],30)
                train=root/'prepared/m1/train'
                self.assertEqual(json.loads((train/'metadata.json').read_text())['max_age_seconds'],[900])
                x=np.load(train/'values.npy')
                np.testing.assert_array_equal(np.flatnonzero(np.load(train/'excluded.npy')),np.arange(30,60))
                for split in ('calibration','assessment'):
                    np.testing.assert_array_equal(x,np.load(root/f'prepared/m1/{split}/values.npy'))
                    self.assertFalse(np.load(root/f'prepared/m1/{split}/excluded.npy').any())

    def test_job_design_and_splits_unchanged(self):
        self.assertEqual(new.SPLITS,old.SPLITS)
        self.assertEqual(new.BATCH,old.BATCH)
        sel={'panels':{str(m):{'base':['a']*13,'expanded':['a']*(25 if m==1 else 26)} for m in (1,2)}}
        self.assertEqual(new.jobs(sel),old.jobs(sel))
        self.assertEqual(len(new.jobs(sel)),18)


if __name__=='__main__':unittest.main()
