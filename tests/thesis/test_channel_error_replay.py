import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

spec=importlib.util.spec_from_file_location('channel_error_replay',Path(__file__).resolve().parents[2]/'scripts/research/channel_error_replay.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class ChannelReplayTests(unittest.TestCase):
    def test_closed_intervals_clipped_without_bridging(self):
        times=np.arange(10,dtype=np.int64)
        np.testing.assert_array_equal(module.interval_mask(times,[(-4,1),(5,6),(12,14)]),[1,1,0,0,0,1,1,0,0,0])
        self.assertEqual(module.tail_fraction(np.array([1,2,2,4]),2),.75)

    def test_each_channel_uses_its_own_interval(self):
        times=pd.date_range('2001-01-01',periods=10,freq='30s').asi8
        labels=pd.DataFrame(dict(ID=['a','a'],Channel=['channel_40','channel_41'],Category=['Anomaly','Anomaly'],StartTime=pd.to_datetime(times[[2,7]],utc=True),EndTime=pd.to_datetime(times[[3,8]],utc=True)))
        values=np.ones((8,10),dtype=np.float32)
        values[0,2:4]=5;values[0,7:9]=100;values[1,7:9]=0
        events,monthly=module.summarize(values,times,labels,2001,42,'validation')
        self.assertEqual(len(events),2)
        self.assertEqual(events[0]['peak_error'],5)
        self.assertEqual(events[1]['peak_error'],0)
        self.assertEqual(events[1]['median_change'],-1)
        self.assertEqual(events[0]['samples'],2)
        self.assertTrue(all(row['samples']==6 for row in monthly))

if __name__=='__main__':unittest.main()
