"""Regression checks for validation-only selection and incompatible-cache rejection."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from esa_thesis.evaluation.esa import EventEvaluator
from esa_thesis.metrics import metrics, legacy_metrics
from esa_thesis.protocol import guard_run, validation_split, read_timestamps
from esa_thesis.thresholds import sweep_thresholds, select_rows


class ProtocolTests(unittest.TestCase):
    def test_timestamp_only_reader_reports_invalid_feature_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.csv'
            path.write_bytes(b'timestamp,channel_1\n2000-01-01,1\n2000-01-02,\xb1\n')
            with self.assertWarns(RuntimeWarning):
                times = read_timestamps(path)
            self.assertEqual(len(times), 2)
            bad = Path(directory) / 'bad.csv'
            bad.write_bytes(b'timestamp,channel_1\nnot-a-date,1\n2000-01-02,1\n')
            with self.assertRaises(ValueError):
                read_timestamps(bad)

    def test_selection_ignores_test_columns(self):
        frame = pd.DataFrame([
            dict(threshold=1.,merge_gap=0,min_dur=1,val_esa_f05=.8,val_false_positive_seconds=1.,val_pred_anomaly_rate=.005,val_saturated=False,test_esa_f05=.1),
            dict(threshold=2.,merge_gap=0,min_dur=1,val_esa_f05=.7,val_false_positive_seconds=0.,val_pred_anomaly_rate=.004,val_saturated=False,test_esa_f05=.99)])
        chosen = select_rows(frame).iloc[0]
        self.assertEqual(chosen.threshold, 1.)
        frame.test_esa_f05 = frame.test_esa_f05.iloc[::-1].to_numpy()
        self.assertEqual(select_rows(frame).iloc[0].threshold, 1.)

    def test_zero_candidate_and_no_test_sweep(self):
        times = pd.date_range('2000-01-01', periods=1000, freq='s').asi8
        labels = pd.DataFrame({'ID':['a'], 'StartTime':[pd.Timestamp(times[100])],
                               'EndTime':[pd.Timestamp(times[105])], 'Category':['Anomaly']})
        ev = EventEvaluator(times, labels)
        scores = np.ones(1000)
        grid = sweep_thresholds(scores,evaluator=ev,threshold_values=[0],merge_gaps=[0],min_durs=[1])
        self.assertFalse(any(c.startswith('test_') for c in grid))
        self.assertEqual(select_rows(grid).iloc[0].val_pred_anomaly_rate,0)
        with self.assertRaises(ValueError):
            sweep_thresholds(np.full(1000,np.nan),evaluator=ev)

    def test_short_event_filter_can_remove_saturation(self):
        times = pd.date_range('2000-01-01', periods=1000, freq='s').asi8
        labels = pd.DataFrame({'ID':['a'], 'StartTime':[pd.Timestamp(times[100])],
                               'EndTime':[pd.Timestamp(times[105])], 'Category':['Anomaly']})
        scores = np.tile([1,1,1,0],250)
        grid = sweep_thresholds(scores,evaluator=EventEvaluator(times,labels),threshold_values=[.5],merge_gaps=[0],min_durs=[1,4])
        self.assertEqual(grid.loc[(grid.threshold==.5)&(grid.min_dur==4),'val_pred_anomaly_rate'].iloc[0],0)

    def test_calendar_split(self):
        times = pd.date_range('2000-01-01','2001-01-01',freq='h').asi8
        split=validation_split(times)
        self.assertEqual(pd.Timestamp(times[split]),pd.Timestamp('2000-10-01'))

    def test_guard_rejects_legacy_and_changed_protocol(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)
            (path/'checkpoint.pt').write_text('legacy')
            with self.assertRaises(ValueError):guard_run(path,{'version':1})
            (path/'checkpoint.pt').unlink()
            guard_run(path,{'version':1})
            guard_run(path,{'version':1})
            with self.assertRaises(ValueError):guard_run(path,{'version':2})

    def test_metrics_require_annotations_and_prefix_legacy(self):
        y=np.zeros(1000,dtype=np.int8);y[10:188:2]=1
        with self.assertRaises(TypeError):metrics(y,np.ones_like(y))
        times=pd.date_range('2000-01-01',periods=1000,freq='s').asi8
        labels=pd.DataFrame([{'ID':str(i),'StartTime':pd.Timestamp(times[t]),'EndTime':pd.Timestamp(times[t+1]),'Category':'Anomaly'} for i,t in enumerate(np.flatnonzero(y))])
        score=metrics(y,np.ones_like(y),evaluator=EventEvaluator(times,labels))
        self.assertEqual(score['esa_f05'],0)
        self.assertGreater(score['legacy_esa_f05'],.99)


if __name__=='__main__':unittest.main()
