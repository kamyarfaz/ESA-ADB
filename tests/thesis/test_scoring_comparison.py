"""Analytic localization checks and a complete synthetic comparison run."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from esa_thesis import config, scoring
from esa_thesis.models import MultivariateAE
from esa_thesis.evaluation.compare_scoring import paired_scores, main


class ZeroModel(torch.nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


class ScoringComparisonTests(unittest.TestCase):
    def test_single_spike_localization(self):
        x = np.zeros((256,1),dtype=np.float32);x[128,0]=16
        scores, uncovered = paired_scores(ZeroModel(), x)
        np.testing.assert_array_equal(scores['window_mean'],np.ones(256))
        self.assertEqual(scores['per_timestep'][128],256)
        self.assertEqual(np.count_nonzero(scores['per_timestep']),1)
        self.assertEqual(uncovered,0)

    def test_overlap_matches_legacy_and_reports_tail(self):
        x=np.random.default_rng(0).normal(size=(329,2)).astype(np.float32)
        scores,uncovered=paired_scores(ZeroModel(),x,batch_size=2)
        old_device=scoring.DEVICE
        try:
            scoring.DEVICE='cpu'
            _,legacy=scoring.score_series(ZeroModel(),x)
        finally:
            scoring.DEVICE=old_device
        np.testing.assert_allclose(scores['window_mean'],legacy,rtol=1e-6)
        np.testing.assert_allclose(scores['per_timestep'][:320],(x[:320]**2).max(axis=1),rtol=1e-6)
        np.testing.assert_array_equal(scores['per_timestep'][320:],np.zeros(9))
        self.assertEqual(uncovered,9)

    def test_complete_comparison_freezes_validation_choice(self):
        torch.set_num_threads(2)
        names=['TRAIN_FILE','TEST_FILE','ANNOTATIONS_FILE','ANOMALY_TYPES_FILE']
        previous={n:getattr(config,n) for n in names};argv=sys.argv
        try:
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);source=root/'source';source.mkdir()
                train=pd.date_range('2000-01-01',periods=4000,freq='h')
                test=pd.date_range(train[-1]+pd.Timedelta(hours=1),periods=320,freq='h')
                for name,times in [('train',train),('test',test)]:
                    pd.DataFrame({'timestamp':times,'channel_1':np.sin(np.arange(len(times))/17)}).to_csv(root/(name+'.csv'),index=False)
                pd.DataFrame({'ID':['v','t'],'Channel':['channel_1']*2,
                              'StartTime':[train[3000],test[100]],'EndTime':[train[3010],test[105]]}).to_csv(root/'labels.csv',index=False)
                pd.DataFrame({'ID':['v','t'],'Category':['Anomaly','Rare Event']}).to_csv(root/'types.csv',index=False)
                for name,filename in zip(names,['train.csv','test.csv','labels.csv','types.csv']):setattr(config,name,root/filename)
                torch.save({'features':['channel_1'],'model_state_dict':MultivariateAE(1).state_dict(),
                            'scaler_median':np.array([0],dtype=np.float32),'scaler_iqr':np.array([1],dtype=np.float32)},source/'model_checkpoint.pt')
                sys.argv=['compare-scoring','--source-run',str(source),'--output',str(root/'out'),'--device','cpu']
                with contextlib.redirect_stdout(io.StringIO()):main()
                rows=pd.read_csv(root/'out/results.csv')
                self.assertEqual(set(rows['mode']),{'window_mean','per_timestep'})
                self.assertEqual(int(rows.validation_preferred.sum()),1)
                rules=json.loads((root/'out/frozen_rules.json').read_text())
                for mode in rules['rules']:
                    grid=pd.read_csv(root/'out'/f'{mode}_validation_candidates.csv')
                    self.assertFalse(any(c.startswith('test_') for c in grid))
        finally:
            sys.argv=argv
            for name,value in previous.items():setattr(config,name,value)


if __name__=='__main__':unittest.main()
