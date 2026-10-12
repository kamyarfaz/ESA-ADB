from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from esa_thesis import config,inner_validation
from esa_thesis.data import RobustChannelScaler
from esa_thesis.loss_weight_comparison import penalty_weight


class LossWeightTests(unittest.TestCase):
    def test_restores_default_on_failure(self):
        old=config.PA_LAMBDA
        with self.assertRaises(RuntimeError):
            with penalty_weight(.1):
                self.assertEqual(config.PA_LAMBDA,.1)
                raise RuntimeError('interrupted')
        self.assertEqual(config.PA_LAMBDA,old)

    def test_actual_first_batch_changes_only_loss_multiplier(self):
        torch.set_num_threads(1)
        x=np.random.default_rng(5).normal(size=(272,8)).astype(np.float32)
        scaler=RobustChannelScaler().fit(x,np.zeros(272))
        times=pd.date_range('2001-01-01',periods=272,freq='30s').asi8
        args=SimpleNamespace(seed=42,device='cpu',epochs=1,batch_size=2,resume=False)
        rows=[]
        with tempfile.TemporaryDirectory() as td:
            for weight in [1.,.1]:
                out=Path(td)/str(weight)
                with penalty_weight(weight),patch.object(inner_validation,'score',return_value=np.zeros(272)):
                    inner_validation.train(x,np.array([0,16]),x,times,scaler,out,args)
                rows.append(pd.read_csv(out/'history.csv').iloc[0])
        self.assertEqual(rows[0].normal_mse,rows[1].normal_mse)
        self.assertEqual(rows[0].pseudo_penalty,rows[1].pseudo_penalty)
        for row,weight in zip(rows,[1.,.1]):
            self.assertAlmostEqual(row.loss,row.normal_mse+weight*row.pseudo_penalty,places=5)
        self.assertNotEqual(rows[0].loss,rows[1].loss)


if __name__=='__main__':unittest.main()
