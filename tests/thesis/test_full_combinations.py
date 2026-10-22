import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from esa_thesis import full_combinations as fc

class TinyAE(torch.nn.Module):
    def __init__(self,channels):
        super().__init__();self.scale=torch.nn.Parameter(torch.ones(channels)*.5);self.drop=torch.nn.Dropout(.2)
    def forward(self,x):return self.drop(x)*self.scale[None,:,None,None]

class CombinationTests(unittest.TestCase):
    def test_epoch_order_and_nominal_windows(self):
        y=np.zeros(1024,np.int8);y[500]=1
        starts=fc.nominal_starts(y)
        self.assertTrue(all(not y[s:s+fc.config.SEQ_LEN].any() for s in starts))
        np.testing.assert_array_equal(fc.epoch_order(starts,4),fc.epoch_order(starts,4))
        self.assertEqual(set(starts),set(fc.epoch_order(starts,4)))

    def test_fusion_endpoints_and_validation(self):
        a=np.array([1.,3.]);b=np.array([8.,2.])
        np.testing.assert_array_equal(fc.fusion_score(a,b,0,2,4),a/2)
        np.testing.assert_array_equal(fc.fusion_score(a,b,1,2,4),b/4)
        np.testing.assert_allclose(fc.fusion_score(a,b,.5,2,4),.5*a/2+.5*b/4)
        with self.assertRaises(ValueError):fc.fusion_score(a,b,0,0,1)

    def test_unmask_transfers_all_shared_weights(self):
        model=fc.MaskedAE(8)
        result=fc.unmask(model,8,'cpu')
        self.assertNotIn('mask_token',result.state_dict())
        for k,v in result.state_dict().items():self.assertTrue(torch.equal(v,model.state_dict()[k]),k)

    def test_exact_mid_epoch_resume(self):
        x=np.random.default_rng(7).normal(size=(288,11)).astype(np.float32)
        def load(*args):return x.copy(),np.zeros(len(x),np.int8),np.arange(len(x))
        def scoring(model,*args):return np.array([float(model.scale.detach().mean())])
        def rule(score,*args):
            row=dict(val_esa_f05=float(score[0]),threshold=1.,merge_gap=0,min_dur=1)
            return row,pd.DataFrame([row])
        with tempfile.TemporaryDirectory() as tmp, patch.object(fc,'EPOCHS',5),patch.object(fc,'BATCH',1),patch.object(fc,'load_split',side_effect=load),patch.object(fc,'MultivariateAE',TinyAE),patch.object(fc,'score_windows',side_effect=scoring),patch.object(fc,'choose_rule',side_effect=rule):
            a,b=Path(tmp)/'a',Path(tmp)/'b'
            fc.train('expanded_short_no_roll',a,'cpu')
            original=fc.atomic_save;tripped=[False]
            def fail_after_save(value,path):
                original(value,path)
                if path.name=='last.pt' and value['epoch']==1 and not tripped[0]:
                    tripped[0]=True;raise RuntimeError('simulated interruption')
            with patch.object(fc,'atomic_save',side_effect=fail_after_save):
                with self.assertRaisesRegex(RuntimeError,'simulated interruption'):fc.train('expanded_short_no_roll',b,'cpu')
            fc.train('expanded_short_no_roll',b,'cpu')
            sa=torch.load(a/'last.pt',weights_only=False);sb=torch.load(b/'last.pt',weights_only=False)
            for k,v in sa['model'].items():self.assertTrue(torch.equal(v,sb['model'][k]),k)
            self.assertEqual(sa['history'],sb['history'])
            self.assertEqual(sa['scheduler'],sb['scheduler'])
            self.assertTrue(torch.equal(sa['rng']['torch'],sb['rng']['torch']))
            self.assertTrue(fc.verify_receipt(b,'trained.json'))

if __name__=='__main__':unittest.main()
