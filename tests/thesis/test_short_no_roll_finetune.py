import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from esa_thesis import short_no_roll_finetune as ft

class TinyAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale=torch.nn.Parameter(torch.ones(8)*.5)
        self.drop=torch.nn.Dropout(.2)
    def forward(self,x):
        return self.drop(x)*self.scale[None,:,None,None]

class FineTuneTests(unittest.TestCase):
    def test_resume_and_original_fallback(self):
        x=np.random.default_rng(7).normal(size=(288,8)).astype(np.float32)
        y=np.zeros(len(x),np.int8)
        def load(*args):return x.copy(),y.copy(),np.arange(len(x))
        def initial(*args):return TinyAE(),ft.RobustChannelScaler().fit(x,y),ft.FEATURES,False
        def rule(*args):
            r=dict(val_esa_f05=0.,threshold=1.,merge_gap=0,min_dur=1)
            return r,pd.DataFrame([r])
        with tempfile.TemporaryDirectory() as tmp,patch.object(ft,'EPOCHS',1),patch.object(ft,'BATCH',1),patch.object(ft,'load_split',side_effect=load),patch.object(ft,'load_model',side_effect=initial),patch.object(ft,'score_windows',return_value=np.ones(288)),patch.object(ft,'choose_rule',side_effect=rule):
            a,b=Path(tmp)/'a',Path(tmp)/'b'
            ft.train(a,'cpu')
            save=ft.atomic_save
            def interrupted(value,path):
                save(value,path)
                if path.name=='last.pt':raise RuntimeError('interrupt')
            with patch.object(ft,'atomic_save',side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError,'interrupt'):ft.train(b,'cpu')
            ft.train(b,'cpu')
            sa=torch.load(a/'last.pt',weights_only=False)
            sb=torch.load(b/'last.pt',weights_only=False)
            for k,v in sa['model'].items():self.assertTrue(torch.equal(v,sb['model'][k]))
            self.assertEqual(sa['history'],sb['history'])
            self.assertEqual(sa['scheduler'],sb['scheduler'])
            self.assertTrue(torch.equal(sa['rng']['torch'],sb['rng']['torch']))
            self.assertEqual(sb['best']['epoch'],0)
            selected=torch.load(b/'model.pt',weights_only=False)
            self.assertTrue(torch.equal(selected['model_state_dict']['scale'],torch.ones(8)*.5))
            self.assertTrue(ft.verify_receipt(b,'trained.json'))

if __name__=='__main__':unittest.main()
