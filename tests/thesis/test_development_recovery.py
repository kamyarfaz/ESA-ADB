import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from esa_thesis import config
from esa_thesis.development import run_fold
from esa_thesis.development_recovery import atomic_save, prepare_fold, check_protocol


class RecoveryTests(unittest.TestCase):
    def test_partial_old_fold_is_preserved_not_warm_started(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); fold = root/'2005'; fold.mkdir()
            (fold/'model_checkpoint.pt').write_bytes(b'old selected weights')
            with self.assertRaisesRegex(ValueError, 'no full epoch'):
                prepare_fold(fold, True, False)
            prepare_fold(fold, True, True)
            archived = list((root/'interrupted').glob('2005_*'))
            self.assertEqual(len(archived), 1)
            self.assertEqual((archived[0]/'model_checkpoint.pt').read_bytes(), b'old selected weights')
            self.assertFalse(any(fold.iterdir()))

    def test_completed_fold_is_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            fold=Path(directory)
            for name in ['model_checkpoint.pt','calibration_candidates.csv','history.csv','frozen_rule.json','assessment_score.npy','assessment_prediction.npy']:
                (fold/name).touch()
            (fold/'results.json').write_text('{"epoch": 25}')
            self.assertEqual(prepare_fold(fold, True, False), {'epoch':25})

    def test_protocol_changes_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'out'
            expected={'epochs':40,'source_sha256':{'development.py':'new','models.py':'stable'}}
            check_protocol(output, expected, False)
            check_protocol(output, expected, True)
            with self.assertRaisesRegex(ValueError,'protocol differs'):
                check_protocol(output, {**expected,'epochs':41}, True)

    def test_interrupted_checkpoint_write_keeps_previous(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'last.pt';atomic_save({'epoch':1},path)
            with patch('esa_thesis.development_recovery.torch.save', side_effect=RuntimeError('disk failure')):
                with self.assertRaises(RuntimeError):atomic_save({'epoch':2},path)
            self.assertEqual(torch.load(path, weights_only=False)['epoch'],1)

    def test_epoch_resume_matches_uninterrupted_training(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);times=pd.date_range('2000-01-01',periods=961,freq='30s')
            pd.DataFrame({'timestamp':times,'channel_1':np.sin(np.arange(961)/17),'is_anomaly_channel_1':0}).to_csv(root/'train.csv',index=False)
            pd.DataFrame({'ID':['v','a'],'Channel':['channel_1']*2,'StartTime':[times[400],times[700]],'EndTime':[times[410],times[710]]}).to_csv(root/'labels.csv',index=False)
            pd.DataFrame({'ID':['v','a'],'Category':['Anomaly']*2}).to_csv(root/'types.csv',index=False)
            whole=root/'whole';partial=root/'partial';whole.mkdir();partial.mkdir()
            kwargs=dict(epochs=2,device='cpu',batch_size=8,seed=42)
            args=({'features':['channel_1'],'pseudo':True},tuple(str(times[i]) for i in [320,640,960]))
            def interrupt(payload,path):
                atomic_save(payload,path)
                if path.name=='checkpoint_last.pt' and payload['epoch']==1:raise RuntimeError('disconnect')
            with patch.multiple(config,TRAIN_FILE=root/'train.csv',TEST_FILE=root/'NO_TEST',ANNOTATIONS_FILE=root/'labels.csv',ANOMALY_TYPES_FILE=root/'types.csv'), contextlib.redirect_stdout(io.StringIO()):
                run_fold(*args,whole,**kwargs)
                with patch('esa_thesis.development.atomic_save',side_effect=interrupt):
                    with self.assertRaisesRegex(RuntimeError,'disconnect'):run_fold(*args,partial,**kwargs)
                run_fold(*args,partial,**kwargs)
            a=torch.load(whole/'checkpoint_last.pt',map_location='cpu',weights_only=False)
            b=torch.load(partial/'checkpoint_last.pt',map_location='cpu',weights_only=False)
            self.assertEqual(a['history'],b['history'])
            self.assertEqual(a['best'],b['best'])
            for key in a['model']:torch.testing.assert_close(a['model'][key],b['model'][key],rtol=0,atol=0)
            np.testing.assert_array_equal(np.load(whole/'assessment_prediction.npy'),np.load(partial/'assessment_prediction.npy'))
