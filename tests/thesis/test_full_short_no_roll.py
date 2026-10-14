import contextlib,io,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from esa_thesis import config
from esa_thesis.development import run_fold
from esa_thesis.development_recovery import atomic_save
from esa_thesis.full_short_no_roll import report_predictions
from esa_thesis.evaluation.esa import EventEvaluator

class FullShortNoRollTests(unittest.TestCase):
    def test_separate_files_and_short_augmentation_resume(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);times=pd.date_range('2000-01-01',periods=960,freq='30s')
            frame=pd.DataFrame({'timestamp':times,'channel_1':np.sin(np.arange(960)/17),'is_anomaly_channel_1':0})
            for name,lo in [('train',0),('calibration',320),('testing',640)]:frame.iloc[lo:lo+320].to_csv(root/f'{name}.csv',index=False)
            pd.DataFrame({'ID':['v','a'],'Channel':['channel_1']*2,'StartTime':[times[400],times[700]],'EndTime':[times[410],times[710]]}).to_csv(root/'labels.csv',index=False)
            pd.DataFrame({'ID':['v','a'],'Category':['Anomaly']*2}).to_csv(root/'types.csv',index=False)
            kwargs=dict(epochs=2,device='cpu',batch_size=8,seed=42,training_path=root/'train.csv',calibration_path=root/'calibration.csv',assessment_path=root/'testing.csv',augmentation_mode='short_no_roll')
            bounds=(str(times[320]),str(times[640]),str(times[-1]+pd.Timedelta(seconds=30)))
            whole=root/'whole';part=root/'part';whole.mkdir();part.mkdir()
            def interrupt(payload,path):
                atomic_save(payload,path)
                if path.name=='checkpoint_last.pt' and payload['epoch']==1:raise RuntimeError('disconnect')
            with patch.multiple(config,TRAIN_FILE=root/'MUST_NOT_READ',ANNOTATIONS_FILE=root/'labels.csv',ANOMALY_TYPES_FILE=root/'types.csv'),contextlib.redirect_stdout(io.StringIO()):
                run_fold({'features':['channel_1'],'pseudo':True},bounds,whole,**kwargs)
                with patch('esa_thesis.development.atomic_save',side_effect=interrupt):
                    with self.assertRaisesRegex(RuntimeError,'disconnect'):run_fold({'features':['channel_1'],'pseudo':True},bounds,part,**kwargs)
                run_fold({'features':['channel_1'],'pseudo':True},bounds,part,**kwargs)
            a=torch.load(whole/'checkpoint_last.pt',weights_only=False);b=torch.load(part/'checkpoint_last.pt',weights_only=False)
            for k in a['model']:torch.testing.assert_close(a['model'][k],b['model'][k],rtol=0,atol=0)
            np.testing.assert_array_equal(np.load(whole/'assessment_prediction.npy'),np.load(part/'assessment_prediction.npy'))
    def test_pooled_event_ids_are_not_sum_of_yearly_counts(self):
        times=pd.DatetimeIndex([f'{y}-01-01 00:00:{s:02}' for y in range(2007,2014) for s in (0,30)]).asi8
        labels=pd.DataFrame({'ID':['shared']*2,'StartTime':pd.to_datetime(['2007-01-01','2008-01-01']),'EndTime':pd.to_datetime(['2007-01-01','2008-01-01']),'Category':['Anomaly']*2})
        pred=np.zeros(14,dtype=int);pred[0]=pred[2]=1
        with patch('esa_thesis.full_short_no_roll.assessment_scopes',side_effect=lambda t:{'system_all_channels':EventEvaluator(t,labels)}):annual,pooled=report_predictions(times,pred)
        self.assertEqual(len(annual),7);self.assertEqual(annual.TPe.sum(),2);self.assertEqual(pooled.iloc[0].TPe,1)
