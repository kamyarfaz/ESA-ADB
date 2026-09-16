"""One-epoch AE+MLP pipeline smoke test on synthetic data; no ESA training."""
import os
os.environ['WANDB_MODE'] = 'disabled'
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from esa_thesis import config


def main():
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix='esa-training-smoke-') as directory:
        base=Path(directory)
        rng=np.random.default_rng(42)
        train_times=pd.date_range('2000-01-01',periods=4000,freq='h')
        test_times=pd.date_range(train_times[-1]+pd.Timedelta(hours=1),periods=1000,freq='h')
        for name,times,lo,hi in [('train',train_times,2600,2610),('test',test_times,400,410)]:
            frame=pd.DataFrame({'timestamp':times,'channel_1':rng.normal(size=len(times)),
                                'channel_2':rng.normal(size=len(times)),'is_anomaly_channel_1':0})
            frame.loc[lo:hi,'is_anomaly_channel_1']=1
            frame.to_csv(base/(name+'.csv'),index=False)
        labels=pd.DataFrame({'ID':['a','b'],'Channel':['channel_1']*2,
                             'StartTime':[train_times[2600],test_times[400]],
                             'EndTime':[train_times[2610],test_times[410]]})
        labels.to_csv(base/'labels.csv',index=False)
        pd.DataFrame({'ID':['a','b'],'Category':['Anomaly','Rare Event']}).to_csv(base/'types.csv',index=False)
        config.TRAIN_FILE=base/'train.csv';config.TEST_FILE=base/'test.csv'
        config.ANNOTATIONS_FILE=base/'labels.csv';config.ANOMALY_TYPES_FILE=base/'types.csv'
        config.EPOCHS=1;config.MLP_EPOCHS=1
        config.THRESH_PERCENTILES=[98,99.5,100];config.MERGE_GAPS=[0,16];config.MIN_DURS=[1,8]
        from esa_thesis import training
        spec={'run':'synthetic','features':['channel_1','channel_2'],'group':'smoke'}
        result=training.train_one(spec,base,list(pd.read_csv(config.TRAIN_FILE,nrows=0).columns))
        assert result['plot_selection']['test_metric_version']=='esa-ew-id-duration-v1'
        assert result['num_val_rows']!=int(len(train_times)*.2)
        grid=pd.read_csv(base/'synthetic/threshold_sweep_val_to_test.csv')
        assert not any(c.startswith('test_') for c in grid)
        assert len(pd.read_csv(base/'synthetic/selected_thresholds.csv'))==1
        assert (base/'synthetic/mlp_checkpoint_best.pt').exists()
        again=training.train_one(spec,base,list(pd.read_csv(config.TRAIN_FILE,nrows=0).columns))
        assert again['best_epoch']==result['best_epoch']
        config.LR*=2
        try:
            training.train_one(spec,base,list(pd.read_csv(config.TRAIN_FILE,nrows=0).columns))
        except ValueError:
            pass
        else:
            raise AssertionError('Changed config must reject resume')
    print('PASS: synthetic AE+MLP training, corrected selection, final evaluation, safe resume and mismatch rejection')


if __name__=='__main__':main()
