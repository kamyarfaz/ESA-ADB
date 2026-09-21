"""Read-only channel attribution plus sampled nominal training residual references."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from esa_thesis import config
from esa_thesis.development import FOLDS
from esa_thesis.models import MultivariateAE
from esa_thesis.data import RobustChannelScaler
from esa_thesis.protocol import evaluator_for


def load_source():
    features=[f'channel_{i}' for i in [*range(40,48),14,21,29]]
    header=pd.read_csv(config.TRAIN_FILE,nrows=0).columns
    labels=[c for c in header if c.startswith('is_anomaly_')]
    xs=[];ys=[];ts=[]
    with pd.read_csv(config.TRAIN_FILE,usecols=['timestamp',*features,*labels],chunksize=100000) as reader:
        for frame in reader:
            times=pd.to_datetime(frame.timestamp,utc=True).astype('int64').to_numpy()
            keep=times<pd.Timestamp('2006-01-01',tz='UTC').value
            values=frame.loc[keep,features].to_numpy(np.float32)
            annotation=frame.loc[keep,labels].to_numpy()
            if not np.isfinite(values).all() or not np.isin(annotation,[0,1,2,3]).all():raise ValueError('Invalid input')
            xs.append(values);ys.append((annotation>0).any(axis=1));ts.append(times[keep])
            if not keep.all():break
    times=np.concatenate(ts)
    if not np.all(np.diff(times)==30_000_000_000):raise ValueError('Timestamp grid mismatch')
    return np.concatenate(xs),np.concatenate(ys),times,features


@torch.no_grad()
def point_errors(model,x,plans,batch=32):
    # Each plan lists all full windows contributing to one frozen score point.
    unique=np.unique(np.concatenate(plans));errors=[]
    for offset in range(0,len(unique),batch):
        starts=unique[offset:offset+batch]
        inputs=torch.from_numpy(np.stack([x[s:s+config.SEQ_LEN].T[:,:,None] for s in starts]))
        errors.append((model(inputs)-inputs).square().mean(dim=(2,3)).numpy())
    errors=np.concatenate(errors)
    return np.stack([errors[np.searchsorted(unique,p)].mean(axis=0) for p in plans])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    print('Loading training CSV through 2005 once...',flush=True)
    values,labels,times,columns=load_source()
    output=args.output;summary=[];alarms=[]
    root=config.TRAIN_FILE.parents[4]/'results_longrun/development'
    provenance={'sample_points':128,'seed':42,'reference':'random nominal training points on 32-sample grid; all contributing windows nominal; same points for both variants',
                'purpose':'diagnosis only, no scaling/threshold/model changes','checkpoint_sha256':{}}
    for fold,(train_end,cal_end,assess_end) in FOLDS.items():
        train_n=np.searchsorted(times,pd.Timestamp(train_end,tz='UTC').value)
        lower=np.searchsorted(times,pd.Timestamp(cal_end,tz='UTC').value)
        upper=np.searchsorted(times,pd.Timestamp(assess_end,tz='UTC').value)
        cumulative=np.r_[0,np.cumsum(labels[:train_n])]
        points=np.arange(config.SEQ_LEN-config.SCORE_STRIDE,train_n-config.SEQ_LEN+1,config.SCORE_STRIDE)
        left=points-config.SEQ_LEN+config.SCORE_STRIDE
        points=points[(cumulative[points+config.SEQ_LEN]-cumulative[left])==0]
        points=np.sort(np.random.default_rng(42).choice(points,size=min(128,len(points)),replace=False))
        if not len(points):raise ValueError('No nominal reference points')
        offsets=np.arange(-config.SEQ_LEN+config.SCORE_STRIDE,1,config.SCORE_STRIDE)
        for variant in ['baseline','expanded']:
            parent='ae_expanded_seed42' if variant=='expanded' else ('ae_baseline_2003_seed42' if fold=='2003' else 'ae_baseline_remaining_seed42')
            run=root/parent/fold
            checkpoint=run/'model_checkpoint.pt';provenance['checkpoint_sha256'][str(checkpoint)]=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False);features=saved['features']
            model=MultivariateAE(len(features));model.load_state_dict(saved['model_state_dict']);model.eval()
            scaler=RobustChannelScaler();scaler.median_=saved['scaler_median'];scaler.iqr_=saved['scaler_iqr']
            x=scaler.transform(values[:,[columns.index(f) for f in features]])
            print(f'{fold} {variant}: sampling nominal residuals and attributing alarm peaks',flush=True)
            reference=point_errors(model,x,[p+offsets for p in points])
            pred=np.load(run/'assessment_prediction.npy');score=np.load(run/'assessment_score.npy')
            evaluator=evaluator_for(times[lower:upper]);actual=evaluator.score(pred);reported=json.loads((run/'results.json').read_text())
            for key in ['EW_F_0.50','TPe','FPe','FNe']:assert np.isclose(actual[key],reported[key],rtol=0,atol=1e-12)
            starts,ends,closed=evaluator.prediction_intervals(pred);plans=[];details=[]
            for start,end,shut in zip(starts,ends,closed):
                matched=any(evaluator._overlaps(np.array([start]),np.array([end]),np.array([shut]),lo,hi)[0] for ints in evaluator.by_id.values() for lo,hi in ints)
                neutral=any(evaluator._overlaps(np.array([start]),np.array([end]),np.array([shut]),lo,hi)[0] for lo,hi in evaluator.all_intervals)
                lo=np.searchsorted(times[lower:upper],start);hi=np.searchsorted(times[lower:upper],end,side='right' if shut else 'left')
                peak=lo+int(np.argmax(score[lo:hi]));first=max(0,(peak-config.SEQ_LEN+config.SCORE_STRIDE)//config.SCORE_STRIDE)*config.SCORE_STRIDE
                contributing=np.arange(first,min(peak,upper-lower-config.SEQ_LEN)//config.SCORE_STRIDE*config.SCORE_STRIDE+1,config.SCORE_STRIDE)
                plans.append(lower+contributing)
                details.append({'start':str(pd.Timestamp(start,tz='UTC')),'kind':'event_overlap' if matched else ('neutral' if neutral else 'false_alarm'),
                                'peak_score':float(score[peak]),'seconds':(end-start)/1e9})
            peaks=point_errors(model,x,plans) if plans else np.empty((0,len(features)))
            winners=peaks.argmax(axis=1) if len(peaks) else np.array([],dtype=int)
            for i,item in enumerate(details):
                assert np.isclose(peaks[i].max(),item['peak_score'],rtol=2e-4,atol=1e-5),(fold,variant,item)
                for j,channel in enumerate(features):alarms.append({'fold':fold,'variant':variant,**item,'channel':channel,'error':float(peaks[i,j]),'dominant':bool(winners[i]==j)})
            for j,channel in enumerate(features):
                summary.append({'fold':fold,'variant':variant,'channel':channel,'input_median':float(scaler.median_[j]),'input_iqr':float(scaler.iqr_[j]),
                                'nominal_error_median':float(np.median(reference[:,j])),'nominal_error_p95':float(np.percentile(reference[:,j],95)),
                                'nominal_error_p99':float(np.percentile(reference[:,j],99)),'reference_points':len(points),
                                'reference_dominant':int((reference.argmax(axis=1)==j).sum()),
                                'false_alarm_peaks_dominated':sum(d['kind']=='false_alarm' and winners[i]==j for i,d in enumerate(details)),
                                'event_alarm_peaks_dominated':sum(d['kind']=='event_overlap' and winners[i]==j for i,d in enumerate(details))})
            pd.DataFrame(summary).to_csv(output/'channel_summary.csv',index=False)
            pd.DataFrame(alarms).to_csv(output/'alarm_channel_errors.csv',index=False)
            del x,model
    (output/'protocol.json').write_text(json.dumps(provenance,indent=2))
    print('Complete:',output,flush=True)


if __name__=='__main__':main()
