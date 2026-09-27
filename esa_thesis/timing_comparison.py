"""Matched full-period channels40–47 baseline versus observation-age context."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
from . import config
from .augmentation import make_pseudo_anomaly
from .data import RobustChannelScaler
from .development import load_period,QUANTILES
from .development_recovery import atomic_save,capture_rng,restore_rng,check_protocol
from .diagnose_channels import FEATURES,aggregate,digest
from .long_baseline import verify_inputs,BOUNDARIES
from .masked_pilot import assessment_scopes
from .protocol import evaluator_for
from .runtime import set_seed
from .thresholds import sweep_thresholds,select_rows,postprocess
from .timing_model import TimingAwareAE


def verify_timing(root):
    p=json.loads((root/'protocol.json').read_text())
    if (p.get('status')!='complete' or p.get('profile')!='full-period' or p.get('features')!=FEATURES
        or p.get('train')!=['2000-01-01',BOUNDARIES[0]] or p.get('calibration')!=list(BOUNDARIES[:2])
        or p.get('assessment')!=list(BOUNDARIES[1:])):
        raise ValueError('A completed full-period timing dataset with matching intervals is required')
    pairs=p.get('matching_worker_pairs',{})
    if set(pairs)!=set(FEATURES) or any(len(v)!=2 for v in pairs.values()):
        raise ValueError('Two matching independent workers per channel are required')
    for name,h in p['files_sha256'].items():
        if Path(name).name!=name or digest(root/name)!=h:raise ValueError(f'Timing hash mismatch: {name}')
    required={'timestamps.npy',*[f'{c}.npz' for c in FEATURES]}
    if set(p['files_sha256'])!=required:raise ValueError('Incomplete timing files')
    return p


def load_timing(root,times):
    all_times=np.load(root/'timestamps.npy',mmap_mode='r')
    start=int(np.searchsorted(all_times,times[0]));end=start+len(times)
    if not np.array_equal(all_times[start:end],times):raise ValueError('Timing grid differs from telemetry')
    result=np.empty((len(times),8,2),dtype=np.float32)
    for j,c in enumerate(FEATURES):
        with np.load(root/f'{c}.npz') as f:
            if len(f['log_age'])!=len(all_times) or len(f['available'])!=len(all_times):raise ValueError('Timing length differs')
            age=f['log_age'][start:end];available=f['available'][start:end]
            if not np.isfinite(age).all() or np.any(age<0) or not np.isin(available,[False,True]).all():raise ValueError('Invalid timing context')
            result[:,j,0]=age;result[:,j,1]=available
    return result


class TimingWindows(Dataset):
    def __init__(self,x,context,starts):self.x=x;self.context=context;self.starts=starts
    def __len__(self):return len(self.starts)
    def __getitem__(self,index):
        start=int(self.starts[index]);stop=start+config.SEQ_LEN
        return torch.from_numpy(self.x[start:stop].T[:,:,None]),torch.from_numpy(self.context[start:stop].transpose(1,0,2))


def nominal_starts(labels):
    starts=np.arange(0,len(labels)-config.SEQ_LEN+1,config.STRIDE)
    counts=np.r_[0,np.cumsum(labels!=0)]
    starts=starts[counts[starts+config.SEQ_LEN]==counts[starts]]
    if not len(starts):raise ValueError('No nominal training windows')
    return starts


@torch.no_grad()
def score(model,x,context,enabled,device,batch_size):
    model.eval();starts=np.arange(0,len(x)-config.SEQ_LEN+1,config.SCORE_STRIDE)
    errors=np.empty((len(starts),8),dtype=np.float32);offset=0
    for batch,(values,timing) in enumerate(DataLoader(TimingWindows(x,context,starts),batch_size=batch_size),1):
        values=values.to(device);timing=timing.to(device) if enabled else None
        e=(model(values,timing)-values).square().mean(dim=(2,3)).cpu().numpy()
        if not np.isfinite(e).all():raise ValueError('Nonfinite scores')
        errors[offset:offset+len(e)]=e;offset+=len(e)
        if batch==1 or batch%100==0 or offset==len(starts):print(f'scoring {offset}/{len(starts)}',flush=True)
    return aggregate(errors,starts,len(x)).max(axis=0)


def train_arm(name,x,context,starts,val,val_context,ev,scaler,out,args):
    out.mkdir(exist_ok=True);set_seed(args.seed);enabled=name=='observation_age'
    model=TimingAwareAE(8).to(args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.LR,weight_decay=config.WEIGHT_DECAY)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=1e-6)
    history=[];best=None;best_state=None;best_grid=None;first=1
    if (out/'last.pt').exists():
        state=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        history=state['history'];best=state['best'];best_state=state['best_state'];best_grid=state['best_grid'];first=state['epoch']+1
        restore_rng(state['rng'])
    loader=DataLoader(TimingWindows(x,context,starts),batch_size=args.batch_size,shuffle=True)
    for epoch in range(first,args.epochs+1):
        model.train();losses=[];began=time.monotonic()
        for batch,(values,timing) in enumerate(loader,1):
            values=values.to(args.device);timing=timing.to(args.device) if enabled else None
            optimizer.zero_grad(set_to_none=True)
            normal=(model(values,timing)-values).square().mean(dim=(1,2,3))
            pseudo=make_pseudo_anomaly(values)
            abnormal=(model(pseudo,timing)-pseudo).square().mean(dim=(1,2,3))
            penalty=torch.relu(config.PA_MARGIN-abnormal/(normal.mean().detach()+1e-8)).mean()
            loss=normal.mean()+config.PA_LAMBDA*penalty
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),config.GRAD_CLIP);optimizer.step()
            losses.append((loss.item(),normal.mean().item(),penalty.item()))
            if batch==1 or batch%100==0:print(name,epoch,batch,len(loader),flush=True)
        scheduler.step();values=score(model,val,val_context,enabled,args.device,args.batch_size)
        grid=sweep_thresholds(values,evaluator=ev,threshold_values=np.percentile(values,QUANTILES),merge_gaps=[0,16,64],min_durs=[1,8,32])
        rule=select_rows(grid).iloc[0].to_dict()
        if best is None or rule['val_esa_f05']>best['rule']['val_esa_f05']:
            best=dict(epoch=epoch,rule=rule);best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};best_grid=grid.to_dict('records')
        mean=np.mean(losses,axis=0);history.append(dict(epoch=epoch,loss=mean[0],normal_mse=mean[1],pseudo_penalty=mean[2],calibration_f05=rule['val_esa_f05'],seconds=time.monotonic()-began))
        atomic_save(dict(epoch=epoch,model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),history=history,best=best,best_state=best_state,best_grid=best_grid,rng=capture_rng()),out/'last.pt')
        pd.DataFrame(history).to_csv(out/'history.csv',index=False);print(name,history[-1],flush=True)
    atomic_save(dict(model_state_dict=best_state,features=FEATURES,scaler_median=scaler.median_,scaler_iqr=scaler.iqr_,timing_enabled=enabled),out/'model_checkpoint.pt')
    pd.DataFrame(best_grid).to_csv(out/'calibration_candidates.csv',index=False)
    (out/'frozen_rule.json').write_text(json.dumps(best,indent=2)+'\n')
    return best


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--timing-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=5);p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    if min(a.epochs,a.batch_size)<1:p.error('Positive budgets required')
    plan=dict(experiment='full-period-timing-comparison-v1',features=FEATURES,train=['2000-01-01',BOUNDARIES[0]],calibration=list(BOUNDARIES[:2]),assessment=list(BOUNDARIES[1:]),
        arms=['control','observation_age'],epochs=a.epochs,batch_size=a.batch_size,seed=a.seed,device=a.device,
        windows='All globally nominal stride16 windows of256samples; no cap',selection='Each arm: calibration every epoch, earliest F0.5 tie; freeze BOTH before assessment',
        context='log1p(observation age/training median native cadence), availability. Control receives zero context; same model parameter count and initial outputs.',
        assessment_policy='One frozen rule per arm; continuous2007–2013 predictions, pooled primary and yearly local/system metrics; no yearly retuning',
        limitations='Retrospective reconstruction; inherited telemetry preparation; benchmark previously inspected, not untouched test. Five-epoch matched comparison is not the historical40epoch baseline. Intermittent integrity fault remains unresolved; gates do not certify hardware.')
    print(json.dumps(plan,indent=2),flush=True)
    if a.dry_run:return
    if a.device=='cuda' and not torch.cuda.is_available():p.error('CUDA unavailable; run in GPU-enabled environment')
    print('Verifying telemetry and independently reproduced timing sidecars...',flush=True)
    plan['inputs']=verify_inputs();plan['timing_protocol']=verify_timing(a.timing_dir)
    plan['annotations']={str(f):digest(f) for f in [config.ANNOTATIONS_FILE,config.ANOMALY_TYPES_FILE]}
    package=Path(__file__).parent;plan['source_sha256']={str(f.relative_to(package)):digest(f) for f in package.rglob('*.py')}
    check_protocol(a.output,plan,a.resume)
    x,y,t=load_period(config.TRAIN_FILE,FEATURES,None,BOUNDARIES[0]);context=load_timing(a.timing_dir,t)
    scaler=RobustChannelScaler().fit(x,y);x=scaler.transform(x);starts=nominal_starts(y)
    np.save(a.output/'training_starts.npy',starts);print(len(starts),'nominal training windows',flush=True)
    val,_,vt=load_period(config.TRAIN_FILE,FEATURES,*BOUNDARIES[:2]);val=scaler.transform(val);val_context=load_timing(a.timing_dir,vt)
    verify_inputs();verify_timing(a.timing_dir)
    decisions={}
    for name in plan['arms']:decisions[name]=train_arm(name,x,context,starts,val,val_context,evaluator_for(vt),scaler,a.output/name,a)
    del x,y,t,context,val,val_context
    if not all((a.output/name/'frozen_rule.json').exists() for name in plan['arms']):raise ValueError('Rules not frozen')
    print('Both rules frozen. Loading2007–2013 assessment.',flush=True)
    x,_,times=load_period(config.TEST_FILE,FEATURES,*BOUNDARIES[1:]);x=scaler.transform(x);context=load_timing(a.timing_dir,times)
    rows=[];annual=[]
    for name in plan['arms']:
        model=TimingAwareAE(8).to(a.device);saved=torch.load(a.output/name/'model_checkpoint.pt',map_location='cpu',weights_only=False);model.load_state_dict(saved['model_state_dict'])
        values=score(model,x,context,name=='observation_age',a.device,a.batch_size);rule=decisions[name]['rule']
        pred=postprocess(values>rule['threshold'],int(rule['merge_gap']),int(rule['min_dur']))
        np.save(a.output/name/'assessment_score.npy',values);np.save(a.output/name/'assessment_prediction.npy',pred)
        for scope,ev in assessment_scopes(times).items():rows.append(dict(model=name,scope=scope,epoch=decisions[name]['epoch'],calibration_f05=rule['val_esa_f05'],**ev.score(pred)))
        for year in range(2007,2014):
            keep=(times>=pd.Timestamp(f'{year}-01-01').value)&(times<pd.Timestamp(f'{year+1}-01-01').value)
            for scope,ev in assessment_scopes(times[keep]).items():annual.append(dict(model=name,year=year,scope=scope,**ev.score(pred[keep])))
    verify_inputs();verify_timing(a.timing_dir)
    pd.DataFrame(rows).to_csv(a.output/'pooled_results.csv',index=False);pd.DataFrame(annual).to_csv(a.output/'yearly_results.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)

if __name__=='__main__':main()
