"""Matched development comparison of legacy and short coordinated augmentation."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from . import config
from .data import AllWindows,RobustChannelScaler
from .development import FOLDS,QUANTILES,load_period
from .development_recovery import atomic_save,capture_rng,restore_rng,check_protocol
from .diagnose_channels import FEATURES,TRAIN_SHA,digest,window_errors,aggregate
from .masked_pilot import assessment_scopes
from .models import MultivariateAE
from .protocol import evaluator_for
from .runtime import set_seed
from .short_augmentation import make_training_anomaly
from .thresholds import sweep_thresholds,select_rows,postprocess


@contextmanager
def isolated_random(seed):
    state=capture_rng()
    try:
        set_seed(seed)
        yield
    finally:
        restore_rng(state)


def training_starts(labels,budget,seed):
    starts=np.arange(0,len(labels)-config.SEQ_LEN+1,config.STRIDE)
    counts=np.r_[0,np.cumsum(labels!=0)]
    starts=starts[counts[starts+config.SEQ_LEN]==counts[starts]]
    if not len(starts):raise ValueError('No globally nominal windows')
    if budget and len(starts)>budget:starts=np.sort(np.random.default_rng(seed).choice(starts,budget,replace=False))
    return starts


def score(model,x,args):
    starts=np.arange(0,len(x)-config.SEQ_LEN+1,config.SCORE_STRIDE)
    errors=window_errors(model,x,starts,args.device,args.batch_size,'scoring')
    return aggregate(errors,starts,len(x)).max(axis=0)


def train_arm(mode,x,starts,val,ev,scaler,out,args):
    out.mkdir(exist_ok=True);set_seed(args.seed)
    model=MultivariateAE(8).to(args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.LR,weight_decay=config.WEIGHT_DECAY)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=1e-6)
    first=1;history=[];best=None;weights=None;best_grid=None
    if (out/'last.pt').exists():
        if not args.resume:raise ValueError('Resume flag required')
        checkpoint=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(checkpoint['model']);optimizer.load_state_dict(checkpoint['optimizer']);scheduler.load_state_dict(checkpoint['scheduler'])
        first=checkpoint['epoch']+1;history=checkpoint['history'];best=checkpoint['best'];weights=checkpoint['weights'];best_grid=checkpoint['grid']
    dataset=AllWindows(x);dataset.starts=starts
    for epoch in range(first,args.epochs+1):
        # Epoch-addressed streams make interrupted epochs replayable. Both arms
        # see identical order and dropout draws; augmentation draws are isolated.
        set_seed(args.seed+epoch*100)
        generator=torch.Generator().manual_seed(args.seed+epoch)
        loader=DataLoader(dataset,batch_size=args.batch_size,shuffle=True,generator=generator)
        model.train();losses=[];began=time.monotonic()
        for batch,values in enumerate(loader):
            values=values.to(args.device);optimizer.zero_grad(set_to_none=True)
            normal=(model(values)-values).square().mean(dim=(1,2,3))
            with isolated_random(args.seed+1000000+epoch*100000+batch):
                corrupted=make_training_anomaly(values,mode=mode)
            abnormal=(model(corrupted)-corrupted).square().mean(dim=(1,2,3))
            penalty=torch.relu(config.PA_MARGIN-abnormal/(normal.mean().detach()+1e-8)).mean()
            loss=normal.mean()+config.PA_LAMBDA*penalty
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),config.GRAD_CLIP);optimizer.step()
            losses.append([loss.item(),normal.mean().item(),penalty.item()])
            if batch%100==0:print(mode,epoch,batch+1,len(loader),flush=True)
        scheduler.step();values=score(model,val,args)
        grid=sweep_thresholds(values,evaluator=ev,threshold_values=np.percentile(values,QUANTILES),merge_gaps=[0,16,64],min_durs=[1,8,32])
        rule=select_rows(grid).iloc[0].to_dict()
        if best is None or rule['val_esa_f05']>best['rule']['val_esa_f05']:
            best=dict(epoch=epoch,rule=rule);weights={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};best_grid=grid.to_dict('records')
        means=np.mean(losses,axis=0)
        history.append(dict(epoch=epoch,loss=means[0],normal_mse=means[1],pseudo_penalty=means[2],calibration_f05=rule['val_esa_f05'],seconds=time.monotonic()-began))
        atomic_save(dict(epoch=epoch,model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),history=history,best=best,weights=weights,grid=best_grid),out/'last.pt')
        pd.DataFrame(history).to_csv(out/'history.csv',index=False);print(mode,history[-1],flush=True)
    atomic_save(dict(model_state_dict=weights,features=FEATURES,scaler_median=scaler.median_,scaler_iqr=scaler.iqr_,augmentation=mode),out/'model_checkpoint.pt')
    pd.DataFrame(best_grid).to_csv(out/'calibration_candidates.csv',index=False)
    (out/'frozen_rule.json').write_text(json.dumps(best,indent=2)+'\n')
    return best


def verify_inputs():
    if digest(config.TRAIN_FILE)!=TRAIN_SHA:raise ValueError('Training CSV checksum mismatch')
    return {str(p):digest(p) for p in [config.ANNOTATIONS_FILE,config.ANOMALY_TYPES_FILE]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--folds',nargs='+',choices=list(FOLDS),default=['2003','2004','2005'])
    p.add_argument('--epochs',type=int,default=5);p.add_argument('--max-train-windows',type=int,default=25000)
    p.add_argument('--batch-size',type=int,default=64);p.add_argument('--seed',type=int,default=42);p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--comparison',choices=['original','short-no-roll'],default='original',
                   help='Original legacy comparison, or short-coordinated control versus short branch without rolls')
    p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true');args=p.parse_args()
    if min(args.epochs,args.batch_size)<1 or args.max_train_windows<0:p.error('Invalid budgets')
    if len(set(args.folds))!=len(args.folds):p.error('Duplicate folds')
    plan=dict(experiment='short-coordinated-development-v1',features=FEATURES,folds={f:FOLDS[f] for f in args.folds},arms=['legacy','short_coordinated'],epochs=args.epochs,max_train_windows=args.max_train_windows,batch_size=args.batch_size,seed=args.seed,device=args.device,
        candidate='Per-window50/50legacy/short; short1–10samples,1–8channels, shared offset/noise/roll; full-window population std; length1 no roll',
        selection='Original quantiles/postprocessing and total-alarm1%cap; calibration each epoch, earliest tie; BOTH arms frozen before each fold assessment',
        rng='Same initial weights; epoch-addressed independent shuffle/dropout; corruption isolated with deterministic per-batch seeds; source and settings locked',
        limitations='Exploratory previously inspected development periods; one seed; budget differs from40epoch baseline; not exact GPU determinism guarantee; no2007–2013 access',
        settings={k:getattr(config,k) for k in ['SEQ_LEN','STRIDE','SCORE_STRIDE','D_MODEL','N_HEADS','N_LAYERS','DROPOUT','LR','WEIGHT_DECAY','GRAD_CLIP','PA_LAMBDA','PA_MARGIN','CLIP_Z']})
    if args.comparison=='short-no-roll':
        plan.update(experiment='short-no-roll-development-v1',arms=['short_coordinated','short_no_roll'],
                    candidate='Per-window50/50unchanged legacy/short; short branch uses equally likely offset/noise only; same1–10sample and1–8channel laws and full-window scale',
                    comparison_note='Current short-coordinated control rerun at equal budget; legacy-generator baseline remains a separately reported historical reference. Corruption draws need not align after a type choice changes.')
    print(json.dumps(plan,indent=2),flush=True)
    if args.dry_run:return
    if args.device=='cuda' and not torch.cuda.is_available():p.error('CUDA unavailable')
    print('Verifying training CSV...',flush=True);annotations=verify_inputs()
    plan['annotations']=annotations;plan['train_sha256']=TRAIN_SHA
    package=Path(__file__).parent;plan['source_sha256']={str(f.relative_to(package)):digest(f) for f in package.rglob('*.py')}
    check_protocol(args.output,plan,args.resume);rows=[]
    for fold in args.folds:
        out=args.output/fold;out.mkdir(exist_ok=True)
        if (out/'results.csv').exists():
            rows.extend(pd.read_csv(out/'results.csv').to_dict('records'));continue
        train_end,cal_end,assessment_end=FOLDS[fold]
        x,y,_=load_period(config.TRAIN_FILE,FEATURES,None,train_end)
        scaler=RobustChannelScaler().fit(x,y);x=scaler.transform(x);starts=training_starts(y,args.max_train_windows,args.seed)
        np.save(out/'training_starts.npy',starts)
        val,_,times=load_period(config.TRAIN_FILE,FEATURES,train_end,cal_end);val=scaler.transform(val)
        if verify_inputs()!=annotations:raise ValueError('Annotations changed')
        decisions={mode:train_arm(mode,x,starts,val,evaluator_for(times),scaler,out/mode,args) for mode in plan['arms']}
        if not all((out/m/'frozen_rule.json').exists() for m in plan['arms']):raise ValueError('Missing frozen decision')
        if verify_inputs()!=annotations:raise ValueError('Annotations changed')
        del x,y,val
        print(fold,'both rules frozen; scoring development assessment',flush=True)
        x,_,times=load_period(config.TRAIN_FILE,FEATURES,cal_end,assessment_end);x=scaler.transform(x);fold_rows=[]
        for mode in plan['arms']:
            saved=torch.load(out/mode/'model_checkpoint.pt',map_location='cpu',weights_only=False)
            model=MultivariateAE(8).to(args.device);model.load_state_dict(saved['model_state_dict'])
            values=score(model,x,args);rule=decisions[mode]['rule']
            pred=postprocess(values>rule['threshold'],int(rule['merge_gap']),int(rule['min_dur']))
            np.save(out/mode/'assessment_score.npy',values);np.save(out/mode/'assessment_prediction.npy',pred)
            for scope,ev in assessment_scopes(times).items():fold_rows.append(dict(fold=fold,model=mode,scope=scope,epoch=decisions[mode]['epoch'],calibration_f05=rule['val_esa_f05'],**ev.score(pred)))
        if verify_inputs()!=annotations:raise ValueError('Annotations changed')
        pd.DataFrame(fold_rows).to_csv(out/'results.csv',index=False);rows.extend(fold_rows)
    pd.DataFrame(rows).to_csv(args.output/'results.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)

if __name__=='__main__':main()
