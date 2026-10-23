"""Low-rate continuation of the original short-no-roll checkpoint."""
import argparse
import fcntl
import gc
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from . import config
from .calibration_audit import digest, verify_recorded
from .data import RobustChannelScaler
from .development import QUANTILES
from .development_recovery import atomic_save, capture_rng, restore_rng, check_protocol
from .full_short_no_roll import verify_frozen, report_predictions, MANIFEST_SHA
from .full_similarity_mosad import write_json, save_npy, verify_receipt
from .masked_pilot import MaskedAE, reconstruction_error, score_windows
from .models import MultivariateAE
from .protocol import evaluator_for
from .runtime import set_seed
from .short_augmentation import make_training_anomaly
from .thresholds import postprocess, sweep_thresholds, select_rows

from .full_combinations import (ROOT, CONTROL, FEATURES, load_split, nominal_starts,
    epoch_order, inputs_at, load_model, choose_rule, score_model)
EPOCHS=10
BATCH=64
SEED=42
LR=1e-5

def train(folder,device='cuda'):
    name='short_no_roll_finetune'
    folder.mkdir(exist_ok=True)
    if verify_receipt(folder,'trained.json'): return
    set_seed(SEED)
    features = FEATURES
    raw,y,_ = load_split('train',features)
    scaler = RobustChannelScaler().fit(raw,y)
    x = scaler.transform(raw); del raw
    starts = nominal_starts(y); del y
    raw,_,val_times = load_split('calibration',features)
    val = scaler.transform(raw); del raw
    model,original_scaler,features,_=load_model(CONTROL/'full_period/model_checkpoint.pt',device)
    if not np.array_equal(scaler.median_,original_scaler.median_) or not np.array_equal(scaler.iqr_,original_scaler.iqr_):
        raise ValueError('Original scaler differs')
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=config.WEIGHT_DECAY)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS,eta_min=1e-6)
    epoch,batch,history = 1,0,[]
    original_rule=json.loads((CONTROL/'full_period/frozen_rule.json').read_text())['rule']
    best=dict(epoch=0,rule=original_rule,state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},grid=[])
    loss_sum,loss_count = 0.,0
    last = folder/'last.pt'
    if last.exists():
        saved = torch.load(last,map_location='cpu',weights_only=False)
        epoch,batch = saved['epoch'],saved['batch']
        model.load_state_dict(saved['model']); opt.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
        if not np.array_equal(saved['median'],scaler.median_) or not np.array_equal(saved['iqr'],scaler.iqr_): raise ValueError('Scaler changed')
        best,history=saved['best'],saved['history']
        loss_sum,loss_count=saved['loss_sum'],saved['loss_count']
        restore_rng(saved['rng']); del saved
        print(name,'resumed',epoch,batch,flush=True)
    def checkpoint(ep,ba):
        atomic_save(dict(epoch=ep,batch=ba,model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),best=best,history=history,loss_sum=loss_sum,loss_count=loss_count,median=scaler.median_,iqr=scaler.iqr_,rng=capture_rng()),last)
    while epoch<=EPOCHS:
        masked = False
        order=epoch_order(starts,epoch); total=(len(order)+BATCH-1)//BATCH
        model.train()
        for b in range(batch,total):
            inputs=inputs_at(x,order[b*BATCH:(b+1)*BATCH],device)
            opt.zero_grad(set_to_none=True)
            phase=(epoch-1+b)%4
            normal=reconstruction_error(model,inputs,masked,phase)
            loss=normal.mean()
            if True:
                corrupted=make_training_anomaly(inputs,mode='short_no_roll')
                abnormal=reconstruction_error(model,corrupted,masked,phase)
                loss=loss+config.PA_LAMBDA*torch.relu(config.PA_MARGIN-abnormal/(normal.mean().detach()+1e-8)).mean()
            if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),config.GRAD_CLIP); opt.step()
            loss_sum+=loss.item();loss_count+=1
            if b==0 or (b+1)%100==0 or b+1==total:
                checkpoint(epoch,b+1)
                print(name,f'epoch {epoch}/{EPOCHS} batch {b+1}/{total} loss {loss.item():.6g}',flush=True)
        scheduler.step()
        row=dict(epoch=epoch,loss=loss_sum/loss_count)
        if epoch%5==0 or epoch==EPOCHS:
            print(name,'calibrating epoch',epoch,flush=True)
            score=score_windows(model,val,masked,device,BATCH)
            rule,grid=choose_rule(score,val_times)
            grid.to_csv(folder/f'calibration_candidates_epoch{epoch}.csv',index=False)
            row['calibration_f05']=rule['val_esa_f05']
            if best is None or rule['val_esa_f05']>best['rule']['val_esa_f05']:
                best=dict(epoch=epoch,rule=rule)
                # Store candidate in the resumable checkpoint too: atomic epoch transaction.
                best['state']={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                best['grid']=grid.to_dict('records')
        history.append(row)
        epoch+=1;batch=0;loss_sum=0.;loss_count=0
        checkpoint(epoch,0)
        pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
    atomic_save(dict(model_state_dict=best['state'],features=features,scaler_median=scaler.median_,scaler_iqr=scaler.iqr_,masked=False),folder/'model.pt')
    write_json(folder/'frozen_rule.json',dict(epoch=best['epoch'],rule=best['rule']))
    pd.DataFrame(best['grid']).to_csv(folder/'calibration_candidates.csv',index=False)
    write_json(folder/'trained.json',dict(files={n:digest(folder/n) for n in ['model.pt','frozen_rule.json','history.csv','calibration_candidates.csv']}))
    del x,val,model,opt,scheduler,best;gc.collect();torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    lock=open(ROOT/'.short-no-roll-finetune.lock','w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not torch.cuda.is_available(): raise ValueError('CUDA unavailable')
    torch.set_num_threads(2)
    print('Verifying destination frozen data and reference...',flush=True)
    verify_frozen(ROOT)
    annotations=verify_recorded(ROOT,[config.ANNOTATIONS_FILE,config.ANOMALY_TYPES_FILE])
    receipt=json.loads((CONTROL/'complete.json').read_text())
    refs=['full_period/model_checkpoint.pt','full_period/frozen_rule.json','full_period/assessment_prediction.npy']
    for n in refs:
        if digest(CONTROL/n)!=receipt['artifacts'][n]: raise ValueError('Reference changed: '+n)
    protocol_path=ROOT/'docs/research/short-no-roll-finetune-protocol-2026-10-23.md'
    plan=dict(experiment='short-no-roll-finetune-20261023-v1',features=FEATURES,epochs=EPOCHS,
        lr=LR,batch=BATCH,seed=SEED,train='2000-2005',calibration='2006',assessment='2007-2013',
        original_epoch=10,calibration_epochs=[0,5,10],selection='corrected system F0.5; original/earliest on tie',
        reference={n:receipt['artifacts'][n] for n in refs},annotations=annotations,
        protocol_sha256=digest(protocol_path),manifest_sha=MANIFEST_SHA,
        settings={k:str(v) for k,v in vars(config).items() if k.isupper()},
        runtime=dict(torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,cuda=torch.version.cuda),
        source_sha256={str(f.relative_to(ROOT/'esa_thesis')):digest(f) for f in (ROOT/'esa_thesis').rglob('*.py')})
    check_protocol(a.output,plan,a.resume)
    folder=a.output/'finetuned'
    train(folder)
    # Training receipt freezes checkpoint and rule before opening assessment.
    if not verify_receipt(folder,'trained.json'): raise ValueError('Missing frozen selection')
    score=score_model(folder/'model.pt',folder,'testing')
    rule=json.loads((folder/'frozen_rule.json').read_text())['rule']
    pred=postprocess(score>rule['threshold'],int(rule['merge_gap']),int(rule['min_dur']))
    save_npy(folder/'testing_prediction.npy',pred)
    times=pd.date_range('2007-01-01','2014-01-01',inclusive='left',freq='30s').asi8
    results=[]
    for name,prediction in [('finetuned',pred),('original',np.load(CONTROL/'full_period/assessment_prediction.npy'))]:
        annual,pooled=report_predictions(times,prediction)
        frame=pd.concat([annual,pooled],ignore_index=True);frame.insert(0,'method',name);results.append(frame)
    pd.concat(results,ignore_index=True).to_csv(a.output/'all_results.csv',index=False)
    verify_frozen(ROOT)
    if annotations!=verify_recorded(ROOT,[config.ANNOTATIONS_FILE,config.ANOMALY_TYPES_FILE]): raise ValueError('Annotations changed')
    write_json(a.output/'complete.json',dict(needs_independent_verification=True,files={str(f.relative_to(a.output)):digest(f) for f in a.output.rglob('*') if f.is_file() and f.name!='complete.json'}))
    print('Fine-tuning and reassessment complete; independent verification required.',flush=True)

if __name__=='__main__': main()
