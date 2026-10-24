"""Frozen nine-run W&B learning-rate comparison, calibration-only selection."""
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
import wandb
from .masked_pilot import assessment_scopes
EPOCHS=10
BATCH=64
SEED=42
RATES=(3e-6,1e-5,3e-5)
METHODS=('short_no_roll','pretrain_finetune','expanded_short_no_roll')
ENTITY='kamyarfaz-politecnico-di-torino'
PROJECT='esa-thesis-finetuning'
GROUP='learning-rate-comparison-20261024-seed42-v1'
COMBINATIONS=ROOT/'results_longrun/combinations_full_20261021_seed42_v1'

def source_for(method):
    if method=='short_no_roll':return CONTROL/'full_period'
    if method not in METHODS:raise ValueError('Unknown method')
    return COMBINATIONS/method

def model_path(source):
    return source/('model_checkpoint.pt' if source.name=='full_period' else 'model.pt')

def safe_log(run,metrics):
    # Tracking must not perturb the training RNG trajectory.
    rng=capture_rng()
    try:
        run.log(metrics)
    except Exception as exc:
        print('W&B logging unavailable:',type(exc).__name__,flush=True)
    finally:
        restore_rng(rng)

def train(folder,source,lr,run,device='cuda'):
    name='short_no_roll_finetune'
    folder.mkdir(exist_ok=True)
    if verify_receipt(folder,'trained.json'): return
    set_seed(SEED)
    model,original_scaler,features,_=load_model(model_path(source),device)
    raw,y,_ = load_split('train',features)
    scaler = RobustChannelScaler().fit(raw,y)
    x = scaler.transform(raw); del raw
    starts = nominal_starts(y); del y
    raw,_,val_times = load_split('calibration',features)
    val = scaler.transform(raw); del raw
    if not np.array_equal(scaler.median_,original_scaler.median_) or not np.array_equal(scaler.iqr_,original_scaler.iqr_):
        raise ValueError('Original scaler differs')
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=config.WEIGHT_DECAY)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS,eta_min=1e-6)
    epoch,batch,history = 1,0,[]
    original_rule=json.loads((source/'frozen_rule.json').read_text())['rule']
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
                safe_log(run,{'progress/epoch':epoch,'progress/batch':b+1,'train/batch_loss':float(loss.item()),'train/learning_rate':opt.param_groups[0]['lr']})
                print(name,f'epoch {epoch}/{EPOCHS} batch {b+1}/{total} loss {loss.item():.6g}',flush=True)
        scheduler.step()
        row=dict(epoch=epoch,loss=loss_sum/loss_count)
        if epoch%5==0 or epoch==EPOCHS:
            print(name,'calibrating epoch',epoch,flush=True)
            score=score_windows(model,val,masked,device,BATCH)
            save_npy(folder/f'calibration_score_epoch{epoch}.npy',score)
            atomic_save(dict(model_state_dict=model.state_dict(),features=features,scaler_median=scaler.median_,scaler_iqr=scaler.iqr_,masked=False),folder/f'candidate_epoch{epoch}.pt')
            rule,grid=choose_rule(score,val_times)
            grid.to_csv(folder/f'calibration_candidates_epoch{epoch}.csv',index=False)
            row['calibration_f05']=rule['val_esa_f05']
            pred=postprocess(score>rule['threshold'],int(rule['merge_gap']),int(rule['min_dur']))
            for scope,ev in assessment_scopes(val_times).items():
                metrics=ev.score(pred)
                for key in ('EW_F_0.50','EW_precision','EW_recall','nominal_false_alarm_rate','TPe'):
                    row['calibration/'+scope+'/'+key]=metrics[key]
            if best is None or rule['val_esa_f05']>best['rule']['val_esa_f05']:
                best=dict(epoch=epoch,rule=rule)
                # Store candidate in the resumable checkpoint too: atomic epoch transaction.
                best['state']={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                best['grid']=grid.to_dict('records')
        history.append(row)
        epoch+=1;batch=0;loss_sum=0.;loss_count=0
        checkpoint(epoch,0)
        pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
        safe_log(run,{'epoch':row['epoch'],'train/loss':row['loss'],**{k:v for k,v in row.items() if k.startswith('calibration/')}})
    atomic_save(dict(model_state_dict=best['state'],features=features,scaler_median=scaler.median_,scaler_iqr=scaler.iqr_,masked=False),folder/'model.pt')
    write_json(folder/'frozen_rule.json',dict(epoch=best['epoch'],rule=best['rule']))
    pd.DataFrame(best['grid']).to_csv(folder/'calibration_candidates.csv',index=False)
    write_json(folder/'trained.json',dict(files={n:digest(folder/n) for n in ['model.pt','frozen_rule.json','history.csv','calibration_candidates.csv']}))
    del x,val,model,opt,scheduler,best;gc.collect();torch.cuda.empty_cache()


def freeze_selections(output):
    selections={}
    for method in METHODS:
        candidates=[]
        for i,lr in enumerate(RATES):
            folder=output/f'{method}_lr{i+1}'
            if not verify_receipt(folder,'trained.json'):raise ValueError('Incomplete training')
            frozen=json.loads((folder/'frozen_rule.json').read_text())
            candidates.append(dict(folder=folder.name,lr=lr,epoch=frozen['epoch'],rule=frozen['rule']))
        # Original fallback on any exact tie; otherwise earliest epoch, then lower LR.
        chosen=sorted(candidates,key=lambda r:(-r['rule']['val_esa_f05'],r['epoch'],r['lr']))[0]
        chosen['model_sha256']=digest(output/chosen['folder']/'model.pt')
        selections[method]=chosen
    write_json(output/'selection_complete.json',selections)
    return selections

def assess_selected(output,selections):
    results=[]
    times=pd.date_range('2007-01-01','2014-01-01',inclusive='left',freq='30s').asi8
    for method,selected in selections.items():
        folder=output/selected['folder']
        if digest(folder/'model.pt')!=selected['model_sha256']:raise ValueError('Selected model changed')
        score=score_model(folder/'model.pt',folder,'testing')
        rule=selected['rule']
        pred=postprocess(score>rule['threshold'],int(rule['merge_gap']),int(rule['min_dur']))
        save_npy(folder/'testing_prediction.npy',pred)
        source=source_for(method)
        original=np.load(source/('assessment_prediction.npy' if method=='short_no_roll' else 'testing_prediction.npy'))
        for variant,prediction in [('original',original),('selected',pred)]:
            annual,pooled=report_predictions(times,prediction)
            frame=pd.concat([annual,pooled],ignore_index=True)
            frame.insert(0,'variant',variant);frame.insert(0,'method',method);results.append(frame)
    pd.concat(results,ignore_index=True).to_csv(output/'all_results.csv',index=False)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    lock=open(ROOT/'.wandb-finetuning.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not torch.cuda.is_available():raise ValueError('CUDA unavailable')
    torch.set_num_threads(2)
    print('Verifying frozen destination data and source checkpoints',flush=True)
    verify_frozen(ROOT)
    annotations=verify_recorded(ROOT,[config.ANNOTATIONS_FILE,config.ANOMALY_TYPES_FILE])
    reference={}
    for method in METHODS:
        source=source_for(method);parent=source.parent
        receipt=json.loads((parent/'complete.json').read_text())
        hashes=receipt.get('files',receipt.get('artifacts'))
        for f in [model_path(source),source/'frozen_rule.json',source/('assessment_prediction.npy' if method=='short_no_roll' else 'testing_prediction.npy')]:
            if digest(f)!=hashes[str(f.relative_to(parent))]:raise ValueError('Reference changed')
            reference[str(f.relative_to(ROOT))]=digest(f)
    plan=dict(experiment=GROUP,methods=METHODS,learning_rates=RATES,additional_epochs=EPOCHS,
      seed=SEED,batch=BATCH,train='2000-2005',calibration='2006',assessment='2007-2013',
      selection='system calibration F0.5; original then earliest epoch then lower LR on ties',
      reference=reference,annotations=annotations,manifest_sha=MANIFEST_SHA,
      research_protocol_sha256=digest(ROOT/'docs/research/wandb-finetuning-protocol-2026-10-24.md'),
      settings={k:str(v) for k,v in vars(config).items() if k.isupper()},
      runtime=dict(torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,wandb=wandb.__version__,cuda=torch.version.cuda),
      source_sha256={str(f.relative_to(ROOT/'esa_thesis')):digest(f) for f in (ROOT/'esa_thesis').rglob('*.py')})
    check_protocol(args.output,plan,args.resume)
    for method in METHODS:
        for i,lr in enumerate(RATES):
            name=f'{method}_lr{i+1}';folder=args.output/name;folder.mkdir(exist_ok=True)
            if verify_receipt(folder,'trained.json'):continue
            cfg=dict(method=method,learning_rate=lr,additional_epochs=EPOCHS,seed=SEED,batch_size=BATCH,
                train_years='2000-2005',calibration_years='2006',assessment_years='2007-2013',
                input_channels=list(range(40,48))+([14,21,29] if method=='expanded_short_no_roll' else []),
                source_model_sha256=digest(model_path(source_for(method))),
                selection_metric='calibration/system_all_channels/EW_F_0.50',
                repeated_benchmark=True,original_checkpoint_eligible=True,
                repeats_prior_continuation=(method=='short_no_roll' and lr==1e-5))
            with wandb.init(entity=ENTITY,project=PROJECT,group=GROUP,job_type='fine-tuning',
                id='ft20261024-'+name,name=name,mode='online',resume='allow',config=cfg,
                settings=wandb.Settings(disable_git=True,save_code=False,init_timeout=60)) as run:
                run.define_metric('epoch');run.define_metric('train/loss',step_metric='epoch')
                run.define_metric('calibration/*',step_metric='epoch')
                write_json(folder/'wandb.json',dict(id=run.id,url=run.url))
                train(folder,source_for(method),lr,run)
                selected=json.loads((folder/'frozen_rule.json').read_text())
                run.summary['selection/epoch']=selected['epoch']
                run.summary['selection/calibration_system_f05']=selected['rule']['val_esa_f05']
    selections=freeze_selections(args.output)
    assess_selected(args.output,selections)
    verify_frozen(ROOT)
    if annotations!=verify_recorded(ROOT,[config.ANNOTATIONS_FILE,config.ANOMALY_TYPES_FILE]):raise ValueError('Annotations changed')
    write_json(args.output/'complete.json',dict(needs_independent_verification=True,
      files={str(f.relative_to(args.output)):digest(f) for f in args.output.rglob('*') if f.is_file() and f.name!='complete.json'}))
    print('Nine runs completed; independent verification and W&B assessment publication required.',flush=True)

if __name__=='__main__':main()
