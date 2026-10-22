"""Frozen, resumable comparison of three authorized combinations (21 Oct 2026)."""
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
from .augmentation import make_pseudo_anomaly
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

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'data/frozen/mission1-reference-v1'
CONTROL = ROOT/'results_longrun/short_no_roll_full_20261014_seed42_v1'
FEATURES = [f'channel_{i}' for i in range(40,48)]
EXPANDED = FEATURES + ['channel_14','channel_21','channel_29']
EPOCHS = 40
BATCH = 64
SEED = 42
ARMS = ('masked', 'pretrain_finetune', 'expanded_short_no_roll')


def load_split(name, features):
    """Preallocate only requested features and global binary labels; bounded CSV chunks."""
    dates = {'train': ('2000-01-01','2006-01-01'),
             'calibration': ('2006-01-01','2007-01-01'),
             'testing': ('2007-01-01','2014-01-01')}
    start, end = dates[name]
    times = pd.date_range(start,end,inclusive='left',freq='30s').asi8
    x = np.empty((len(times),len(features)),np.float32)
    y = np.zeros(len(times),np.int8)
    path = BASE/(name+'.csv')
    labels = [c for c in pd.read_csv(path,nrows=0).columns if c.startswith('is_anomaly_')]
    if not labels: raise ValueError('Missing labels')
    offset = 0
    for frame in pd.read_csv(path,usecols=['timestamp',*features,*labels],chunksize=25000):
        stop = offset+len(frame)
        observed = pd.DatetimeIndex(pd.to_datetime(frame.timestamp,utc=True)).as_unit('ns').asi8
        if not np.array_equal(observed,times[offset:stop]): raise ValueError('Split timestamps changed')
        values = frame[features].to_numpy(dtype=np.float32)
        lab = frame[labels].to_numpy()
        if not np.isfinite(values).all() or not np.isin(lab,[0,1,2,3]).all(): raise ValueError('Invalid input')
        x[offset:stop] = values
        y[offset:stop] = (lab>0).any(axis=1)
        offset = stop
    if offset != len(times): raise ValueError('Incomplete split')
    return x,y,times


def nominal_starts(y):
    c = np.r_[0,np.cumsum(y!=0)]
    starts = np.arange(0,len(y)-config.SEQ_LEN+1,config.STRIDE)
    starts = starts[c[starts+config.SEQ_LEN]==c[starts]]
    if not len(starts): raise ValueError('No nominal windows')
    return starts


def epoch_order(starts,epoch):
    return np.random.default_rng(SEED+epoch).permutation(starts)


def inputs_at(x,starts,device):
    return torch.from_numpy(np.stack([x[s:s+config.SEQ_LEN].T[:,:,None] for s in starts])).to(device)


def unmask(model,channels,device):
    result = MultivariateAE(channels).to(device)
    state = {k:v for k,v in model.state_dict().items() if k!='mask_token'}
    result.load_state_dict(state,strict=True)
    return result


def choose_rule(score,times):
    if not np.isfinite(score).all(): raise ValueError('Nonfinite score')
    grid = sweep_thresholds(score,evaluator=evaluator_for(times),threshold_values=np.percentile(score,QUANTILES),merge_gaps=[0,16,64],min_durs=[1,8,32])
    return select_rows(grid).iloc[0].to_dict(),grid


def make_optimizer(model,epochs):
    opt = torch.optim.AdamW(model.parameters(),lr=config.LR,weight_decay=config.WEIGHT_DECAY)
    return opt,torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs,eta_min=1e-6)


def train(name,folder,device='cuda'):
    folder.mkdir(exist_ok=True)
    if verify_receipt(folder,'trained.json'): return
    set_seed(SEED)
    features = EXPANDED if name=='expanded_short_no_roll' else FEATURES
    raw,y,_ = load_split('train',features)
    scaler = RobustChannelScaler().fit(raw,y)
    x = scaler.transform(raw); del raw
    starts = nominal_starts(y); del y
    raw,_,val_times = load_split('calibration',features)
    val = scaler.transform(raw); del raw
    masked = name in ('masked','pretrain_finetune')
    model = (MaskedAE(len(features)) if masked else MultivariateAE(len(features))).to(device)
    opt,scheduler = make_optimizer(model,20 if name=='pretrain_finetune' else EPOCHS)
    epoch,batch,best,history = 1,0,None,[]
    loss_sum,loss_count = 0.,0
    last = folder/'last.pt'
    if last.exists():
        saved = torch.load(last,map_location='cpu',weights_only=False)
        epoch,batch = saved['epoch'],saved['batch']
        if name=='pretrain_finetune' and epoch>20:
            model=MultivariateAE(len(features)).to(device)
            opt,scheduler=make_optimizer(model,20)
        model.load_state_dict(saved['model']); opt.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
        if not np.array_equal(saved['median'],scaler.median_) or not np.array_equal(saved['iqr'],scaler.iqr_): raise ValueError('Scaler changed')
        best,history=saved['best'],saved['history']
        loss_sum,loss_count=saved['loss_sum'],saved['loss_count']
        restore_rng(saved['rng']); del saved
        print(name,'resumed',epoch,batch,flush=True)
    def checkpoint(ep,ba):
        atomic_save(dict(epoch=ep,batch=ba,model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),best=best,history=history,loss_sum=loss_sum,loss_count=loss_count,median=scaler.median_,iqr=scaler.iqr_,rng=capture_rng()),last)
    while epoch<=EPOCHS:
        masked = name=='masked' or (name=='pretrain_finetune' and epoch<=20)
        order=epoch_order(starts,epoch); total=(len(order)+BATCH-1)//BATCH
        model.train()
        for b in range(batch,total):
            inputs=inputs_at(x,order[b*BATCH:(b+1)*BATCH],device)
            opt.zero_grad(set_to_none=True)
            phase=(epoch-1+b)%4
            normal=reconstruction_error(model,inputs,masked,phase)
            loss=normal.mean()
            if name!='pretrain_finetune' or epoch>20:
                corrupted=make_pseudo_anomaly(inputs) if name=='masked' else make_training_anomaly(inputs,mode='short_no_roll')
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
        if epoch%5==0 and (name!='pretrain_finetune' or epoch>20):
            print(name,'calibrating epoch',epoch,flush=True)
            score=score_windows(model,val,masked,device,BATCH)
            rule,grid=choose_rule(score,val_times)
            row['calibration_f05']=rule['val_esa_f05']
            if best is None or rule['val_esa_f05']>best['rule']['val_esa_f05']:
                best=dict(epoch=epoch,rule=rule)
                # Store candidate in the resumable checkpoint too: atomic epoch transaction.
                best['state']={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                best['grid']=grid.to_dict('records')
        history.append(row)
        if name=='pretrain_finetune' and epoch==20:
            model=unmask(model,len(features),device)
            opt,scheduler=make_optimizer(model,20)
        epoch+=1;batch=0;loss_sum=0.;loss_count=0
        checkpoint(epoch,0)
        pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
    atomic_save(dict(model_state_dict=best['state'],features=features,scaler_median=scaler.median_,scaler_iqr=scaler.iqr_,masked=name=='masked'),folder/'model.pt')
    write_json(folder/'frozen_rule.json',dict(epoch=best['epoch'],rule=best['rule']))
    pd.DataFrame(best['grid']).to_csv(folder/'calibration_candidates.csv',index=False)
    write_json(folder/'trained.json',dict(files={n:digest(folder/n) for n in ['model.pt','frozen_rule.json','history.csv','calibration_candidates.csv']}))
    del x,val,model,opt,scheduler,best;gc.collect();torch.cuda.empty_cache()


def load_model(path,device):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    model=(MaskedAE(len(saved['features'])) if saved.get('masked',False) else MultivariateAE(len(saved['features']))).to(device)
    model.load_state_dict(saved['model_state_dict'])
    scaler=RobustChannelScaler();scaler.median_=saved['scaler_median'];scaler.iqr_=saved['scaler_iqr']
    return model,scaler,saved['features'],saved.get('masked',False)


def score_model(path,folder,split):
    receipt=split+'_receipt.json';filename=split+'_score.npy'
    if verify_receipt(folder,receipt): return np.load(folder/filename,allow_pickle=False)
    model,scaler,features,masked=load_model(path,'cuda')
    raw,_,_=load_split(split,features)
    x=scaler.transform(raw);del raw
    score=score_windows(model,x,masked,'cuda',BATCH)
    save_npy(folder/filename,score)
    write_json(folder/receipt,dict(files={filename:digest(folder/filename)}))
    del model,x;gc.collect();torch.cuda.empty_cache()
    return score


def fusion_score(a,b,weight,scale_a,scale_b):
    if a.shape!=b.shape or scale_a<=0 or scale_b<=0: raise ValueError('Invalid fusion input')
    if weight==0: return a/scale_a
    if weight==1: return b/scale_b
    return (1-weight)*a/scale_a+weight*b/scale_b


def freeze_fusion(output):
    folder=output/'fusion';folder.mkdir(exist_ok=True)
    if verify_receipt(folder,'frozen.json'): return
    control=output/'control';control.mkdir(exist_ok=True)
    a=score_model(CONTROL/'full_period/model_checkpoint.pt',control,'calibration')
    b=score_model(output/'masked/model.pt',output/'masked','calibration')
    _,y,times=load_split('calibration',FEATURES)
    sa=max(float(np.percentile(a[y==0],99)),1e-8)
    sb=max(float(np.percentile(b[y==0],99)),1e-8)
    rows=[];best=None
    for weight in (0.,.25,.5,.75,1.):
        rule,grid=choose_rule(fusion_score(a,b,weight,sa,sb),times)
        grid['masked_weight']=weight;rows.append(grid)
        if best is None or rule['val_esa_f05']>best['rule']['val_esa_f05']:
            best=dict(masked_weight=weight,scale_control=sa,scale_masked=sb,rule=rule)
    pd.concat(rows,ignore_index=True).to_csv(folder/'calibration_candidates.csv',index=False)
    write_json(folder/'frozen_rule.json',best)
    write_json(folder/'frozen.json',dict(files={n:digest(folder/n) for n in ['frozen_rule.json','calibration_candidates.csv']}))


def assess(output):
    times=pd.date_range('2007-01-01','2014-01-01',inclusive='left',freq='30s').asi8
    results=[]
    for name in ('control',*ARMS,'fusion'):
        folder=output/name;folder.mkdir(exist_ok=True)
        if name=='fusion':
            frozen=json.loads((folder/'frozen_rule.json').read_text())
            score=fusion_score(np.load(output/'control/testing_score.npy'),np.load(output/'masked/testing_score.npy'),frozen['masked_weight'],frozen['scale_control'],frozen['scale_masked'])
            save_npy(folder/'testing_score.npy',score);rule=frozen['rule']
        else:
            path=CONTROL/'full_period/model_checkpoint.pt' if name=='control' else folder/'model.pt'
            score=score_model(path,folder,'testing')
            rulepath=CONTROL/'full_period/frozen_rule.json' if name=='control' else folder/'frozen_rule.json'
            rule=json.loads(rulepath.read_text())['rule']
        pred=postprocess(score>rule['threshold'],int(rule['merge_gap']),int(rule['min_dur']))
        if name=='control' and not np.array_equal(pred,np.load(CONTROL/'full_period/assessment_prediction.npy')):
            raise ValueError('Control prediction replay differs')
        save_npy(folder/'testing_prediction.npy',pred)
        annual,pooled=report_predictions(times,pred)
        combined=pd.concat([annual,pooled],ignore_index=True);combined.insert(0,'method',name)
        combined.to_csv(folder/'all_results.csv',index=False);results.append(combined)
    pd.concat(results,ignore_index=True).to_csv(output/'all_results.csv',index=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    lock=open(ROOT/'.combination-training.lock','w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not torch.cuda.is_available(): raise ValueError('CUDA unavailable')
    torch.set_num_threads(2)
    print('Verifying frozen data and reference artifacts...',flush=True)
    verify_frozen(ROOT)
    annotations=verify_recorded(ROOT,[config.ANNOTATIONS_FILE,config.ANOMALY_TYPES_FILE])
    receipt=json.loads((CONTROL/'complete.json').read_text())
    reference_names=['full_period/model_checkpoint.pt','full_period/frozen_rule.json','full_period/assessment_prediction.npy']
    for name in reference_names:
        if digest(CONTROL/name)!=receipt['artifacts'][name]: raise ValueError('Control artifact changed')
    plan=dict(experiment='combinations-20261021-v1',train=['2000-01-01','2006-01-01'],calibration=['2006-01-01','2007-01-01'],assessment=['2007-01-01','2014-01-01'],features=FEATURES,expanded_features=EXPANDED,seed=SEED,epochs=EPOCHS,batch=BATCH,pretraining_epochs=20,arms=ARMS,fusion_weights=[0,.25,.5,.75,1],fusion_normalization='99th percentile of globally nominal calibration scores',checkpoint_selection='calibration only, every five epochs; sequential arm only post-pretraining; earliest tie',manifest_sha=MANIFEST_SHA,annotations=annotations,reference={n:receipt['artifacts'][n] for n in reference_names},settings={k:str(v) for k,v in vars(config).items() if k.isupper()},runtime=dict(torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,cuda=torch.version.cuda),source_sha256={str(f.relative_to(ROOT/'esa_thesis')):digest(f) for f in (ROOT/'esa_thesis').rglob('*.py')})
    check_protocol(a.output,plan,a.resume)
    for name in ARMS:
        print('Starting arm',name,flush=True);train(name,a.output/name)
    freeze_fusion(a.output)
    # All checkpoint, fusion and threshold decisions are frozen before opening testing.csv.
    assess(a.output)
    verify_frozen(ROOT)
    if annotations!=verify_recorded(ROOT,[config.ANNOTATIONS_FILE,config.ANOMALY_TYPES_FILE]): raise ValueError('Annotations changed')
    write_json(a.output/'complete.json',dict(needs_independent_verification=True,files={str(f.relative_to(a.output)):digest(f) for f in a.output.rglob('*') if f.is_file() and f.name!='complete.json'}))
    print('All three approaches complete; independent verification required.',flush=True)

if __name__=='__main__':main()
