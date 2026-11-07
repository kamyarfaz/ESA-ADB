"""Frozen, paired three-seed input expansion (4 November 2026)."""
import argparse
import fcntl
import gc
import json
import os
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import config
from .calibration_audit import digest, verify_recorded, detection_ids
from .data import RobustChannelScaler
from .development import QUANTILES
from .development_recovery import atomic_save, capture_rng, restore_rng, check_protocol
from .full_combinations import load_split, nominal_starts, inputs_at, unmask, make_optimizer, load_model
from .full_short_no_roll import verify_frozen, report_predictions, MANIFEST_SHA
from .full_similarity_mosad import write_json, save_npy, verify_receipt
from .masked_pilot import MaskedAE, reconstruction_error, assessment_scopes
from .models import MultivariateAE
from .runtime import set_seed
from .short_augmentation import make_training_anomaly
from .thresholds import postprocess, sweep_thresholds, select_rows

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT/'docs/research/context-expansion-protocol-2026-11-04.md'
CORRELATIONS = ROOT/'results_longrun/all_channel_correlations_20261102_v1'
SELECTION = ROOT/'docs/research/tables/context-expansion-2026-11-04/input-selection.json'
FEATURES = [f'channel_{i}' for i in range(40, 48)]
METHODS = ('short_no_roll', 'pretrain_finetune', 'expanded_short_no_roll')
SEEDS = (42, 43, 44)
EPOCHS, PRETRAIN_EPOCHS, BATCH = 40, 20, 64
GROUP = 'context-expansion-20261104-v1'
ENTITY, PROJECT = 'kamyarfaz-politecnico-di-torino', 'esa-thesis-finetuning'


def select_inputs(folder=CORRELATIONS):
    """Only training matrices enter selection; raw grid independently replays all pairs."""
    from scipy.stats import spearmanr
    receipt = json.loads((folder/'complete.json').read_text())
    hashes = {r['path']: r['sha256'] for r in receipt['files']}
    names = ('mission1_training_grid.npz', 'mission1_block_matrices.npz')
    for name in names:
        if digest(folder/name) != hashes[name]:
            raise ValueError('Training correlation artifact changed: '+name)
    with np.load(folder/names[0], allow_pickle=False) as z:
        times, channels, values = z['times'], list(z['channels']), z['values']
    dates = pd.DatetimeIndex(times)
    if not (dates.min() == pd.Timestamp('2000-01-01') and dates.max() < pd.Timestamp('2006-01-01')):
        raise ValueError('Channel selection must only use 2000–2005')
    quarter = dates.to_period('Q').astype(str).to_numpy()
    rows, selected, replay_errors = [], [], []
    with np.load(folder/names[1], allow_pickle=False) as z:
        if list(z['channels']) != channels or len(z['blocks']) != 25:
            raise ValueError('Unexpected training matrix inventory')
        for anchor, pool in ((40, (16,24,32,33)), (47, (48,49,50,51,52))):
            candidates = []
            for number in pool:
                i, j = channels.index(f'channel_{anchor}'), channels.index(f'channel_{number}')
                replay = []
                for b, block in enumerate(z['blocks']):
                    keep = np.ones(len(times), bool) if b == 0 else quarter == block
                    a, v = values[keep, i], values[keep, j]
                    valid = np.isfinite(a) & np.isfinite(v)
                    r = float(spearmanr(a[valid], v[valid]).statistic)
                    if int(valid.sum()) != int(z['pairs'][b,i,j]) or not np.isclose(r,z['spearman'][b,i,j],atol=1e-9,rtol=0,equal_nan=True):
                        raise ValueError('Independent quarterly correlation replay failed')
                    replay_errors.append(abs(r-float(z['spearman'][b,i,j])))
                    replay.append(r)
                q = np.array(replay[1:])
                eligible = bool(np.isfinite(q).all() and ((q>0).all() or (q<0).all()) and z['pairs'][1:,i,j].min() >= 500)
                row = dict(anchor=anchor, channel=number, eligible=eligible,
                           min_abs_quarter_spearman=float(np.abs(q).min()), pooled_spearman=replay[0],
                           pooled_pearson=float(z['pearson'][0,i,j]), delta_pearson=float(z['delta'][0,i,j]),
                           minimum_quarter_pairs=int(z['pairs'][1:,i,j].min()), quarter_spearman=replay[1:])
                rows.append(row)
                if eligible: candidates.append(row)
            if not candidates: raise ValueError('No eligible representative for anchor '+str(anchor))
            candidates.sort(key=lambda r: (-r['min_abs_quarter_spearman'], -abs(r['pooled_spearman']), r['channel']))
            selected.append(candidates[0]['channel'])
    return dict(selected=selected, candidates=rows, assessment_used=False, calibration_used=False,
                replayed_coefficients=len(rows)*25, max_replay_absolute_error=float(np.nanmax(replay_errors)), protocol_sha256=digest(PROTOCOL),
                dependencies={str((folder/n).relative_to(ROOT)):hashes[n] for n in names})


def jobs(selected):
    result = []
    for method in METHODS:
        base = FEATURES + (['channel_14','channel_21','channel_29'] if method == 'expanded_short_no_roll' else [])
        for seed in SEEDS:
            for arm in ('control', 'context'):
                features = base + ([f'channel_{i}' for i in selected] if arm == 'context' else [])
                result.append(dict(name=f'{method}_s{seed}_{arm}', method=method, seed=seed, arm=arm, features=features))
    return result


def epoch_order(starts, epoch, seed):
    return np.random.default_rng(seed+epoch).permutation(starts)


def scale_inplace(raw, scaler):
    for start in range(0, len(raw), 25000):
        raw[start:start+25000] = scaler.transform(raw[start:start+25000])
    return raw


def aggregate_max(errors, starts, length):
    """Float32 additions in original order, without simultaneous C×T allocations."""
    count = np.zeros(length, np.float32)
    for offset in range(0, len(starts), 64):
        ix = (starts[offset:offset+64,None]+np.arange(config.SEQ_LEN)[None,:]).ravel()
        np.add.at(count, ix, 1)
    if np.any(count == 0): raise ValueError('Uncovered scoring samples')
    score = np.zeros(length, np.float32)
    for channel in range(errors.shape[1]):
        total = np.zeros(length, np.float32)
        for offset in range(0, len(starts), 64):
            s = starts[offset:offset+64]
            ix = (s[:,None]+np.arange(config.SEQ_LEN)[None,:]).ravel()
            np.add.at(total, ix, np.repeat(errors[offset:offset+len(s),channel], config.SEQ_LEN))
        np.maximum(score, total/count, out=score)
    return score


@torch.no_grad()
def score_bounded(model, x, device, batch_size=BATCH):
    model.eval()
    starts = np.arange(0,len(x)-config.SEQ_LEN+1,config.SCORE_STRIDE,dtype=np.int64)
    errors = np.empty((len(starts),x.shape[1]),np.float32)
    for offset in range(0,len(starts),batch_size):
        # Match the original DataLoader's contiguous B,C,T,1 batch layout.
        # NumPy stacking preserves a different stride and changes float32 reductions.
        inputs = inputs_at(x,starts[offset:offset+batch_size],device).contiguous()
        errors[offset:offset+len(inputs)] = (model(inputs)-inputs).square().mean(dim=(2,3)).cpu().numpy()
        if offset == 0 or (offset//batch_size+1)%100 == 0 or offset+batch_size>=len(starts):
            print(f'scoring {min(offset+batch_size,len(starts))}/{len(starts)} windows',flush=True)
    if not np.isfinite(errors).all(): raise ValueError('Nonfinite reconstruction errors')
    return aggregate_max(errors, starts, len(x))


def choose_rule(score, times):
    evaluator = assessment_scopes(times)['input_channels_40_47']
    grid = sweep_thresholds(score,evaluator=evaluator,threshold_values=np.percentile(score,QUANTILES),
                            merge_gaps=[0,16,64],min_durs=[1,8,32])
    return select_rows(grid).iloc[0].to_dict(), grid


def safe_log(run, metrics):
    if run is None: return
    state = capture_rng()
    try:
        run.log(metrics)
    finally:
        restore_rng(state)


def train(job, folder, device='cuda', run=None):
    folder.mkdir(parents=True,exist_ok=True)
    if verify_receipt(folder,'trained.json'): return
    set_seed(job['seed'])
    raw,y,_ = load_split('train',job['features'])
    scaler = RobustChannelScaler().fit(raw,y)
    x = scale_inplace(raw,scaler); del raw
    starts = nominal_starts(y); del y
    val,_,val_times = load_split('calibration',job['features']); scale_inplace(val,scaler)
    pretrain = job['method'] == 'pretrain_finetune'
    model = (MaskedAE(len(job['features'])) if pretrain else MultivariateAE(len(job['features']))).to(device)
    opt,scheduler = make_optimizer(model, PRETRAIN_EPOCHS if pretrain else EPOCHS)
    epoch,batch,best,history,loss_sum,loss_count = 1,0,None,[],0.,0
    last = folder/'last.pt'
    if last.exists():
        saved = torch.load(last,map_location='cpu',weights_only=False)
        if saved['job'] != job: raise ValueError('Checkpoint job changed')
        epoch,batch = saved['epoch'],saved['batch']
        if pretrain and epoch>PRETRAIN_EPOCHS:
            model = MultivariateAE(len(job['features'])).to(device)
            opt,scheduler = make_optimizer(model,EPOCHS-PRETRAIN_EPOCHS)
        model.load_state_dict(saved['model']); opt.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
        if not np.array_equal(saved['median'],scaler.median_) or not np.array_equal(saved['iqr'],scaler.iqr_):
            raise ValueError('Training scaler changed')
        best,history = saved['best'],saved['history']
        for row in history:
            for name,sha in row.get('candidate_files',{}).items():
                if digest(folder/name)!=sha: raise ValueError('Committed calibration artifact changed: '+name)
        loss_sum,loss_count = saved['loss_sum'],saved['loss_count']
        restore_rng(saved['rng']); del saved
        print(job['name'],'resumed',epoch,batch,flush=True)

    def checkpoint(ep, ba):
        atomic_save(dict(job=job,epoch=ep,batch=ba,model=model.state_dict(),optimizer=opt.state_dict(),
                         scheduler=scheduler.state_dict(),best=best,history=history,loss_sum=loss_sum,
                         loss_count=loss_count,median=scaler.median_,iqr=scaler.iqr_,rng=capture_rng()),last)

    while epoch<=EPOCHS:
        masked = pretrain and epoch<=PRETRAIN_EPOCHS
        order = epoch_order(starts,epoch,job['seed']); total=(len(order)+BATCH-1)//BATCH
        model.train()
        for b in range(batch,total):
            inputs=inputs_at(x,order[b*BATCH:(b+1)*BATCH],device)
            opt.zero_grad(set_to_none=True)
            phase=(epoch-1+b)%4
            normal=reconstruction_error(model,inputs,masked,phase)
            loss=normal.mean()
            if not masked:
                corrupted=make_training_anomaly(inputs,mode='short_no_roll')
                abnormal=reconstruction_error(model,corrupted,False,phase)
                loss=loss+config.PA_LAMBDA*torch.relu(config.PA_MARGIN-abnormal/(normal.mean().detach()+1e-8)).mean()
            if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),config.GRAD_CLIP);opt.step()
            loss_sum+=loss.item();loss_count+=1
            if b==0 or (b+1)%100==0 or b+1==total:
                checkpoint(epoch,b+1)
                print(time.strftime('%Y-%m-%d %H:%M:%S'),job['name'],f'epoch {epoch}/{EPOCHS} batch {b+1}/{total} loss {loss.item():.6g}',flush=True)
                safe_log(run,dict(epoch=epoch,batch=b+1,train_loss=loss.item(),learning_rate=opt.param_groups[0]['lr']))
        scheduler.step()
        row=dict(epoch=epoch,loss=loss_sum/loss_count)
        if epoch%5==0 and not masked:
            score=score_bounded(model,val,device,BATCH)
            rule,grid=choose_rule(score,val_times)
            row['calibration_target_f05']=rule['val_esa_f05']
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            stem=f'epoch_{epoch:02d}'
            atomic_save(dict(model_state_dict=state,features=job['features'],scaler_median=scaler.median_,
                             scaler_iqr=scaler.iqr_,masked=False),folder/(stem+'.pt'))
            save_npy(folder/(stem+'_calibration_score.npy'),score)
            grid.to_csv(folder/(stem+'_grid.csv'),index=False)
            row['candidate_files']={n:digest(folder/n) for n in (stem+'.pt',stem+'_calibration_score.npy',stem+'_grid.csv')}
            if best is None or rule['val_esa_f05']>best['rule']['val_esa_f05']:
                best=dict(epoch=epoch,rule=rule)
            safe_log(run,dict(epoch=epoch,calibration_target_f05=rule['val_esa_f05'],calibration_target_tpr=rule['val_event_recall']))
        history.append(row)
        if pretrain and epoch==PRETRAIN_EPOCHS:
            model=unmask(model,len(job['features']),device)
            opt,scheduler=make_optimizer(model,EPOCHS-PRETRAIN_EPOCHS)
        epoch+=1;batch=0;loss_sum=0.;loss_count=0
        checkpoint(epoch,0)
        pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
    if best is None: raise ValueError('No calibration-selected checkpoint')
    stem=f'epoch_{best["epoch"]:02d}'
    shutil.copyfile(folder/(stem+'.pt'),folder/'model.pt')
    shutil.copyfile(folder/(stem+'_calibration_score.npy'),folder/'calibration_score.npy')
    shutil.copyfile(folder/(stem+'_grid.csv'),folder/'calibration_candidates.csv')
    # Regenerate history even after interruption between final checkpoint and CSV write.
    pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
    write_json(folder/'frozen_rule.json',best)
    names=[p.name for p in folder.iterdir() if p.name.startswith('epoch_')]
    names += ['model.pt','calibration_score.npy','calibration_candidates.csv','frozen_rule.json','history.csv','last.pt']
    write_json(folder/'trained.json',dict(job=job,files={n:digest(folder/n) for n in sorted(names)}))
    if run is not None: run.summary.update(dict(selected_epoch=best['epoch'],calibration_target_f05=best['rule']['val_esa_f05'],completed=True))
    del x,val,model,opt,scheduler;gc.collect();torch.cuda.empty_cache()


def frozen_plan():
    selection=json.loads(SELECTION.read_text())
    if selection['protocol_sha256'] != digest(PROTOCOL): raise ValueError('Input-selection protocol changed')
    dependencies={**selection['dependencies'],str(SELECTION.relative_to(ROOT)):digest(SELECTION)}
    for name,sha in dependencies.items():
        if digest(ROOT/name)!=sha: raise ValueError('Selection dependency changed: '+name)
    source={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'esa_thesis').rglob('*.py')}
    for p in (PROTOCOL,ROOT/'scripts/research/run_context_expansion.sh',ROOT/'scripts/research/esa-context-expansion.service'):
        source[str(p.relative_to(ROOT))]=digest(p)
    import wandb
    return dict(experiment=GROUP,jobs=jobs(selection['selected']),manifest_sha256=MANIFEST_SHA,
                annotations=verify_recorded(ROOT,[config.ANNOTATIONS_FILE,config.ANOMALY_TYPES_FILE]),
                dependencies=dependencies,source_sha256=source,epochs=EPOCHS,pretrain_epochs=PRETRAIN_EPOCHS,
                batch_size=BATCH,selection_scope='input_channels_40_47',quantiles=QUANTILES,
                train=['2000-01-01','2006-01-01'],calibration=['2006-01-01','2007-01-01'],assessment=['2007-01-01','2014-01-01'],
                settings={k:str(v) for k,v in vars(config).items() if k.isupper()},
                runtime=dict(python=platform.python_version(),torch=str(torch.__version__),cuda=torch.version.cuda,
                             gpu=torch.cuda.get_device_name(0),numpy=np.__version__,pandas=pd.__version__,wandb=wandb.__version__,
                             threads=torch.get_num_threads(),deterministic=torch.are_deterministic_algorithms_enabled(),
                             matmul_tf32=torch.backends.cuda.matmul.allow_tf32))


def check_sources(plan):
    for section in ('source_sha256','dependencies','annotations'):
        for name,sha in plan[section].items():
            if digest(ROOT/name)!=sha: raise ValueError('Frozen dependency changed: '+name)


def freeze_selections(output, all_jobs):
    records=[]
    for job in all_jobs:
        folder=output/job['name']
        if not verify_receipt(folder,'trained.json'): raise ValueError('All18 jobs must finish before assessment')
        records.append(dict(job=job,trained_sha256=digest(folder/'trained.json'),
                            rule=json.loads((folder/'frozen_rule.json').read_text()),model_sha256=digest(folder/'model.pt')))
    value=dict(selections=records,selection_scope='input_channels_40_47')
    path=output/'selection_complete.json'
    if path.exists():
        if json.loads(path.read_text())!=value: raise ValueError('Joint selection changed')
    else: write_json(path,value)
    return value


def assess(output, job):
    folder=output/job['name']
    if verify_receipt(folder,'assessed.json'): return
    if not (output/'selection_complete.json').exists(): raise ValueError('Missing joint freeze')
    model,scaler,features,_=load_model(folder/'model.pt','cuda')
    x,_,times=load_split('testing',features);scale_inplace(x,scaler)
    score=score_bounded(model,x,'cuda',BATCH)
    rule=json.loads((folder/'frozen_rule.json').read_text())['rule']
    pred=postprocess(score>rule['threshold'],int(rule['merge_gap']),int(rule['min_dur']))
    save_npy(folder/'assessment_score.npy',score);save_npy(folder/'assessment_prediction.npy',pred)
    del x,model;gc.collect();torch.cuda.empty_cache()
    annual,pooled=report_predictions(times,pred)
    result=pd.concat([annual,pooled],ignore_index=True)
    for key in ('name','method','seed','arm'): result[key]=job[key]
    result.to_csv(folder/'metrics.csv',index=False)
    event_ids={}
    for period in [str(y) for y in range(2007,2014)]+['2007-2013']:
        keep=np.ones(len(times),bool) if period=='2007-2013' else ((times>=pd.Timestamp(period+'-01-01').value)&(times<pd.Timestamp(f'{int(period)+1}-01-01').value))
        for scope,ev in assessment_scopes(times[keep]).items():
            hits=sorted(detection_ids(ev,pred[keep]));event_ids[period+'/'+scope]=dict(detected=hits,missed=sorted(set(ev.by_id)-set(hits)))
    write_json(folder/'event_ids.json',event_ids)
    names=['assessment_score.npy','assessment_prediction.npy','metrics.csv','event_ids.json']
    write_json(folder/'assessed.json',dict(files={n:digest(folder/n) for n in names}))


def require_resources():
    # This Linux user service uses the kernel's available-memory estimate without
    # introducing an extra dependency into the established research environment.
    available_kib = next(int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))
    if available_kib*1024<2.5*2**30: raise RuntimeError('Need at least2.5GiB available RAM; resume later')
    if shutil.disk_usage(ROOT).free<40*2**30: raise RuntimeError('Need at least40GiB free disk')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    if torch.cuda.mem_get_info()[0]<8*2**30: raise RuntimeError('Need at least8GiB free GPU memory')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    lock=(ROOT/'.context-expansion.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    other_locks=[]
    for name in ('.combination-training.lock','.wandb-finetuning.lock','.short-no-roll-finetune.lock'):
        handle=(ROOT/name).open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);other_locks.append(handle)
    torch.set_num_threads(2)
    require_resources()
    print('Verifying frozen destination inputs...',flush=True)
    verify_frozen(ROOT)
    plan=frozen_plan();check_protocol(args.output,plan,args.resume)
    write_json(args.output/'destination_verification.json',dict(manifest_sha256=MANIFEST_SHA,annotations=plan['annotations'],passed=True))
    import wandb
    for index,job in enumerate(plan['jobs'],1):
        folder=args.output/job['name'];folder.mkdir(exist_ok=True)
        if verify_receipt(folder,'trained.json'): continue
        check_sources(plan);require_resources();verify_frozen(ROOT)
        run_id='ctx1104-'+str(index).zfill(2)+'-s'+str(job['seed'])
        write_json(folder/'wandb_run.json',dict(id=run_id,url=f'https://wandb.ai/{ENTITY}/{PROJECT}/runs/{run_id}'))
        print('Starting',index,'of',len(plan['jobs']),job['name'],flush=True)
        with wandb.init(entity=ENTITY,project=PROJECT,group=GROUP,id=run_id,name=job['name'],resume='allow',
                        job_type='training-calibration',mode='online',dir=str(folder),
                        config={**job,'epochs':EPOCHS,'calibration_scope':'input_channels_40_47','assessment_used_for_selection':False},
                        settings=wandb.Settings(init_timeout=90,disable_git=True,save_code=False)) as run:
            train(job,folder,run=run)
    check_sources(plan);verify_frozen(ROOT)
    freeze_selections(args.output,plan['jobs'])
    print('All18 selections jointly frozen. Beginning assessment.',flush=True)
    for job in plan['jobs']:
        check_sources(plan);require_resources();assess(args.output,job)
    check_sources(plan);verify_frozen(ROOT)
    frames=[pd.read_csv(args.output/j['name']/'metrics.csv') for j in plan['jobs']]
    metrics=pd.concat(frames,ignore_index=True)
    if len(metrics)!=288: raise ValueError('Expected288 annual/pooled metric rows')
    metrics.to_csv(args.output/'all_results.csv',index=False)
    write_json(args.output/'complete.json',dict(computation_complete=True,independent_verification_pending=True,
               files={str(p.relative_to(args.output)):digest(p) for p in args.output.rglob('*') if p.is_file() and
                      ('wandb' not in p.relative_to(args.output).parts) and p.name!='complete.json'}))
    print('Computation complete; independent verification and delivery pending.',flush=True)


if __name__=='__main__': main()
