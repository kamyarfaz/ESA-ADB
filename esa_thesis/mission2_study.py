"""Mission 2 adapter: fresh three-technique study, frozen before training."""
import argparse
import fcntl
import gc
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from . import config, context_expansion as engine
from .calibration_audit import digest, detection_ids
from .development import QUANTILES
from .development_recovery import check_protocol
from .evaluation.esa import EventEvaluator
from .full_combinations import load_model
from .full_similarity_mosad import write_json, save_npy, verify_receipt
from .thresholds import postprocess, sweep_thresholds, select_rows

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT/'data/ESA-Mission2'
OUTPUT = ROOT/'results_longrun/mission2_three_technique_20261114_v1'
PREP = OUTPUT/'prepared'
PROTOCOL = ROOT/'docs/research/mission2-three-technique-protocol-2026-11-14.md'
INTEGRITY = ROOT/'docs/research/tables/mission-cross-channel-2026-11-02/raw-integrity.json'
CORR = ROOT/'results_longrun/all_channel_correlations_20261102_v1'
BASE = [f'channel_{i}' for i in (58,59,75,76,77,83,84,85)]
GROUP = 'mission2-three-technique-20261114-v1'
STEP = 18_000_000_000
SPLITS = {'train':('2000-01-01','2001-01-01'), 'calibration':('2001-01-01','2001-10-01')}


def verify_raw():
    receipt = json.loads(INTEGRITY.read_text())
    mission = next(m for m in receipt['missions'] if m['mission']==2)
    checked = {}
    for item in mission['extracted_files']:
        if digest(ROOT/item['path']) != item['sha256']:
            raise ValueError('Mission2 raw input changed: '+item['path'])
        checked[item['path']] = item['sha256']
    return checked


def select_auxiliary():
    receipt = json.loads((CORR/'complete.json').read_text())
    files = {r['path']:r['sha256'] for r in receipt['files']}
    for n in ('mission2_training_grid.npz','mission2_block_matrices.npz'):
        if digest(CORR/n)!=files[n]: raise ValueError('Correlation input changed')
    meta = pd.read_csv(RAW/'channels.csv')
    pairs = set(map(tuple,meta[meta.Channel.isin(BASE)][['Subsystem','Physical Unit']].values))
    pool = [r.Channel for r in meta.itertuples(index=False) if r.Channel not in BASE and r.Target=='YES'
            and tuple(meta.loc[meta.Channel==r.Channel,['Subsystem','Physical Unit']].iloc[0]) in pairs]
    with np.load(CORR/'mission2_training_grid.npz',allow_pickle=False) as z:
        channels=list(z['channels']); times=z['times']; values=z['values']
    dates=pd.DatetimeIndex(times); quarters=dates.to_period('Q').astype(str).to_numpy()
    if dates.min()!=pd.Timestamp('2000-01-01') or dates.max()>=pd.Timestamp('2001-01-01'):
        raise ValueError('Selection data must only use 2000')
    rows=[]
    with np.load(CORR/'mission2_block_matrices.npz',allow_pickle=False) as z:
        if list(z['channels'])!=channels or len(z['blocks'])!=5: raise ValueError('Matrix inventory')
        for candidate in pool:
            for anchor in BASE:
                i,j=channels.index(candidate),channels.index(anchor); coefs=[]; counts=[]
                for b,block in enumerate(z['blocks']):
                    keep=np.ones(len(times),bool) if b==0 else quarters==block
                    a,v=values[keep,i],values[keep,j]; valid=np.isfinite(a)&np.isfinite(v)
                    r=float(spearmanr(a[valid],v[valid]).statistic)
                    if not np.isclose(r,z['spearman'][b,i,j],atol=1e-9,rtol=0,equal_nan=True) or valid.sum()!=z['pairs'][b,i,j]:
                        raise ValueError('Independent auxiliary correlation replay failed')
                    coefs.append(r);counts.append(int(valid.sum()))
                q=np.asarray(coefs[1:]);eligible=bool(np.isfinite(q).all() and ((q>0).all() or (q<0).all()) and min(counts[1:])>=500)
                rows.append(dict(candidate=candidate,anchor=anchor,eligible=eligible,min_abs_quarter=float(np.abs(q).min()),pooled=coefs[0],quarters=coefs[1:],counts=counts))
    def key(r): return (-r['min_abs_quarter'],-abs(r['pooled']),int(r['anchor'].split('_')[1]))
    best=[sorted([r for r in rows if r['candidate']==c and r['eligible']],key=key)[0] for c in pool if any(r['candidate']==c and r['eligible'] for r in rows)]
    best.sort(key=lambda r:(-r['min_abs_quarter'],-abs(r['pooled']),int(r['candidate'].split('_')[1])))
    if len(best)<3: raise ValueError('Fewer than three eligible auxiliary channels')
    return dict(selected=[r['candidate'] for r in best[:3]],ranked_candidates=best,all_pairs=rows,
                calibration_used=False,assessment_used=False,dependencies={str((CORR/n).relative_to(ROOT)):files[n] for n in ('mission2_training_grid.npz','mission2_block_matrices.npz')})


def labels():
    return pd.read_csv(RAW/'labels.csv').merge(pd.read_csv(RAW/'anomaly_types.csv'),on='ID',validate='many_to_one')


def interval_mask(times, annotations, width=0):
    """Bins [t,t+width) overlap closed raw annotations; width0 tests point inclusion."""
    out=np.zeros(len(times),bool)
    for r in annotations.itertuples():
        lo=pd.Timestamp(r.StartTime).value;hi=pd.Timestamp(r.EndTime).value
        first=np.searchsorted(times,lo-width,side='right' if width else 'left')
        last=np.searchsorted(times,hi,side='right')
        out[first:last]=True
    return out


def resample(raw_times, raw_values, grid, start, end):
    """Causal, split-contained previous observation with finite-age validity."""
    keep=(raw_times>=start)&(raw_times<end)
    rt=raw_times[keep];rv=np.asarray(raw_values)[keep]
    if len(rt)==0: raise ValueError('Empty raw split')
    if np.any(np.diff(rt)<=0): raise ValueError('Duplicate/nonmonotone raw timestamps')
    ix=np.searchsorted(rt,grid,side='right')-1
    safe=np.maximum(ix,0);age=grid-rt[safe]
    valid=(ix>=0)&(age>=0)&(age<=600_000_000_000)&np.isfinite(rv[safe])
    result=np.full(len(grid),np.nan,np.float32);result[valid]=rv[safe[valid]]
    age=np.where(ix>=0,age,-1)
    return result,valid,age,rt[safe]


def read_channel(channel):
    frame=pd.read_pickle(RAW/'channels'/(channel+'.zip'))
    if frame.shape[1]!=1: raise ValueError('Expected single-channel source')
    return pd.DatetimeIndex(frame.index).as_unit('ns').asi8,frame.iloc[:,0].to_numpy()


def prepare_split(name,features,bounds):
    folder=PREP/name
    if verify_receipt(folder,'complete.json'):return
    folder.mkdir(parents=True,exist_ok=True)
    start,end=map(lambda s:pd.Timestamp(s).value,bounds)
    grid=np.arange(start,end,STEP,dtype=np.int64)
    if len(grid)<config.SEQ_LEN:raise ValueError('Split too short')
    x=np.empty((len(grid),len(features)),np.float32);available=np.empty(x.shape,bool)
    ages=np.empty(x.shape,np.int64);annotations=labels()
    excluded=interval_mask(grid,annotations,STEP) if name=='train' else np.zeros(len(grid),bool)
    for j,ch in enumerate(features):
        rt,rv=read_channel(ch)
        x[:,j],available[:,j],ages[:,j],source=resample(rt,rv,grid,start,end)
        if name=='train':
            # Source times are nondecreasing; mask repeated held observations too.
            excluded|=interval_mask(source,annotations[annotations.Channel==ch])
        del rt,rv;gc.collect()
    excluded|=~available.all(axis=1)
    if name=='train':
        good=x[~excluded]
        if len(good)<100:raise ValueError('Insufficient complete globally nominal rows')
        iqr=np.percentile(good,75,axis=0)-np.percentile(good,25,axis=0)
        if not np.isfinite(iqr).all() or (iqr<1e-8).any():raise ValueError('Invalid/constant training scale')
    for n,v in [('values',x),('times',grid),('excluded',excluded.astype(np.int8)),('available',available),('age_ns',ages)]:save_npy(folder/(n+'.npy'),v)
    write_json(folder/'metadata.json',dict(split=name,bounds=bounds,features=features,rows=len(grid),cadence_seconds=18,
               availability=available.mean(axis=0).tolist(),globally_nominal_complete_rows=int((~excluded).sum()) if name=='train' else None))
    names=['values.npy','times.npy','excluded.npy','available.npy','age_ns.npy','metadata.json']
    write_json(folder/'complete.json',dict(files={n:digest(folder/n) for n in names}))
    print('Prepared',name,len(grid),'rows',flush=True)


def prepare():
    OUTPUT.mkdir(parents=True,exist_ok=True);PREP.mkdir(exist_ok=True)
    if (OUTPUT/'preparation.json').exists():
        verify_prepared();return
    raw=verify_raw();selection=select_auxiliary()
    selection_path=OUTPUT/'input-selection.json'
    if selection_path.exists() and json.loads(selection_path.read_text())!=selection:raise ValueError('Selection changed')
    write_json(selection_path,selection)
    features=BASE+selection['selected']
    ends=[]
    for ch in BASE:
        times,_=read_channel(ch);ends.append(int(times[-1]))
    end=min(ends)
    # Include the final observed endpoint if it lies on the split's regular grid.
    assessment_bounds=('2001-10-01',str(pd.Timestamp(end+1)))
    for name,bounds in SPLITS.items():prepare_split(name,features,bounds)
    write_json(OUTPUT/'preparation.json',dict(raw=raw,features=features,selection_sha256=digest(selection_path),
               assessment_bounds=assessment_bounds,protocol_sha256=digest(PROTOCOL),
               prepared={n:digest(PREP/n/'complete.json') for n in SPLITS}))
    print('Preparation complete; assessment values not prepared or summarized.',flush=True)


def verify_prepared():
    p=json.loads((OUTPUT/'preparation.json').read_text())
    if p['protocol_sha256']!=digest(PROTOCOL) or p['selection_sha256']!=digest(OUTPUT/'input-selection.json'):raise ValueError('Preparation dependency changed')
    for n,sha in p['prepared'].items():
        if digest(PREP/n/'complete.json')!=sha or not verify_receipt(PREP/n,'complete.json'):raise ValueError('Prepared split changed')
    return p


def load_split(name,features):
    if name=='testing':name='assessment'
    meta=json.loads((PREP/name/'metadata.json').read_text())
    x=np.load(PREP/name/'values.npy',mmap_mode='r')[:,[meta['features'].index(c) for c in features]].copy()
    return x,np.load(PREP/name/'excluded.npy'),np.load(PREP/name/'times.npy')


def scopes(times,anomaly_only=False):
    annotations=labels();categories=('Anomaly',) if anomaly_only else ('Anomaly','Rare Event')
    system=EventEvaluator(times,annotations,categories)
    local=annotations.copy();local.loc[~local.Channel.isin(BASE),'Category']='Out-of-scope (neutral)'
    return {'system_all_channels':system,'selected_eight_channels':EventEvaluator(times,local,categories)}


def choose_rule(score,times):
    ev=scopes(times)['selected_eight_channels']
    if not ev.by_id:raise ValueError('No selected-scope calibration events')
    grid=sweep_thresholds(score,evaluator=ev,threshold_values=np.percentile(score,QUANTILES),merge_gaps=[0,16,64],min_durs=[1,8,32])
    return select_rows(grid).iloc[0].to_dict(),grid


def scoring_starts(length):
    if length<config.SEQ_LEN:raise ValueError('Scoring split shorter than a window')
    starts=np.arange(0,length-config.SEQ_LEN+1,config.SCORE_STRIDE,dtype=np.int64)
    if starts[-1]!=length-config.SEQ_LEN:starts=np.r_[starts,length-config.SEQ_LEN]
    return starts


@torch.no_grad()
def score(model,x,device,batch_size=64):
    starts=scoring_starts(len(x));errors=np.empty((len(starts),x.shape[1]),np.float32);model.eval()
    for offset in range(0,len(starts),batch_size):
        inputs=engine.inputs_at(x,starts[offset:offset+batch_size],device).contiguous()
        errors[offset:offset+len(inputs)]=(model(inputs)-inputs).square().mean(dim=(2,3)).cpu().numpy()
        if offset==0 or (offset//batch_size+1)%200==0:print('Scoring',min(offset+batch_size,len(starts)),'/',len(starts),flush=True)
    if not np.isfinite(errors).all():raise ValueError('Nonfinite scores')
    return engine.aggregate_max(errors,starts,len(x))


def jobs(features):
    return [dict(name=f'{method}_s{seed}',method=method,seed=seed,arm='expanded' if method=='expanded_short_no_roll' else 'base',
                 features=features if method=='expanded_short_no_roll' else BASE)
            for method in engine.METHODS for seed in engine.SEEDS]


def assess(job):
    folder=OUTPUT/job['name']
    if verify_receipt(folder,'assessed.json'):return
    model,scaler,features,_=load_model(folder/'model.pt','cuda')
    x,_,times=load_split('assessment',features);engine.scale_inplace(x,scaler)
    scores=score(model,x,'cuda');rule=json.loads((folder/'frozen_rule.json').read_text())['rule']
    pred=postprocess(scores>rule['threshold'],int(rule['merge_gap']),int(rule['min_dur']))
    save_npy(folder/'assessment_score.npy',scores);save_npy(folder/'assessment_prediction.npy',pred)
    rows=[];ids={};years=pd.DatetimeIndex(times).year
    periods=[(str(y),years==y) for y in sorted(set(years))]+[('pooled',np.ones(len(times),bool))]
    for period,keep in periods:
        for only in (False,True):
            definition='anomaly_only' if only else 'anomaly_and_rare'
            for scope,ev in scopes(times[keep],only).items():
                rows.append(dict(name=job['name'],method=job['method'],seed=job['seed'],arm=job['arm'],period=period,scope=scope,event_definition=definition,**ev.score(pred[keep])))
                hits=sorted(detection_ids(ev,pred[keep]));ids[period+'/'+definition+'/'+scope]=dict(detected=hits,missed=sorted(set(ev.by_id)-set(hits)))
    pd.DataFrame(rows).to_csv(folder/'metrics.csv',index=False);write_json(folder/'event_ids.json',ids)
    names=['assessment_score.npy','assessment_prediction.npy','metrics.csv','event_ids.json']
    write_json(folder/'assessed.json',dict(files={n:digest(folder/n) for n in names}))
    del x,model;gc.collect();torch.cuda.empty_cache()


def source_hashes():
    paths=list((ROOT/'esa_thesis').rglob('*.py'))+[PROTOCOL,ROOT/'scripts/research/run_mission2_study.sh',ROOT/'scripts/research/esa-mission2-study.service',ROOT/'scripts/research/verify_mission2_preparation_20261114.py']
    return {str(p.relative_to(ROOT)):digest(p) for p in paths}


def run():
    torch.set_num_threads(2);engine.require_resources();p=verify_prepared()
    preflight=json.loads((OUTPUT/'preflight-verification.json').read_text())
    if not preflight['passed'] or preflight['preparation_sha256']!=digest(OUTPUT/'preparation.json'):
        raise ValueError('Missing or mismatched independent preparation preflight')
    if preflight['verifier_sha256']!=digest(ROOT/'scripts/research/verify_mission2_preparation_20261114.py'):
        raise ValueError('Preflight verifier changed')
    if verify_raw()!=p['raw']:raise ValueError('Raw inventory changed')
    import wandb
    plan=dict(experiment=GROUP,jobs=jobs(p['features']),preparation_sha256=digest(OUTPUT/'preparation.json'),preflight_sha256=digest(OUTPUT/'preflight-verification.json'),source_sha256=source_hashes(),
              settings={k:str(v) for k,v in vars(config).items() if k.isupper()},
              runtime=dict(python=platform.python_version(),torch=str(torch.__version__),numpy=np.__version__,pandas=pd.__version__,wandb=wandb.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),threads=torch.get_num_threads(),tf32=torch.backends.cuda.matmul.allow_tf32),epochs=40,batch=64,selection_scope='selected_eight_channels')
    freeze=OUTPUT/'frozen'
    check_protocol(freeze,plan,freeze.exists())
    def check():
        if source_hashes()!=plan['source_sha256']:raise ValueError('Frozen source changed')
        if digest(OUTPUT/'preparation.json')!=plan['preparation_sha256']:raise ValueError('Preparation receipt changed')
        verify_prepared();engine.require_resources()
    # Process-local dependency injection: unchanged M1 engine provides training,
    # optimizer/RNG recovery and calibration checkpointing; all data/scoring/rules
    # are supplied by this M2 adapter. No source file or global config is rewritten.
    engine.load_split=load_split;engine.choose_rule=choose_rule;engine.score_bounded=score
    for i,job in enumerate(plan['jobs'],1):
        folder=OUTPUT/job['name'];folder.mkdir(exist_ok=True)
        if verify_receipt(folder,'trained.json'):continue
        check();rid=f'm2-1114-{i:02d}-s{job["seed"]}'
        write_json(folder/'wandb_run.json',dict(id=rid,url=f'https://wandb.ai/{engine.ENTITY}/{engine.PROJECT}/runs/{rid}'))
        print('Starting',i,'of 9',job['name'],flush=True)
        with wandb.init(entity=engine.ENTITY,project=engine.PROJECT,group=GROUP,id=rid,name=job['name'],resume='allow',mode='online',job_type='mission2-training-calibration',dir=str(folder),
                        config={**job,'mission':2,'epochs':40,'selection_scope':'selected_eight_channels','assessment_used_for_selection':False},settings=wandb.Settings(init_timeout=90,disable_git=True,save_code=False)) as tracking:
            engine.train(job,folder,run=tracking)
    check()
    records=[]
    for job in plan['jobs']:
        folder=OUTPUT/job['name']
        if not verify_receipt(folder,'trained.json'):raise ValueError('Missing trained receipt')
        records.append(dict(job=job,trained_sha256=digest(folder/'trained.json'),rule=json.loads((folder/'frozen_rule.json').read_text()),model_sha256=digest(folder/'model.pt')))
    joint=OUTPUT/'selection_complete.json';value=dict(selections=records,selection_scope='selected_eight_channels')
    if joint.exists() and json.loads(joint.read_text())!=value:raise ValueError('Joint selection changed')
    if not joint.exists():write_json(joint,value)
    print('All nine selections jointly frozen; preparing assessment.',flush=True)
    if verify_raw()!=p['raw']:raise ValueError('Raw inventory changed before assessment')
    prepare_split('assessment',p['features'],p['assessment_bounds'])
    for job in plan['jobs']:
        check();verify_receipt(PREP/'assessment','complete.json');assess(job)
    metrics=pd.concat([pd.read_csv(OUTPUT/j['name']/'metrics.csv') for j in plan['jobs']],ignore_index=True)
    metrics.to_csv(OUTPUT/'all_results.csv',index=False)
    write_json(OUTPUT/'complete.json',dict(computation_complete=True,independent_verification_pending=True,metric_rows=len(metrics),files={str(f.relative_to(OUTPUT)):digest(f) for f in OUTPUT.rglob('*') if f.is_file() and 'wandb' not in f.relative_to(OUTPUT).parts and f.name!='complete.json'}))
    print('Computation complete; independent verification/delivery pending.',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','run']);args=parser.parse_args()
    handles=[]
    for name in ('.mission2-study.lock','.context-expansion.lock','.combination-training.lock','.wandb-finetuning.lock','.short-no-roll-finetune.lock'):
        h=(ROOT/name).open('a');fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);handles.append(h)
    if args.stage=='prepare':prepare()
    elif not (OUTPUT/'complete.json').exists():run()

if __name__=='__main__':main()
