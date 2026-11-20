"""Broad whole-system training and fixed source transfer, November 2026."""
import argparse
import fcntl
import gc
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from . import config, context_expansion as engine
from . import mission2_study as sampling
from .calibration_audit import digest, detection_ids
from .data import RobustChannelScaler
from .development import QUANTILES
from .development_recovery import check_protocol
from .evaluation.esa import EventEvaluator
from .full_combinations import load_model, nominal_starts
from .full_similarity_mosad import write_json, save_npy, verify_receipt
from .thresholds import sweep_thresholds, select_rows, postprocess

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'results_longrun/cross_mission_transfer_20261120_v2'
RESEARCH=ROOT/'results_longrun/cross_mission_transfer_20261118_v1/research'
PREP=OUTPUT/'prepared'
CORR=ROOT/'results_longrun/all_channel_correlations_20261102_v1'
SOURCE=ROOT/'results_longrun/context_expansion_20261104_v1'
PROTOCOL=ROOT/'docs/research/cross-mission-whole-system-protocol-v2-2026-11-20.md'
GROUP='cross-mission-whole-system-20261120-v2'
BATCH=32
SPLITS={1:{'train':('2000-01-01','2006-01-01'),'calibration':('2006-01-01','2007-01-01'),'assessment':('2007-01-01','2014-01-01')},
        2:{'train':('2000-01-01','2001-01-01'),'calibration':('2001-01-01','2001-10-01'),'assessment':('2001-10-01','2003-07-01')}}


def number(ch):return int(ch.split('_')[1])


def medoid_panel(metadata, inventory, channels, level, changes):
    """One nominal medoid and one change-related companion per target group."""
    base=[];extra=[];ranks=[]
    for group,g in metadata[metadata.Target=='YES'].groupby('Group',sort=True):
        members=sorted(g.Channel,key=number);ix=[channels.index(c) for c in members]
        if any(not inventory.loc[c,'numeric'] or inventory.loc[c,'constant'] for c in members):
            raise ValueError('Target group includes unsupported channel type')
        l=np.abs(np.nan_to_num(level[np.ix_(ix,ix)],nan=0.));d=np.abs(np.nan_to_num(changes[np.ix_(ix,ix)],nan=0.))
        strength=(l+d)/2;np.fill_diagonal(strength,0)
        central=strength.sum(axis=1)/(len(ix)-1) if len(ix)>1 else np.ones(1)
        order=sorted(range(len(ix)),key=lambda j:(-central[j],-float(inventory.loc[members[j],'coverage']),number(members[j])))
        anchor=order[0];base.append(members[anchor]);companion=None
        if len(ix)>1:
            companion=sorted([j for j in range(len(ix)) if j!=anchor],key=lambda j:(-d[anchor,j],-l[anchor,j],-float(inventory.loc[members[j],'coverage']),number(members[j])))[0]
            extra.append(members[companion])
        for j,c in enumerate(members):
            ranks.append(dict(group=int(group),channel=c,medoid_score=float(central[j]),coverage=float(inventory.loc[c,'coverage']),
                              selected_base=j==anchor,selected_extra=j==companion,anchor=members[anchor],
                              abs_level_to_anchor=float(l[anchor,j]),abs_changes_to_anchor=float(d[anchor,j])))
    return dict(base=base,expanded=base+extra,rankings=ranks)


def selection():
    if not verify_receipt(RESEARCH,'complete.json'):raise ValueError('Missing completed channel research')
    receipt=json.loads((CORR/'complete.json').read_text());hashes={r['path']:r['sha256'] for r in receipt['files']}
    inv=pd.read_csv(CORR/'channel_inventory.csv')
    if digest(CORR/'channel_inventory.csv')!=hashes['channel_inventory.csv']:raise ValueError('Inventory changed')
    panels={}
    for m in (1,2):
        name=f'mission{m}_block_matrices.npz'
        if digest(CORR/name)!=hashes[name]:raise ValueError('Matrices changed')
        with np.load(CORR/name,allow_pickle=False) as z:
            channels=list(z['channels']);meta=pd.read_csv(ROOT/f'data/ESA-Mission{m}/channels.csv')
            # All-NaN non-target matrix entries are irrelevant to the numeric target groups.
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                level=np.nanmedian(z['pearson'][1:],axis=0);changes=np.nanmedian(z['delta'][1:],axis=0)
        panels[str(m)]=medoid_panel(meta,inv[inv.mission==m].set_index('channel'),channels,level,changes)
        assert len(panels[str(m)]['base'])==13 and len(panels[str(m)]['expanded'])==(25 if m==1 else 26)
    mapping=json.loads((RESEARCH/'mapping.json').read_text())['primary']
    selected=json.loads((SOURCE/'selection_complete.json').read_text())['selections']
    sources=[]
    for r in selected:
        if r['job']['arm']!='context':continue
        folder=SOURCE/r['job']['name']
        if digest(folder/'trained.json')!=r['trained_sha256'] or digest(folder/'model.pt')!=r['model_sha256']:
            raise ValueError('Source checkpoint receipt changed')
        if json.loads((folder/'frozen_rule.json').read_text())!=r['rule']:raise ValueError('Source rule changed')
        features=r['job']['features'];mapped=[mapping[c] for c in features]
        if len(set(mapped))!=len(mapped):raise ValueError('Duplicated transfer input')
        sources.append(dict(**r,mapped_features=mapped,rule_sha256=digest(folder/'frozen_rule.json')))
    if len(sources)!=9:raise ValueError('Expected all nine context-arm source models')
    return dict(panels=panels,mapping=mapping,sources=sources,research_sha256=digest(RESEARCH/'complete.json'),
                historical_selection_sha256=digest(SOURCE/'selection_complete.json'),calibration_used=False,assessment_used=False)


def jobs(sel):
    return [dict(name=f'm{m}_{method}_s{seed}',mission=m,method=method,seed=seed,
                 arm='expanded' if method=='expanded_short_no_roll' else 'base',
                 features=sel['panels'][str(m)]['expanded' if method=='expanded_short_no_roll' else 'base'])
            for m in (2,1) for method in engine.METHODS for seed in engine.SEEDS]


def source_hashes():
    paths=list((ROOT/'esa_thesis').rglob('*.py'))+[PROTOCOL,ROOT/'docs/research/cross-mission-transfer-protocol-2026-11-18.md',
        ROOT/'scripts/research/run_cross_mission_study_v2.sh',ROOT/'scripts/research/esa-cross-mission-study-v2.service',
        ROOT/'scripts/research/verify_cross_mission_preparation_20261120.py',
        ROOT/'scripts/research/verify_cross_mission_research_20261118.py',
        ROOT/'scripts/research/preflight_cross_mission_gpu_20261120.py',
        ROOT/'scripts/research/cross_mission_channel_audit_20261118.py']+list((ROOT/'tests/thesis').glob('test_*.py'))
    return {str(p.relative_to(ROOT)):digest(p) for p in paths}


def verify_raw():
    raw=json.loads((RESEARCH/'integrity.json').read_text())['raw']
    for p,sha in raw.items():
        if digest(ROOT/p)!=sha:raise ValueError('Raw file changed: '+p)
    return raw


def labels(mission):
    p=ROOT/f'data/ESA-Mission{mission}'
    return pd.read_csv(p/'labels.csv').merge(pd.read_csv(p/'anomaly_types.csv'),on='ID',validate='many_to_one')


def maximum_age_seconds(mission,channel):
    """Approved v2 exception applies only to these two Mission1 inputs."""
    return 900 if mission==1 and channel in ('channel_62','channel_63') else 600


def resample(raw_times,raw_values,grid,start,end,mission,channel):
    """Causal split-contained lookup with the fixed, channel-specific age limit."""
    keep=(raw_times>=start)&(raw_times<end)
    rt=raw_times[keep];rv=np.asarray(raw_values)[keep]
    if len(rt)==0:raise ValueError('Empty raw split')
    if np.any(np.diff(rt)<=0):raise ValueError('Duplicate/nonmonotone raw timestamps')
    ix=np.searchsorted(rt,grid,side='right')-1
    safe=np.maximum(ix,0);age=grid-rt[safe]
    valid=(ix>=0)&(age>=0)&(age<=maximum_age_seconds(mission,channel)*10**9)&np.isfinite(rv[safe])
    result=np.full(len(grid),np.nan,np.float32);result[valid]=rv[safe[valid]]
    age=np.where(ix>=0,age,-1)
    return result,valid,age,rt[safe]


def prepare_split(key,mission,split,features,step):
    folder=PREP/key/split
    if verify_receipt(folder,'complete.json'):return
    folder.mkdir(parents=True,exist_ok=True)
    bounds=SPLITS[mission][split];start,end=[pd.Timestamp(x).value for x in bounds]
    times=np.arange(start,end,step*10**9,dtype=np.int64);shape=(len(times),len(features))
    arrays={n:np.lib.format.open_memmap(folder/(n+'.npy.tmp'),mode='w+',dtype=dtype,shape=shape)
            for n,dtype in [('values',np.float32),('available',np.bool_),('age_ns',np.int64)]}
    annotations=labels(mission)
    excluded=sampling.interval_mask(times,annotations,step*10**9) if split=='train' else np.zeros(len(times),bool)
    for j,ch in enumerate(features):
        raw=pd.read_pickle(ROOT/f'data/ESA-Mission{mission}/channels/{ch}.zip')
        if raw.shape[1]!=1 or not pd.api.types.is_numeric_dtype(raw.iloc[:,0]):raise ValueError('Unsupported source '+ch)
        rt=pd.DatetimeIndex(raw.index).as_unit('ns').asi8;rv=raw.iloc[:,0].to_numpy()
        values,available,ages,source=resample(rt,rv,times,start,end,mission,ch)
        arrays['values'][:,j]=values;arrays['available'][:,j]=available;arrays['age_ns'][:,j]=ages
        excluded|=~available
        if split=='train':excluded|=sampling.interval_mask(source,annotations[annotations.Channel==ch])
        print('Prepared channel',key,split,ch,flush=True)
        del raw,rt,rv,values,available,ages,source;gc.collect()
    availability=np.array([arrays['available'][:,j].mean() for j in range(len(features))])
    for n in list(arrays):arrays[n].flush();del arrays[n];(folder/(n+'.npy.tmp')).replace(folder/(n+'.npy'))
    save_npy(folder/'times.npy',times);save_npy(folder/'excluded.npy',excluded.astype(np.int8))
    write_json(folder/'metadata.json',dict(key=key,mission=mission,split=split,features=features,bounds=bounds,
               cadence_seconds=step,max_age_seconds=[maximum_age_seconds(mission,c) for c in features],rows=len(times),availability=availability.tolist(),complete_nominal_rows=int((~excluded).sum()) if split=='train' else None))
    names=['values.npy','available.npy','age_ns.npy','times.npy','excluded.npy','metadata.json']
    write_json(folder/'complete.json',dict(files={n:digest(folder/n) for n in names}))


def load_split(key,split,features):
    folder=PREP/key/split;meta=json.loads((folder/'metadata.json').read_text())
    x=np.load(folder/'values.npy',mmap_mode='r')[:,[meta['features'].index(c) for c in features]].copy()
    return x,np.load(folder/'excluded.npy'),np.load(folder/'times.npy')


def require_resources():
    available=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
    if available<6*2**30:raise RuntimeError('Need6GiB available host RAM')
    if shutil.disk_usage(ROOT).free<80*2**30:raise RuntimeError('Need80GiB disk free')
    engine.require_resources()


def runtime():
    import wandb
    return dict(python=platform.python_version(),torch=str(torch.__version__),numpy=np.__version__,pandas=pd.__version__,
                wandb=wandb.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),threads=torch.get_num_threads(),
                tf32=torch.backends.cuda.matmul.allow_tf32)


def preparation_plan(sel):
    test=json.loads((OUTPUT/'test-verification.json').read_text())
    audit=json.loads((OUTPUT/'research-verification.json').read_text())
    gpu=json.loads((OUTPUT/'gpu-preflight.json').read_text())
    if not test['passed'] or test['source_sha256']!=source_hashes():raise ValueError('Tests missing or source changed since tests')
    if not audit['passed'] or audit['research_sha256']!=digest(RESEARCH/'complete.json') or audit['verifier_sha256']!=digest(ROOT/'scripts/research/verify_cross_mission_research_20261118.py'):raise ValueError('Research verification missing/changed')
    if not gpu['passed'] or gpu['script_sha256']!=digest(ROOT/'scripts/research/preflight_cross_mission_gpu_20261120.py') or gpu['gpu']!=torch.cuda.get_device_name(0):raise ValueError('GPU preflight missing/changed')
    return dict(experiment=GROUP,jobs=jobs(sel),selection_sha256=digest(OUTPUT/'input-selection.json'),
                source_sha256=source_hashes(),settings={k:str(v) for k,v in vars(config).items() if k.isupper()},runtime=runtime(),
                prelaunch_receipts={n:digest(OUTPUT/n) for n in ('test-verification.json','research-verification.json','gpu-preflight.json')},
                splits=SPLITS,batch=BATCH,epochs=40,score_batch=BATCH,selection_scope='system_all_channels',
                max_age_seconds={'default':600,'Mission1/channel_62':900,'Mission1/channel_63':900},
                amendment='Explicit user approval20 November2026; preserve failed v1 unchanged')


def prepare():
    OUTPUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);require_resources()
    sel=selection();path=OUTPUT/'input-selection.json'
    if path.exists() and json.loads(path.read_text())!=sel:raise ValueError('Input selection changed')
    if not path.exists():write_json(path,sel)
    plan=preparation_plan(sel);freeze=OUTPUT/'preparation_frozen'
    check_protocol(freeze,plan,freeze.exists());raw=verify_raw()
    specs=[(f'm{m}',m,split,sel['panels'][str(m)]['expanded'],30 if m==1 else 18) for m in (2,1) for split in ('train','calibration')]
    transfer_features=list(sel['mapping'].values())
    specs.append(('transfer_m2',2,'train',transfer_features,30))
    for args in specs:prepare_split(*args)
    x,y,_=load_split('transfer_m2','train',transfer_features)
    scaler=RobustChannelScaler().fit(x,y)
    write_json(OUTPUT/'target-normalization.json',dict(features=transfer_features,median=scaler.median_.tolist(),iqr=scaler.iqr_.tolist(),fit_split='Mission2 training2000 only'))
    del x,y;gc.collect()
    p=dict(raw=raw,selection_sha256=digest(path),plan_sha256=digest(freeze/'protocol.json'),
           normalization_sha256=digest(OUTPUT/'target-normalization.json'),
           splits={f'{a[0]}/{a[2]}':digest(PREP/a[0]/a[2]/'complete.json') for a in specs})
    receipt=OUTPUT/'preparation.json'
    if receipt.exists() and json.loads(receipt.read_text())!=p:raise ValueError('Preparation changed')
    if not receipt.exists():write_json(receipt,p)
    print('PREPARATION COMPLETE; no assessment arrays prepared.',flush=True)


def verify_prepared():
    p=json.loads((OUTPUT/'preparation.json').read_text())
    if p['selection_sha256']!=digest(OUTPUT/'input-selection.json') or p['normalization_sha256']!=digest(OUTPUT/'target-normalization.json'):raise ValueError('Selection/normalization changed')
    if p['plan_sha256']!=digest(OUTPUT/'preparation_frozen/protocol.json'):raise ValueError('Preparation plan changed')
    for key,sha in p['splits'].items():
        folder=PREP/key
        if digest(folder/'complete.json')!=sha or not verify_receipt(folder,'complete.json'):raise ValueError('Prepared arrays changed')
    return p


def scopes(mission,times,base,anomaly_only=False):
    ann=labels(mission);categories=('Anomaly',) if anomaly_only else ('Anomaly','Rare Event')
    local=ann.copy();local.loc[~local.Channel.isin(base),'Category']='Out-of-scope (neutral)'
    return {'system_all_channels':EventEvaluator(times,ann,categories),'fixed_base_channels':EventEvaluator(times,local,categories)}


def choose_rule(mission,base,scores,times):
    ev=scopes(mission,times,base)['system_all_channels']
    if not ev.by_id:raise ValueError('No system calibration events')
    grid=sweep_thresholds(scores,evaluator=ev,threshold_values=np.percentile(scores,QUANTILES),merge_gaps=[0,16,64],min_durs=[1,8,32])
    return select_rows(grid).iloc[0].to_dict(),grid


def report(folder,identity,mission,base,times,scores,rule):
    pred=postprocess(scores>rule['threshold'],int(rule['merge_gap']),int(rule['min_dur']))
    save_npy(folder/'assessment_score.npy',scores);save_npy(folder/'assessment_prediction.npy',pred)
    years=pd.DatetimeIndex(times).year;rows=[];ids={}
    for period,keep in [(str(y),years==y) for y in sorted(set(years))]+[('pooled',np.ones(len(times),bool))]:
        for only in (False,True):
            definition='anomaly_only' if only else 'anomaly_and_rare'
            for scope,ev in scopes(mission,times[keep],base,only).items():
                rows.append(dict(**identity,period=period,scope=scope,event_definition=definition,**ev.score(pred[keep])))
                hits=sorted(detection_ids(ev,pred[keep]));ids[period+'/'+definition+'/'+scope]=dict(detected=hits,missed=sorted(set(ev.by_id)-set(hits)))
    pd.DataFrame(rows).to_csv(folder/'metrics.csv',index=False);write_json(folder/'event_ids.json',ids)
    names=['assessment_score.npy','assessment_prediction.npy','metrics.csv','event_ids.json']
    write_json(folder/'assessed.json',dict(files={n:digest(folder/n) for n in names}))


def assess_fresh(job,sel):
    folder=OUTPUT/job['name']
    if verify_receipt(folder,'assessed.json'):return
    model,scaler,features,_=load_model(folder/'model.pt','cuda')
    x,_,times=load_split(f'm{job["mission"]}','assessment',features);engine.scale_inplace(x,scaler)
    scores=sampling.score(model,x,'cuda',BATCH)
    identity={k:job[k] for k in ('name','mission','method','seed','arm')};identity['condition']='fresh_training'
    report(folder,identity,job['mission'],sel['panels'][str(job['mission'])]['base'],times,scores,json.loads((folder/'frozen_rule.json').read_text())['rule'])
    del x,model;gc.collect();torch.cuda.empty_cache()


def assess_transfer(source,sel,condition):
    job=source['job'];folder=OUTPUT/'transfer'/job['name']/condition;folder.mkdir(parents=True,exist_ok=True)
    if verify_receipt(folder,'assessed.json'):return
    model,scaler,features,_=load_model(SOURCE/job['name']/'model.pt','cuda')
    mapped=[sel['mapping'][c] for c in features]
    x,_,times=load_split('transfer_m2','assessment',mapped)
    if condition=='target_training_normalization':
        norm=json.loads((OUTPUT/'target-normalization.json').read_text());ix=[norm['features'].index(c) for c in mapped]
        scaler.median_=np.asarray(norm['median'],np.float32)[ix];scaler.iqr_=np.asarray(norm['iqr'],np.float32)[ix]
    engine.scale_inplace(x,scaler);scores=sampling.score(model,x,'cuda',BATCH)
    identity=dict(name=job['name'],mission=2,method=job['method'],seed=job['seed'],arm='source_context_transfer',condition=condition)
    base=[sel['mapping'][f'channel_{n}'] for n in range(40,48)]
    report(folder,identity,2,base,times,scores,source['rule']['rule'])
    del x,model;gc.collect();torch.cuda.empty_cache()


def run():
    torch.set_num_threads(2);require_resources();p=verify_prepared();sel=selection()
    check_protocol(OUTPUT/'preparation_frozen',preparation_plan(sel),True)
    preflight=json.loads((OUTPUT/'preflight-verification.json').read_text())
    if not preflight['passed'] or preflight['preparation_sha256']!=digest(OUTPUT/'preparation.json') or preflight['verifier_sha256']!=digest(ROOT/'scripts/research/verify_cross_mission_preparation_20261120.py'):raise ValueError('Independent preflight missing/changed')
    if verify_raw()!=p['raw']:raise ValueError('Raw inventory changed')
    plan=dict(**preparation_plan(sel),preflight_sha256=digest(OUTPUT/'preflight-verification.json'),preparation_sha256=digest(OUTPUT/'preparation.json'))
    check_protocol(OUTPUT/'frozen',plan,(OUTPUT/'frozen').exists())
    def check():
        if source_hashes()!=plan['source_sha256'] or runtime()!=plan['runtime']:raise ValueError('Frozen source/runtime changed')
        if digest(OUTPUT/'preparation.json')!=plan['preparation_sha256']:raise ValueError('Preparation changed')
        if selection()!=sel:raise ValueError('Channel/source selection changed')
        verify_prepared()
        require_resources()
    import wandb
    engine.BATCH=BATCH;engine.score_bounded=lambda model,x,device,batch_size=BATCH:sampling.score(model,x,device,batch_size)
    for i,job in enumerate(plan['jobs'],1):
        folder=OUTPUT/job['name'];folder.mkdir(exist_ok=True)
        if verify_receipt(folder,'trained.json'):continue
        check();m=job['mission'];base=sel['panels'][str(m)]['base']
        engine.load_split=lambda split,features:load_split(f'm{m}',split,features)
        engine.choose_rule=lambda scores,times:choose_rule(m,base,scores,times)
        rid=f'xm1120v2-{i:02d}-m{m}-s{job["seed"]}'
        write_json(folder/'wandb_run.json',dict(id=rid,url=f'https://wandb.ai/{engine.ENTITY}/{engine.PROJECT}/runs/{rid}'))
        print('Starting fresh run',i,'of18',job['name'],flush=True)
        with wandb.init(entity=engine.ENTITY,project=engine.PROJECT,group=GROUP,id=rid,name=job['name'],resume='allow',mode='online',job_type='whole-system-training-calibration',dir=str(folder),
             config={**job,'epochs':40,'batch':BATCH,'selection_scope':'system_all_channels','assessment_used_for_selection':False},settings=wandb.Settings(init_timeout=90,disable_git=True,save_code=False)) as tracking:
            engine.train(job,folder,run=tracking)
    check();records=[]
    for job in plan['jobs']:
        folder=OUTPUT/job['name']
        if not verify_receipt(folder,'trained.json'):raise ValueError('Missing completed training')
        records.append(dict(job=job,trained_sha256=digest(folder/'trained.json'),rule=json.loads((folder/'frozen_rule.json').read_text()),model_sha256=digest(folder/'model.pt')))
    joint=OUTPUT/'selection_complete.json';value=dict(selections=records,sources=sel['sources'],mapping=sel['mapping'],normalization_sha256=p['normalization_sha256'])
    if joint.exists() and json.loads(joint.read_text())!=value:raise ValueError('Joint freeze changed')
    if not joint.exists():write_json(joint,value)
    print('All18 fresh selections and9 source transfers frozen; preparing assessment.',flush=True)
    if verify_raw()!=p['raw']:raise ValueError('Raw changed before assessment')
    for m in (2,1):prepare_split(f'm{m}',m,'assessment',sel['panels'][str(m)]['expanded'],30 if m==1 else 18)
    prepare_split('transfer_m2',2,'assessment',list(sel['mapping'].values()),30)
    for job in plan['jobs']:check();assess_fresh(job,sel)
    for source in sel['sources']:
        for condition in ('frozen_source_scaling','target_training_normalization'):check();assess_transfer(source,sel,condition)
    paths=[OUTPUT/j['name']/'metrics.csv' for j in plan['jobs']]+list((OUTPUT/'transfer').glob('*/*/metrics.csv'))
    results=pd.concat([pd.read_csv(f) for f in paths],ignore_index=True)
    if len(results)!=720:raise ValueError('Unexpected metric inventory')
    results.to_csv(OUTPUT/'all_results.csv',index=False)
    write_json(OUTPUT/'complete.json',dict(computation_complete=True,independent_verification_pending=True,metric_rows=len(results),
         files={str(f.relative_to(OUTPUT)):digest(f) for f in OUTPUT.rglob('*') if f.is_file() and 'wandb' not in f.relative_to(OUTPUT).parts and f.name!='complete.json'}))
    print('COMPUTATION COMPLETE; independent verification and delivery pending.',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['select','prepare','run','pipeline']);args=parser.parse_args()
    handles=[]
    for name in ('.cross-mission-study.lock','.mission2-study.lock','.context-expansion.lock','.combination-training.lock','.wandb-finetuning.lock','.short-no-roll-finetune.lock'):
        h=(ROOT/name).open('a');fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);handles.append(h)
    if args.stage=='select':print(json.dumps(selection(),indent=2));return
    if args.stage in ('prepare','pipeline'):prepare()
    if args.stage=='pipeline' and not (OUTPUT/'preflight-verification.json').exists():
        subprocess.run([sys.executable,str(ROOT/'scripts/research/verify_cross_mission_preparation_20261120.py')],cwd=ROOT,check=True)
    if args.stage in ('run','pipeline') and not (OUTPUT/'complete.json').exists():run()

if __name__=='__main__':main()
