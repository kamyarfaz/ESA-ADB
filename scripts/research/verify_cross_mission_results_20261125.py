"""Independent replay of v2 grids, selection, inference spots, predictions and IDs.

Reuses the reference-parity-tested event evaluator and previously audited
independent interval/grid helpers. Does not import the production study runner,
threshold selector, postprocessing, resampler or neural score aggregator.
"""
import importlib.util
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import wandb

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from esa_thesis import config
from esa_thesis.models import MultivariateAE
from esa_thesis.evaluation.esa import EventEvaluator
from esa_thesis.development import QUANTILES

HELPER=ROOT/'scripts/research/verify_mission2_study_20261115.py'
spec=importlib.util.spec_from_file_location('independent_audit_helpers',HELPER)
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
read,sha,equal,process,event_ids,replay_grid=h.read,h.sha,h.equal,h.process,h.event_ids,h.replay_grid
RUN=ROOT/'results_longrun/cross_mission_transfer_20261120_v2'
SOURCE=ROOT/'results_longrun/context_expansion_20261104_v1'
OUT=ROOT/'docs/research/tables/cross-mission-2026-11-20-v2'

def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def receipt(folder,name):
    r=read(folder/name)
    for n,d in r['files'].items():assert sha(folder/n)==d,(str(folder),n)
    return r

def load(key,split):
    folder=RUN/'prepared'/key/split
    meta=read(folder/'metadata.json')
    x=np.load(folder/'values.npy',mmap_mode='r');t=np.load(folder/'times.npy')
    assert x.shape==(len(t),len(meta['features']))
    assert (np.diff(t)==meta['cadence_seconds']*10**9).all()
    return x,t,meta['features']

def annotations(m):
    folder=ROOT/f'data/ESA-Mission{m}'
    return pd.read_csv(folder/'labels.csv').merge(pd.read_csv(folder/'anomaly_types.csv'),on='ID',validate='many_to_one')

def refit(key):
    x,t,features=load(key,'train')
    excluded=np.load(RUN/'prepared'/key/'train/excluded.npy')!=0
    med=np.empty(len(features),np.float32);iqr=med.copy()
    for j in range(len(features)):
        v=x[~excluded,j];assert np.isfinite(v).all()
        med[j]=np.median(v);q75,q25=np.percentile(v,[75,25]);iqr[j]=1 if abs(q75-q25)<1e-8 else q75-q25
    prefix=np.r_[0,np.cumsum(excluded)];starts=np.arange(0,len(t)-255,16)
    count=int((prefix[starts+256]==prefix[starts]).sum())
    assert count=={'m1':349383,'m2':98224,'transfer_m2':57768}[key]
    return med,iqr,features,count

@torch.no_grad()
def inference(checkpoint,raw,times,columns,stored,expected_scaler,annual=False,mapped=None,override=None):
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    features=ck['features'];ix=[columns.index(c) for c in (mapped or features)]
    for k,v in zip(('scaler_median','scaler_iqr'),expected_scaler):assert np.array_equal(ck[k],v),(str(checkpoint),k)
    assert ck['masked'] is False
    med,iqr=override if override is not None else expected_scaler
    model=MultivariateAE(len(features)).to('cuda');model.load_state_dict(ck['model_state_dict'],strict=True);model.eval()
    n=len(times);spots={0,127,255,256,n//2//32*32+7,n-257,n-1}
    if annual:
        for year in range(pd.Timestamp(times[0]).year,pd.Timestamp(times[-1]).year+1):
            a=int(np.searchsorted(times,pd.Timestamp(f'{year}-01-01').value));b=int(np.searchsorted(times,pd.Timestamp(f'{year+1}-01-01').value))
            spots.update([min(a,n-1),min(a+255,n-1),min((a+b)//2,n-1),b-1])
    starts=list(range(0,n-255,32))
    if starts[-1]!=n-256:starts.append(n-256)
    windows={p:[w for w,s in enumerate(starts) if s<=p<s+256] for p in sorted(spots)}
    batches=sorted({w//32*32 for ws in windows.values() for w in ws});errors={}
    for first in batches:
        ids=list(range(first,min(first+32,len(starts))));seqs=[]
        for w in ids:
            x=(raw[starts[w]:starts[w]+256,ix].copy()-med)/iqr
            x=np.clip(np.nan_to_num(x,nan=0.,posinf=0.,neginf=0.),-config.CLIP_Z,config.CLIP_Z)
            seqs.append(x.T[:,:,None])
        inputs=torch.from_numpy(np.stack(seqs)).to('cuda').contiguous()
        mse=(model(inputs)-inputs).square().mean(dim=(2,3)).cpu().numpy();errors.update(zip(ids,mse))
    rows=[]
    for p,ws in windows.items():
        total=np.zeros(len(features),np.float32)
        for w in ws:total+=errors[w]
        actual=np.max(total/np.float32(len(ws)))
        assert actual==stored[p],(str(checkpoint),p,float(actual),float(stored[p]))
        rows.append(dict(checkpoint=str(checkpoint.relative_to(ROOT)),sample=p,timestamp=str(pd.Timestamp(times[p])),window_count=len(ws),saved_score=float(stored[p]),replayed_score=float(actual),exact_match=True))
    del model;torch.cuda.empty_cache()
    return rows

def main():
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    plan=read(RUN/'frozen/protocol.json');prep=read(RUN/'preparation.json');sel=read(RUN/'input-selection.json')
    assert len(plan['jobs'])==18 and plan['batch']==plan['score_batch']==32 and plan['selection_scope']=='system_all_channels'
    for n,d in plan['source_sha256'].items():assert sha(ROOT/n)==d,n
    runtime=dict(python=platform.python_version(),torch=str(torch.__version__),numpy=np.__version__,pandas=pd.__version__,wandb=wandb.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),threads=torch.get_num_threads(),tf32=torch.backends.cuda.matmul.allow_tf32)
    assert runtime==plan['runtime']
    assert {k:str(v) for k,v in vars(config).items() if k.isupper()}==plan['settings']
    assert sha(RUN/'preparation.json')==plan['preparation_sha256']
    assert sha(RUN/'preflight-verification.json')==plan['preflight_sha256']
    assert sha(RUN/'input-selection.json')==plan['selection_sha256']
    assert sha(RUN/'target-normalization.json')==prep['normalization_sha256']
    assert sha(RUN/'preparation_frozen/protocol.json')==prep['plan_sha256']
    for n,d in prep['raw'].items():assert sha(ROOT/n)==d,n
    complete=receipt(RUN,'complete.json');assert complete['computation_complete'] and complete['metric_rows']==720
    for f in sorted((RUN/'prepared').glob('*/*/complete.json')):receipt(f.parent,'complete.json')
    for key,d in prep['splits'].items():assert sha(RUN/'prepared'/key/'complete.json')==d
    assert (ROOT/'scripts/research/esa-cross-mission-study-v2.service').read_bytes()==Path('/home/k_faz/.config/systemd/user/esa-cross-mission-study-v2.service').read_bytes()
    joint=read(RUN/'selection_complete.json');freeze=(RUN/'selection_complete.json').stat().st_mtime
    assert len(joint['selections'])==18 and len(joint['sources'])==9
    assert joint['sources']==sel['sources'] and joint['mapping']==sel['mapping']
    assert joint['normalization_sha256']==prep['normalization_sha256']
    for f in (RUN/'prepared').glob('*/assessment/*'):assert f.stat().st_mtime>freeze,str(f)
    print('All source/raw/prepared/completion fingerprints and joint assessment gate passed',flush=True)
    scalers={key:refit(key) for key in ('m1','m2','transfer_m2')}
    norm=read(RUN/'target-normalization.json');tm,ti,tf,_=scalers['transfer_m2']
    assert norm['features']==tf and np.array_equal(tm,np.asarray(norm['median'],np.float32)) and np.array_equal(ti,np.asarray(norm['iqr'],np.float32))
    cal_rows=[];spots=[];grids=0;candidates=0
    for job,selection in zip(plan['jobs'],joint['selections']):
        f=RUN/job['name'];trained=receipt(f,'trained.json');receipt(f,'assessed.json')
        assert job==trained['job']==selection['job'] and sha(f/'trained.json')==selection['trained_sha256']
        assert (f/'trained.json').stat().st_mtime<freeze<(f/'assessment_score.npy').stat().st_mtime
        hist=pd.read_csv(f/'history.csv').set_index('epoch');assert hist.index.tolist()==list(range(1,41)) and np.isfinite(hist.loss).all()
        terminal=torch.load(f/'last.pt',map_location='cpu',weights_only=False)
        assert terminal['epoch']==41 and terminal['batch']==0 and terminal['job']==job and len(terminal['history'])==40
        for row in terminal['history']:
            equal(row['loss'],hist.loc[row['epoch'],'loss'])
            for n,d in row.get('candidate_files',{}).items():assert trained['files'][n]==d
        key=f'm{job["mission"]}';raw,times,columns=load(key,'calibration');med,iqr,features,_=scalers[key]
        ix=[features.index(c) for c in job['features']];expected=(med[ix],iqr[ix])
        assert np.array_equal(terminal['median'],expected[0]) and np.array_equal(terminal['iqr'],expected[1])
        ev=EventEvaluator(times,annotations(job['mission']))
        epochs=[25,30,35,40] if job['method']=='pretrain_finetune' else list(range(5,41,5))
        assert sorted(x.name for x in f.glob('epoch_*_grid.csv'))==[f'epoch_{e:02d}_grid.csv' for e in epochs]
        choices=[]
        for ep in epochs:
            stem=f'epoch_{ep:02d}';score=np.load(f/(stem+'_calibration_score.npy'))
            assert score.dtype==np.float32 and len(score)==len(times) and np.isfinite(score).all()
            grid,rule=replay_grid(score,ev,QUANTILES)
            pd.testing.assert_frame_equal(grid,pd.read_csv(f/(stem+'_grid.csv')),check_dtype=False,rtol=1e-11,atol=1e-9)
            equal(rule['val_esa_f05'],hist.loc[ep,'calibration_target_f05'])
            ck=torch.load(f/(stem+'.pt'),map_location='cpu',weights_only=False);assert ck['features']==job['features']
            spots.extend(inference(f/(stem+'.pt'),raw,times,columns,score,expected))
            choices.append((ep,rule));grids+=1;candidates+=len(grid)
        best=max(choices,key=lambda x:x[1]['val_esa_f05']);frozen=read(f/'frozen_rule.json')
        assert best[0]==frozen['epoch'] and frozen==selection['rule']==terminal['best']
        for k,v in best[1].items():equal(v,frozen['rule'][k],k)
        assert sha(f/'model.pt')==sha(f/f'epoch_{best[0]:02d}.pt')==selection['model_sha256']
        assert sha(f/'calibration_score.npy')==sha(f/f'epoch_{best[0]:02d}_calibration_score.npy')
        assert sha(f/'calibration_candidates.csv')==sha(f/f'epoch_{best[0]:02d}_grid.csv')
        for ep,r in choices:cal_rows.append(dict(name=job['name'],mission=job['mission'],method=job['method'],seed=job['seed'],epoch=ep,selected=ep==best[0],**r))
        pd.DataFrame(cal_rows).to_csv(OUT/'verified_calibration.csv',index=False)
        pd.DataFrame(spots).to_csv(OUT/'inference_spots.csv',index=False)
        print(job['name'],'all calibration grids, earliest selection and exact inference passed',flush=True)
    index=['name','condition','period','event_definition','scope']
    saved=pd.read_csv(RUN/'all_results.csv',dtype={'period':str}).set_index(index)
    assert len(saved)==720 and saved.index.is_unique
    work=[]
    for job in plan['jobs']:
        key=f'm{job["mission"]}';med,iqr,features,_=scalers[key];ix=[features.index(c) for c in job['features']]
        work.append((RUN/job['name'],RUN/job['name']/'model.pt',job,key,sel['panels'][str(job['mission'])]['base'],'fresh_training',read(RUN/job['name']/'frozen_rule.json')['rule'],(med[ix],iqr[ix]),None,None))
    assert sha(SOURCE/'selection_complete.json')==sel['historical_selection_sha256']
    for source in joint['sources']:
        job=source['job'];f=SOURCE/job['name'];ck=torch.load(f/'model.pt',map_location='cpu',weights_only=False)
        assert sha(f/'model.pt')==source['model_sha256'] and sha(f/'trained.json')==source['trained_sha256']
        assert sha(f/'frozen_rule.json')==source['rule_sha256'] and read(f/'frozen_rule.json')==source['rule']
        assert ck['features']==job['features'] and source['mapped_features']==[sel['mapping'][c] for c in job['features']]
        base=[sel['mapping'][f'channel_{i}'] for i in range(40,48)]
        mapped=source['mapped_features'];ix=[tf.index(c) for c in mapped]
        for condition in ('frozen_source_scaling','target_training_normalization'):
            override=(tm[ix],ti[ix]) if condition=='target_training_normalization' else None
            identity=dict(name=job['name'],mission=2,method=job['method'],seed=job['seed'],arm='source_context_transfer')
            work.append((RUN/'transfer'/job['name']/condition,f/'model.pt',identity,'transfer_m2',base,condition,source['rule']['rule'],(ck['scaler_median'],ck['scaler_iqr']),mapped,override))
    rows=[];events=[]
    for folder,checkpoint,job,key,base,condition,rule,expected,mapped,override in work:
        receipt(folder,'assessed.json');raw,times,columns=load(key,'assessment')
        score=np.load(folder/'assessment_score.npy');pred=process(score>rule['threshold'],int(rule['merge_gap']),int(rule['min_dur']))
        assert len(score)==len(times) and score.dtype==np.float32 and np.isfinite(score).all()
        assert np.array_equal(pred,np.load(folder/'assessment_prediction.npy'))
        new=inference(checkpoint,raw,times,columns,score,expected,True,mapped,override)
        for row in new:row['condition']=condition
        spots.extend(new)
        ann=annotations(job['mission']);local=ann.copy();local.loc[~local.Channel.isin(base),'Category']='Out-of-scope (neutral)'
        years=range(pd.Timestamp(times[0]).year,pd.Timestamp(times[-1]).year+1)
        periods={str(y):(int(np.searchsorted(times,pd.Timestamp(f'{y}-01-01').value)),int(np.searchsorted(times,pd.Timestamp(f'{y+1}-01-01').value))) for y in years};periods['pooled']=(0,len(times))
        by_job=pd.read_csv(folder/'metrics.csv',dtype={'period':str}).set_index(['period','event_definition','scope'])
        stored_ids=read(folder/'event_ids.json');assert len(stored_ids)==len(by_job)==len(periods)*4
        for period,(a,b) in periods.items():
            for definition,categories in [('anomaly_and_rare',('Anomaly','Rare Event')),('anomaly_only',('Anomaly',))]:
                for scope,labels in [('system_all_channels',ann),('fixed_base_channels',local)]:
                    ev=EventEvaluator(times[a:b],labels,categories);metrics=ev.score(pred[a:b])
                    identity={k:job[k] for k in ('name','mission','method','seed','arm')}
                    for k,v in metrics.items():
                        equal(v,saved.loc[(job['name'],condition,period,definition,scope),k],(job['name'],condition,period,scope,k))
                        equal(v,by_job.loc[(period,definition,scope),k])
                    ids=event_ids(ev,pred[a:b]);assert ids==stored_ids[period+'/'+definition+'/'+scope]
                    assert len(ids['detected'])==metrics['TPe'] and len(ids['missed'])==metrics['FNe']
                    rows.append(dict(**identity,condition=condition,period=period,event_definition=definition,scope=scope,**metrics))
                    for status,idlist in ids.items():events.extend(dict(**identity,condition=condition,period=period,event_definition=definition,scope=scope,event_id=aid,detected=status=='detected') for aid in idlist)
        print(job['name'],condition,'assessment inference, predictions, metrics and IDs passed',flush=True)
    assert grids==120 and len(rows)==720
    pd.DataFrame(rows).to_csv(OUT/'verified_results.csv',index=False);pd.DataFrame(events).to_csv(OUT/'verified_event_ids.csv',index=False);pd.DataFrame(spots).to_csv(OUT/'inference_spots.csv',index=False)
    save('verification.json',dict(verified=True,verified_at_utc=datetime.now(timezone.utc).isoformat(),complete_receipt_sha256=sha(RUN/'complete.json'),verifier_sha256=sha(Path(__file__)),helper_sha256=sha(HELPER),completion_files=len(complete['files']),raw_files=len(prep['raw']),jobs=18,transfer_conditions=18,calibration_grids=grids,calibration_candidate_rows=candidates,metric_rows=len(rows),event_id_rows=len(events),inference_spots=len(spots),inference_exact_matches=len(spots),inference_passes=156,joint_freeze_utc=datetime.fromtimestamp(freeze,timezone.utc).isoformat(),nominal_training_windows={key:value[3] for key,value in scalers.items()},checks=['Full frozen source/runtime, raw, preparation and completed-artifact hashes','Independent scaler refits, all 120 calibration grids and within/between epoch selections','Joint freeze precedes every assessment array','Original batch32 GPU checkpoint inference spots, exact float32 equality','Full prediction replay; all720 annual/pooled corrected metrics and detected/missed IDs','All nine historical source models and source rules, mapping and both normalization conditions unchanged'],limitations=['Inference spots, not a second full neural pass','Corrected evaluator reused with bundled-reference parity tests; independent postprocessing/window/ID enumeration','Raw source resampling independently checked in separate preparation audit','Three seeds and reused benchmarks; no independent generalization claim']))
    print('VERIFIED 720 METRIC ROWS',flush=True)

if __name__=='__main__':main()
