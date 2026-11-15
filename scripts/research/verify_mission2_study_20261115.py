"""Read-only audit of the frozen nine-run study; writes separate audit evidence.

Does not import production threshold/postprocessing or score aggregation routines.
Reuses the corrected, reference-parity-tested event evaluator and architecture.
Inference uses original contiguous batches, but independently enumerates windows
covering fixed timestamp spots. No model fitting or assessment selection occurs.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from esa_thesis import config
from esa_thesis.development import QUANTILES
from esa_thesis.evaluation.esa import EventEvaluator
from esa_thesis.models import MultivariateAE

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'results_longrun/mission2_three_technique_20261114_v1'
OUT = ROOT / 'docs/research/tables/mission2-three-technique-2026-11-14'
BASE = RUN/'prepared'
FEATURES = ['channel_'+str(i) for i in (58,59,75,76,77,83,84,85,90,86,89)]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def save(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def equal(a, b, context=''):
    if isinstance(a, (str, bool, np.bool_)):
        assert a == b, (context, a, b)
    else:
        assert np.isclose(a, b, rtol=1e-11, atol=1e-9), (context, a, b)


def process(mask, gap, duration):
    """Independent interval-list implementation: join, then filter."""
    hits = np.flatnonzero(mask)
    result = np.zeros(len(mask), np.int8)
    if not len(hits):
        return result
    cuts = np.flatnonzero(np.diff(hits) > gap+1)
    starts = hits[np.r_[0, cuts+1]]
    ends = hits[np.r_[cuts, len(hits)-1]]+1
    for start, end in zip(starts, ends):
        if end-start >= duration:
            result[start:end] = 1
    return result


def event_ids(ev, prediction):
    """Enumerate half-open predicted intervals independently of evaluator helpers."""
    hits = np.flatnonzero(prediction)
    intervals = []
    if len(hits):
        cuts = np.flatnonzero(np.diff(hits)>1)
        for s, e in zip(hits[np.r_[0,cuts+1]], hits[np.r_[cuts,len(hits)-1]]+1):
            intervals.append((ev.timestamps[s], ev.timestamps[min(e,len(prediction)-1)], e==len(prediction)))
    detected = []
    for aid, ranges in ev.by_id.items():
        if any(s<=hi and (e>lo or (closed and e==lo)) for lo,hi in ranges for s,e,closed in intervals):
            detected.append(aid)
    return dict(detected=sorted(detected), missed=sorted(set(ev.by_id)-set(detected)))


def replay_grid(score, ev, quantiles):
    rows = []
    for threshold in np.unique([*np.percentile(score, quantiles), float(score.max())]):
        for gap in (0,16,64):
            for duration in (1,8,32):
                pred = process(score>threshold, gap, duration)
                m = ev.score(pred)
                rows.append(dict(threshold=float(threshold), merge_gap=gap, min_dur=duration,
                    val_esa_f05=m['EW_F_0.50'], val_event_f05=m['EW_F_0.50'],
                    val_event_precision=m['EW_precision'], val_event_recall=m['EW_recall'],
                    val_pred_anomaly_rate=float(pred.mean()), val_saturated=bool(pred.mean()>config.SATURATION_RATE),
                    **{'val_'+k:v for k,v in m.items()}))
    eligible = [r for r in rows if r['val_pred_anomaly_rate']<=.01 and not r['val_saturated']]
    best = min(eligible, key=lambda r: (-r['val_esa_f05'],r['val_false_positive_seconds'],-r['threshold'],r['merge_gap'],r['min_dur'])).copy()
    best.update(selection_name='best_val_esa_f05_val_pred_rate_lte_0.01', selection_type='clean_validation_selected')
    return pd.DataFrame(rows), best


def read_values(split, *bounds, labels=False):
    folder=BASE/split
    for name,h in read(folder/'complete.json')['files'].items():
        assert sha(folder/name)==h,name
    values=np.load(folder/'values.npy',mmap_mode='r')
    times=np.load(folder/'times.npy')
    y=np.load(folder/'excluded.npy')!=0 if labels else None
    assert read(folder/'metadata.json')['features']==FEATURES
    assert (np.diff(times)==18_000_000_000).all()
    return values,times,y


def spots_for(times, annual=False):
    n = len(times)
    spots = {0,127,255,256,n//2//32*32+7,n-257,n-1}
    if annual:
        for year in range(2001,2004):
            a = int(np.searchsorted(times,pd.Timestamp(f'{year}-01-01').value))
            b = int(np.searchsorted(times,pd.Timestamp(f'{year+1}-01-01').value))
            spots.update([min(a,n-1),min(a+255,n-1),min((a+b)//2,n-1),b-1])
    return sorted(spots)


@torch.no_grad()
def inference_spots(checkpoint, raw, times, stored, expected_scaler, annual=False):
    ck = torch.load(checkpoint,map_location='cpu',weights_only=False)
    features = ck['features']
    ix = [FEATURES.index(c) for c in features]
    for key, expected in zip(('scaler_median','scaler_iqr'),expected_scaler):
        assert np.array_equal(ck[key],expected[ix]), (checkpoint.name,key)
    assert ck['masked'] is False
    model = MultivariateAE(len(features)).to('cuda')
    model.load_state_dict(ck['model_state_dict'],strict=True)
    model.eval()
    spots = spots_for(times,annual)
    starts=list(range(0,len(raw)-255,32))
    if starts[-1]!=len(raw)-256:starts.append(len(raw)-256)
    count=len(starts)
    windows={p:[i for i,start in enumerate(starts) if start<=p<start+256] for p in spots}
    batches = sorted({w//64*64 for ws in windows.values() for w in ws})
    errors = {}
    for first in batches:
        ids = list(range(first,min(first+64,count)))
        seqs = []
        for w in ids:
            x = raw[starts[w]:starts[w]+256,ix].copy()
            x = (x-ck['scaler_median'])/ck['scaler_iqr']
            x = np.clip(np.nan_to_num(x,nan=0.,posinf=0.,neginf=0.),-config.CLIP_Z,config.CLIP_Z)
            seqs.append(x.T[:,:,None])
        inputs = torch.from_numpy(np.stack(seqs)).to('cuda').contiguous()
        mse = (model(inputs)-inputs).square().mean(dim=(2,3)).cpu().numpy()
        errors.update(zip(ids,mse))
    rows = []
    for p,ws in windows.items():
        total = np.zeros(len(features),np.float32)
        for w in ws:
            total += errors[w]
        actual = np.max(total/np.float32(len(ws)))
        # Exact equality is expected with the original full batch/layout/runtime.
        assert actual==stored[p], (str(checkpoint),p,float(actual),float(stored[p]))
        rows.append(dict(checkpoint=str(checkpoint.relative_to(RUN)),sample=p,timestamp=str(pd.Timestamp(times[p])),
                         window_count=len(ws),saved_score=float(stored[p]),replayed_score=float(actual),exact_match=True))
    del model
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--skip-data-hashes',action='store_true',help='Diagnostics only; cannot issue verified receipt')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    plan=read(RUN/'frozen/protocol.json')
    for name,h in plan['source_sha256'].items():assert sha(ROOT/name)==h,name
    assert (ROOT/'scripts/research/esa-mission2-study.service').read_bytes()==Path('/home/k_faz/.config/systemd/user/esa-mission2-study.service').read_bytes()
    receipt=read(RUN/'complete.json');assert receipt['computation_complete']
    for name,h in receipt['files'].items():assert sha(RUN/name)==h,name
    preparation=read(RUN/'preparation.json')
    assert sha(RUN/'preparation.json')==plan['preparation_sha256']
    assert sha(RUN/'preflight-verification.json')==plan['preflight_sha256']
    if not args.skip_data_hashes:
        for name,h in preparation['raw'].items():assert sha(ROOT/name)==h,name
    import platform,wandb
    runtime=dict(python=platform.python_version(),torch=str(torch.__version__),numpy=np.__version__,pandas=pd.__version__,wandb=wandb.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),threads=torch.get_num_threads(),tf32=torch.backends.cuda.matmul.allow_tf32)
    assert runtime==plan['runtime'],(runtime,plan['runtime'])
    assert {k:str(v) for k,v in vars(config).items() if k.isupper()}==plan['settings']
    print('All completion/raw/source/runtime hashes match',flush=True)
    labels=pd.read_csv(ROOT/'data/ESA-Mission2/labels.csv').merge(pd.read_csv(ROOT/'data/ESA-Mission2/anomaly_types.csv'),on='ID',validate='many_to_one')
    local=labels.copy()
    local.loc[~local.Channel.isin(FEATURES[:8]),'Category']='Out-of-scope (neutral)'
    anns={'system_all_channels':labels,'selected_eight_channels':local}
    # Independently refit all shared-feature scalers once on the nominal training rows.
    training,tt,y=read_values('train',labels=True)
    med=np.empty(len(FEATURES),np.float32);iqr=med.copy()
    for i in range(len(FEATURES)):
        values=training[~y,i]
        med[i]=np.median(values)
        q75,q25=np.percentile(values,[75,25]);spread=q75-q25
        iqr[i]=1. if abs(spread)<1e-8 else spread
    c=np.r_[0,np.cumsum(y)]
    starts=np.arange(0,len(y)-255,16)
    nominal_windows=int((c[starts+256]==c[starts]).sum())
    assert (nominal_windows+63)//64==1535
    del training,tt,y,c,starts
    raw,ct,_=read_values('calibration')
    ce=EventEvaluator(ct,local)
    selections=read(RUN/'selection_complete.json')
    assert selections['selection_scope']==plan['selection_scope']
    freeze_time=(RUN/'selection_complete.json').stat().st_mtime
    assert len(selections['selections'])==9
    cal_rows=[];spot_rows=[];grid_count=0;candidate_rows=0
    for job,selection in zip(plan['jobs'],selections['selections']):
        f=RUN/job['name'];trained=read(f/'trained.json')
        assert job==trained['job']==selection['job']
        assert sha(f/'trained.json')==selection['trained_sha256']
        assert (f/'trained.json').stat().st_mtime<freeze_time<(f/'assessment_score.npy').stat().st_mtime
        for record_name in ('trained.json','assessed.json'):
            for name,h in read(f/record_name)['files'].items():
                assert sha(f/name)==h,(job['name'],name)
        hist=pd.read_csv(f/'history.csv').set_index('epoch')
        assert hist.index.tolist()==list(range(1,41)) and np.isfinite(hist.loss).all()
        terminal=torch.load(f/'last.pt',map_location='cpu',weights_only=False)
        assert terminal['epoch']==41 and terminal['batch']==0 and terminal['job']==job
        assert len(terminal['history'])==40
        for hrow in terminal['history']:
            equal(hrow['loss'],hist.loc[hrow['epoch'],'loss'])
            for name,h in hrow.get('candidate_files',{}).items():
                assert h==trained['files'][name]
        eps=[25,30,35,40] if job['method']=='pretrain_finetune' else list(range(5,41,5))
        assert sorted(p.name for p in f.glob('epoch_*_grid.csv'))==[f'epoch_{ep:02d}_grid.csv' for ep in eps]
        choices=[]
        for ep in eps:
            stem=f'epoch_{ep:02d}'
            score=np.load(f/(stem+'_calibration_score.npy'))
            assert score.dtype==np.float32 and len(score)==len(ct) and np.isfinite(score).all()
            grid,rule=replay_grid(score,ce,QUANTILES)
            pd.testing.assert_frame_equal(grid,pd.read_csv(f/(stem+'_grid.csv')),check_dtype=False,rtol=1e-11,atol=1e-9)
            equal(rule['val_esa_f05'],hist.loc[ep,'calibration_target_f05'])
            choices.append((ep,rule));grid_count+=1;candidate_rows+=len(grid)
            ck=torch.load(f/(stem+'.pt'),map_location='cpu',weights_only=False)
            assert ck['features']==job['features']
            spot_rows.extend(inference_spots(f/(stem+'.pt'),raw,ct,score,(med,iqr)))
        best=max(choices,key=lambda x:x[1]['val_esa_f05'])
        frozen=read(f/'frozen_rule.json')
        assert best[0]==frozen['epoch'] and frozen==selection['rule']==terminal['best']
        for k,v in best[1].items(): equal(v,frozen['rule'][k],k)
        assert sha(f/'model.pt')==sha(f/f'epoch_{best[0]:02d}.pt')==selection['model_sha256']
        assert sha(f/'calibration_score.npy')==sha(f/f'epoch_{best[0]:02d}_calibration_score.npy')
        assert sha(f/'calibration_candidates.csv')==sha(f/f'epoch_{best[0]:02d}_grid.csv')
        ix=[FEATURES.index(c) for c in job['features']]
        assert np.array_equal(terminal['median'],med[ix]) and np.array_equal(terminal['iqr'],iqr[ix])
        for ep,r in choices:cal_rows.append(dict(name=job['name'],method=job['method'],seed=job['seed'],arm=job['arm'],epoch=ep,selected=ep==best[0],**r))
        print(job['name'],'all grids, selection, scalers, calibration inference spots passed',flush=True)
        pd.DataFrame(cal_rows).to_csv(OUT/'verified_calibration.csv',index=False)
        pd.DataFrame(spot_rows).to_csv(OUT/'inference_spots.csv',index=False)
    del raw
    raw,times,_=read_values('assessment')
    saved=pd.read_csv(RUN/'all_results.csv',dtype={'period':str}).set_index(['name','period','event_definition','scope'])
    assert len(saved)==144 and saved.index.is_unique
    periods={str(year):(int(np.searchsorted(times,pd.Timestamp(f'{year}-01-01').value)),int(np.searchsorted(times,pd.Timestamp(f'{year+1}-01-01').value))) for year in range(2001,2004)}
    periods['pooled']=(0,len(times))
    evaluators={(period,definition,scope):EventEvaluator(times[a:b],ann,categories) for period,(a,b) in periods.items() for definition,categories in [('anomaly_and_rare',('Anomaly','Rare Event')),('anomaly_only',('Anomaly',))] for scope,ann in anns.items()}
    rows=[];events=[]
    for job in plan['jobs']:
        f=RUN/job['name'];r=read(f/'frozen_rule.json')['rule']
        score=np.load(f/'assessment_score.npy');stored=np.load(f/'assessment_prediction.npy')
        assert len(score)==len(times) and score.dtype==np.float32 and np.isfinite(score).all()
        pred=process(score>r['threshold'],int(r['merge_gap']),int(r['min_dur']))
        assert np.array_equal(pred,stored)
        spot_rows.extend(inference_spots(f/'model.pt',raw,times,score,(med,iqr),annual=True))
        by_job=pd.read_csv(f/'metrics.csv',dtype={'period':str}).set_index(['period','event_definition','scope'])
        saved_ids=read(f/'event_ids.json');assert len(saved_ids)==16
        for (period,definition,scope),ev in evaluators.items():
            a,b=periods[period];metrics=ev.score(pred[a:b])
            for k,v in metrics.items():
                equal(v,saved.loc[(job['name'],period,definition,scope),k],(job['name'],period,scope,k))
                equal(v,by_job.loc[(period,definition,scope),k])
            ids=event_ids(ev,pred[a:b]);assert ids==saved_ids[period+'/'+definition+'/'+scope]
            assert len(ids['detected'])==metrics['TPe'] and len(ids['missed'])==metrics['FNe']
            rows.append(dict(**{k:job[k] for k in ('name','method','seed','arm')},period=period,event_definition=definition,scope=scope,**metrics))
            for status,idlist in ids.items():
                events.extend(dict(name=job['name'],method=job['method'],seed=job['seed'],arm=job['arm'],period=period,event_definition=definition,scope=scope,event_id=aid,detected=status=='detected') for aid in idlist)
        print(job['name'],'assessment inference, exact prediction, 16 metric rows and IDs passed',flush=True)
    pd.DataFrame(rows).to_csv(OUT/'verified_results.csv',index=False)
    pd.DataFrame(events).to_csv(OUT/'verified_event_ids.csv',index=False)
    pd.DataFrame(spot_rows).to_csv(OUT/'inference_spots.csv',index=False)
    save('verification.json',dict(verified=not args.skip_data_hashes,verified_at_utc=datetime.now(timezone.utc).isoformat(),
         complete_receipt_sha256=sha(RUN/'complete.json'),verifier_sha256=sha(Path(__file__)),completion_files=len(receipt['files']),
         jobs=9,epochs_per_job=40,nominal_training_windows=nominal_windows,batches_per_epoch=1535,
         calibration_grids=grid_count,calibration_candidate_rows=candidate_rows,metric_rows=len(rows),event_id_rows=len(events),
         inference_checkpoints=69,inference_spots=len(spot_rows),inference_exact_matches=len(spot_rows),
         joint_freeze_utc=datetime.fromtimestamp(freeze_time,timezone.utc).isoformat(),
         checks=['Frozen plan/runtime/source/dependencies/annotations, installed unit and prepared data hashes',
                 'All completion, training and assessment receipts; 9 histories and final states',
                 'Independent nominal-training scaler refit, all candidate and selected scalers',
                 'Independent postprocessing and complete calibration-grid replay; within-run selection and earliest checkpoint ties',
                 'All 9 selections jointly frozen before any assessment score; selected artifact equality',
                 'Original-batch GPU inference at fixed calibration/annual assessment timestamp spots; exact float32 scores',
                 'Full continuous prediction replay and all annual/pooled corrected metrics and detected/missed IDs'],
         limitations=['Inference spot checks, not a second full neural inference pass',
                      'Corrected event evaluator reused after bundled-reference parity tests; postprocessing, window enumeration and ID enumeration independently implemented',
                      'Three seeds and reused benchmark; input expansion also expands alarm score sources',
                      'Scoped destination verification does not certify unrelated historical migration exceptions']))
    print('VERIFIED',flush=True)


if __name__=='__main__':
    main()
