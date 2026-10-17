"""Frozen six-model channel-error diagnostic; no fitting or decision changes."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from esa_thesis import config
from esa_thesis.calibration_audit import digest, verify_recorded, load_annotations
from esa_thesis.data import RobustChannelScaler
from esa_thesis.development import load_period
from esa_thesis.diagnose_channels import FEATURES, TRAIN_SHA, window_errors, aggregate
from esa_thesis.evaluation.esa import EventEvaluator, CATEGORIES
from esa_thesis.models import MultivariateAE
from esa_thesis.thresholds import postprocess

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'results_longrun/development/inner_selection_cuda_v2'
OUT=ROOT/'results_longrun/diagnostics/channel_error_replay_20261017_v3'

def interval_mask(times, intervals):
    keep=np.zeros(len(times),bool)
    for lo,hi in intervals:
        keep[np.searchsorted(times,lo,'left'):np.searchsorted(times,hi,'right')]=True
    return keep

def tail_fraction(sorted_values, value):
    return float((len(sorted_values)-np.searchsorted(sorted_values,value,'left'))/len(sorted_values)) if len(sorted_values) else None

def summarize(values,times,annotations,fold,seed,period):
    ev=EventEvaluator(times,annotations)
    nominal=~interval_mask(times,ev.all_intervals)
    month=pd.to_datetime(times,utc=True).strftime('%Y-%m').to_numpy()
    monthly=[];events=[]
    winner=values.argmax(axis=0)
    scalar=values.max(axis=0)
    normal_scalar=np.sort(scalar[nominal])
    for j,ch in enumerate(FEATURES):
        normal=np.sort(values[j,nominal])
        for m in np.unique(month):
            mask=nominal&(month==m)
            x=values[j,mask]
            if len(x):
                monthly.append(dict(fold=fold,seed=seed,period=period,month=m,channel=ch,samples=len(x),median=float(np.median(x)),q01=float(np.percentile(x,1)),q99=float(np.percentile(x,99)),max=float(x.max()),maximum_contributor_fraction=float(np.mean(winner[mask]==j))))
        if period=='training': continue
        lab=annotations[(annotations.Channel==ch)&annotations.Category.isin(CATEGORIES)]
        for aid,g in lab.groupby('ID'):
            intervals=[(a.value,b.value) for a,b in zip(g.StartTime,g.EndTime) if a.value<=times[-1] and b.value>=times[0]]
            if not intervals: continue
            mask=interval_mask(times,intervals)
            if not mask.any():
                events.append(dict(fold=fold,seed=seed,period=period,event_id=aid,channel=ch,samples=0))
                continue
            x=values[j,mask]; mx=float(x.max()); mn=float(x.min()); med=float(np.median(x)); base=float(np.median(normal))
            counts=np.bincount(winner[mask],minlength=8)
            events.append(dict(fold=fold,seed=seed,period=period,event_id=aid,channel=ch,samples=int(mask.sum()),peak_error=mx,min_error=mn,median_error=med,nominal_median_error=base,
                median_change=med-base,nominal_fraction_ge_peak=tail_fraction(normal,mx),nominal_fraction_le_min=float(np.searchsorted(normal,mn,'right')/len(normal)),
                scalar_nominal_fraction_ge_peak=tail_fraction(normal_scalar,float(scalar[mask].max())),
                own_channel_wins_fraction=float(np.mean(winner[mask]==j)),dominant_channel=FEATURES[int(counts.argmax())]))
    return events,monthly

def main():
    torch.set_num_threads(4)
    if not torch.cuda.is_available(): raise ValueError('CUDA unavailable')
    print('Checking source, annotation and training-data integrity',flush=True)
    original=json.loads((SOURCE/'protocol.json').read_text())
    for name in ['models.py','config.py','data.py']:
        assert digest(ROOT/'esa_thesis'/name)==original['source_sha256'][name],name
    assert digest(config.TRAIN_FILE)==TRAIN_SHA
    paths=[ROOT/'data/ESA-Mission1/labels.csv',ROOT/'data/ESA-Mission1/anomaly_types.csv']
    annotation_hashes=verify_recorded(ROOT,paths)
    annotations=load_annotations(*paths)
    protocol=dict(source=str(SOURCE),source_protocol_sha256=digest(SOURCE/'protocol.json'),script_sha256=digest(Path(__file__)),annotation_hashes=annotation_hashes,train_sha256=TRAIN_SHA,features=FEATURES,
        folds=[2001,2002],seeds=[42,43,44],policy='calibration_selected',score_rtol=1e-5,score_atol=1e-6,batch_size=64,
        notes='Nominal distributions exclude all annotation intervals. Error minima/median changes are descriptive, not new anomaly rules. No outer test use.',
        numerical_replay='GPU first; if score tolerance or exact frozen mask differs, repeat that stage on CPU and require original score tolerance and exact mask. No threshold edits. Failed v1/v2 preserved; verified v2 stages reused by hash.')
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists(): assert json.loads((OUT/'protocol.json').read_text())==protocol
    else: (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    all_events=[];all_monthly=[];replays=[]
    for fold in [2001,2002]:
        folder=SOURCE/str(fold)
        # One bounded load per fold, then preserve original segment boundaries.
        print(f'{fold}: loading only 2000 through {fold}',flush=True)
        raw,_,all_times=load_period(config.TRAIN_FILE,FEATURES,None,f'{fold+1}-01-01')
        for seed in [42,43,44]:
            source=folder/f'seed{seed}'
            manifest=json.loads((source/'complete.json').read_text())
            for name,sha in manifest.items(): assert digest(source/name)==sha,name
            decision=json.loads((source/'frozen_decisions.json').read_text())['calibration_selected']
            checkpoint=source/f"epoch_{decision['epoch']}"/'model.pt'
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            assert saved['features']==FEATURES
            scaler=RobustChannelScaler();scaler.median_=saved['scaler_median'];scaler.iqr_=saved['scaler_iqr']
            model=MultivariateAE(8).to('cuda');model.load_state_dict(saved['model_state_dict']);model.eval()
            for period,start,end in [('calibration',f'{fold}-01-01',f'{fold}-07-01'),('validation',f'{fold}-07-01',f'{fold+1}-01-01'),('training','2000-01-01',f'{fold}-01-01')]:
                stage=OUT/f'{fold}_{seed}_{period}';stage.mkdir(exist_ok=True)
                if (stage/'complete.json').exists():
                    state=json.loads((stage/'complete.json').read_text())
                    for name,sha in state['files'].items(): assert digest(stage/name)==sha
                    all_events.extend(json.loads((stage/'events.json').read_text()));all_monthly.extend(json.loads((stage/'monthly.json').read_text()));replays.append(state['replay']);continue
                take=(all_times>=pd.Timestamp(start).value)&(all_times<pd.Timestamp(end).value)
                times=all_times[take];x=scaler.transform(raw[take])
                starts=np.arange(0,len(x)-config.SEQ_LEN+1,config.SCORE_STRIDE)
                errors=window_errors(model,x,starts,'cuda',64,f'{fold} seed{seed} {period}')
                values=aggregate(errors,starts,len(x)); scalar=values.max(axis=0)
                replay=dict(fold=fold,seed=seed,period=period,checkpoint_sha256=digest(checkpoint))
                if period!='training':
                    reference=source/f"epoch_{decision['epoch']}"/'calibration_score.npy' if period=='calibration' else source/'calibration_selected_validation_score.npy'
                    expected=np.load(reference,allow_pickle=False)
                    np.testing.assert_array_equal(times,np.load(folder/'calibration_timestamps.npy' if period=='calibration' else source/'validation_timestamps.npy'))
                    rule=decision['rule'];predict=lambda s:postprocess(s>rule['threshold'],int(rule['merge_gap']),int(rule['min_dur']))
                    replay['device']='cuda'
                    if not np.allclose(scalar,expected,rtol=1e-5,atol=1e-6) or not np.array_equal(predict(scalar),predict(expected)):
                        print('Numerical replay discrepancy: checking CPU with the same frozen rule',flush=True)
                        model.to('cpu')
                        errors=window_errors(model,x,starts,'cpu',64,f'{fold} seed{seed} {period} CPU replay')
                        values=aggregate(errors,starts,len(x));scalar=values.max(axis=0)
                        model.to('cuda')
                        np.testing.assert_allclose(scalar,expected,rtol=1e-5,atol=1e-6)
                        replay['device']='cpu'
                    np.testing.assert_allclose(scalar,expected,rtol=1e-5,atol=1e-6)
                    np.testing.assert_array_equal(predict(scalar),predict(expected))
                    if period=='validation': np.testing.assert_array_equal(predict(scalar),np.load(source/'calibration_selected_validation_prediction.npy'))
                    replay.update(max_abs_error=float(np.max(np.abs(scalar-expected))),mask_exact=True)
                events,monthly=summarize(values,times,annotations,fold,seed,period)
                np.save(stage/'channel_scores.npy',values)
                (stage/'events.json').write_text(json.dumps(events,indent=2)+'\n');(stage/'monthly.json').write_text(json.dumps(monthly,indent=2)+'\n')
                (stage/'complete.json').write_text(json.dumps(dict(replay=replay,files={p.name:digest(p) for p in stage.iterdir() if p.is_file()}),indent=2)+'\n')
                all_events.extend(events);all_monthly.extend(monthly);replays.append(replay)
                print(f'Verified {fold} seed{seed} {period}',flush=True)
                del values,x,errors
            del model
        del raw
    assert len(replays)==18 and sum(r.get('mask_exact',False) for r in replays)==12
    assert digest(config.TRAIN_FILE)==TRAIN_SHA
    assert verify_recorded(ROOT,paths)==annotation_hashes
    pd.DataFrame(all_events).to_csv(OUT/'events.csv',index=False)
    pd.DataFrame(all_monthly).to_csv(OUT/'monthly.csv',index=False)
    (OUT/'complete.json').write_text(json.dumps(dict(verified=True,replays=replays,artifacts={p.name:digest(p) for p in [OUT/'events.csv',OUT/'monthly.csv',OUT/'protocol.json']}),indent=2)+'\n')
    print('COMPLETE: six frozen models, 12 exact calibration/validation masks, 18 stages',flush=True)

if __name__=='__main__': main()
