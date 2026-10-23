"""Calibration-only alarm refinement of two frozen models; never retrain."""
import argparse
import fcntl
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from . import config
from .calibration_audit import digest
from .data import AllWindows
from .diagnose_channels import aggregate
from .full_combinations import load_model, load_split
from .full_short_no_roll import verify_frozen, report_predictions
from .full_similarity_mosad import write_json, save_npy, verify_receipt
from .masked_pilot import assessment_scopes
from .thresholds import postprocess

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'results_longrun/combinations_full_20261021_seed42_v1'
METHODS=('pretrain_finetune','expanded_short_no_roll')
BATCH=64

def read(path):
    return json.loads(path.read_text())

def inference_inputs(x,starts,device):
    # Match DataLoader's torch.stack layout, including floating reduction order.
    return torch.stack([torch.from_numpy(x[s:s+config.SEQ_LEN].T[:,:,None]) for s in starts]).to(device)

def detected_ids(ev,pred):
    starts,ends,closed=ev.prediction_intervals(pred)
    return sorted(aid for aid,intervals in ev.by_id.items()
                  if any(ev._overlaps(starts,ends,closed,lo,hi).any() for lo,hi in intervals))

def prepare(output):
    print('Checking destination frozen data, original sources and model receipts',flush=True)
    verify_frozen(ROOT)
    prior=read(SOURCE/'protocol.json')
    for n,h in prior['source_sha256'].items():
        if digest(ROOT/'esa_thesis'/n)!=h:raise ValueError('Original source changed: '+n)
    for n,h in prior['annotations'].items():
        if digest(ROOT/n)!=h:raise ValueError('Annotations changed: '+n)
    receipt=read(SOURCE/'complete.json')
    for n,h in receipt['files'].items():
        if digest(SOURCE/n)!=h:raise ValueError('Original artifact changed: '+n)
    runtime=dict(torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,cuda=torch.version.cuda)
    if runtime!=prior['runtime']:raise ValueError('Runtime changed')
    plan=dict(experiment='false-alarm-reduction-20261023-v1',methods=METHODS,
              train='2000–2005; existing selected weights unchanged',calibration='2006',
              assessment='2007–2013; reused benchmark',batch=BATCH,
              source_receipt=digest(SOURCE/'complete.json'),runtime=runtime,
              source_sha256={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'esa_thesis').rglob('*.py')},
              research_protocol_sha256=digest(ROOT/'docs/research/false-alarm-reduction-protocol-2026-10-23.md'))
    output.mkdir(parents=True,exist_ok=True)
    if (output/'protocol.json').exists():
        if read(output/'protocol.json')!=json.loads(json.dumps(plan)):raise ValueError('Refinement protocol changed')
    else:write_json(output/'protocol.json',plan)

@torch.no_grad()
def channel_scores(method,folder,split):
    filename=split+'_per_channel.npy'
    if verify_receipt(folder,split+'_receipt.json'):
        return np.load(folder/filename,allow_pickle=False)
    model,scaler,features,masked=load_model(SOURCE/method/'model.pt','cuda')
    if masked:raise ValueError('Both selected models must be unmasked at inference')
    raw,y,times=load_split(split,features)
    x=scaler.transform(raw);del raw,y,times
    starts=AllWindows(x).starts
    errors=np.empty((len(starts),len(features)),np.float32)
    partial=folder/(split+'_window_errors.npy')
    offset=0
    if partial.exists():
        done=np.load(partial,allow_pickle=False)
        if done.ndim!=2 or done.shape[1]!=len(features) or len(done)>len(starts) or not np.isfinite(done).all():raise ValueError('Invalid inference checkpoint')
        offset=len(done);errors[:offset]=done;del done
    model.eval()
    for start in range(offset,len(starts),BATCH):
        xb=inference_inputs(x,starts[start:start+BATCH],'cuda')
        values=(model(xb)-xb).square().mean(dim=(2,3)).cpu().numpy()
        stop=start+len(values);errors[start:stop]=values
        if not np.isfinite(values).all():raise ValueError('Nonfinite model scores')
        if start==offset or stop==len(starts) or (start//BATCH+1)%100==0:
            save_npy(partial,errors[:stop])
            print(f'{method} {split} inference {stop}/{len(starts)} windows',flush=True)
    per=aggregate(errors,starts,len(x))
    save_npy(folder/filename,per)
    write_json(folder/(split+'_receipt.json'),dict(files={filename:digest(folder/filename),partial.name:digest(partial)}))
    del model,x,errors;gc.collect();torch.cuda.empty_cache()
    return per

def calibration_context():
    # Labels are only from 2006. No assessment labels enter selection.
    path=ROOT/'data/frozen/mission1-reference-v1/calibration.csv'
    cols=[c for c in pd.read_csv(path,nrows=0).columns if c.startswith('is_anomaly_')]
    nominal=np.concatenate([~(f.to_numpy()>0).any(axis=1) for f in pd.read_csv(path,usecols=cols,chunksize=25000)])
    times=pd.date_range('2006-01-01','2007-01-01',freq='30s',inclusive='left').asi8
    return nominal,times,assessment_scopes(times)

def diagnose(method,folder,per,nominal,times,scopes):
    rule=read(SOURCE/method/'frozen_rule.json')['rule']
    scores=per.max(axis=0);pred=postprocess(scores>rule['threshold'],int(rule['merge_gap']),int(rule['min_dur']))
    ev=scopes['system_all_channels'];starts,ends,closed=ev.prediction_intervals(pred)
    rows=[]
    for i,(start,end) in enumerate(zip(starts,ends)):
        annotated_ns=sum(max(0,min(end,hi)-max(start,lo)) for lo,hi in ev.all_intervals)
        touches=any(ev._overlaps(np.array([start]),np.array([end]),np.array([closed[i]]),lo,hi)[0] for lo,hi in ev.all_intervals)
        ix=(times>=start)&(times<end)
        if not ix.any():continue
        peak_idx=np.flatnonzero(ix)[np.argmax(scores[ix])]
        rows.append(dict(start=str(pd.Timestamp(start)),end=str(pd.Timestamp(end)),duration_seconds=(end-start)/1e9,
                         nominal_alarm_seconds=((end-start)-annotated_ns)/1e9,false_alarm_event=not touches,
                         peak_score=float(scores[peak_idx]),peak_channel=int(np.argmax(per[:,peak_idx]))))
    pd.DataFrame(rows).to_csv(folder/'calibration_alarm_segments.csv',index=False)
    features=read(SOURCE/'protocol.json')['expanded_features' if method=='expanded_short_no_roll' else 'features']
    dominance=np.argmax(per,axis=0)
    channels=[dict(channel=name,nominal_q99=float(np.percentile(per[i,nominal],99)),nominal_q999=float(np.percentile(per[i,nominal],99.9)),nominal_mean=float(per[i,nominal].mean()),nominal_dominant_alarm_samples=int(((dominance==i)&nominal&(pred>0)).sum())) for i,name in enumerate(features)]
    pd.DataFrame(channels).to_csv(folder/'calibration_channels.csv',index=False)
    months=pd.DatetimeIndex(times).month
    pd.DataFrame([dict(month=m,nominal_samples=int((nominal&(months==m)).sum()),nominal_alarm_samples=int((nominal&(months==m)&(pred>0)).sum())) for m in range(1,13)]).to_csv(folder/'calibration_months.csv',index=False)
    info=dict(method=method,metrics={s:ev.score(pred) for s,ev in scopes.items()},detected_ids={s:detected_ids(ev,pred) for s,ev in scopes.items()},features=features,
              normalization_q99=[c['nominal_q99'] for c in channels],rule=rule)
    for key,value in info['metrics']['system_all_channels'].items():
        recorded=rule['val_'+key]
        if isinstance(value,str):
            if value!=recorded:raise ValueError('Calibration metric version differs')
        elif not np.isclose(value,recorded,rtol=1e-11,atol=1e-9):raise ValueError('Original calibration replay differs: '+key)
    write_json(folder/'diagnosis.json',info)
    print(method+' calibration diagnosis '+json.dumps(info['metrics']['system_all_channels']),flush=True)

def transformed(per,family,scales):
    if family=='raw':return per.max(axis=0)
    if family=='channel_q99':return (per/np.maximum(np.asarray(scales,dtype=np.float32)[:,None],1e-8)).max(axis=0)
    raise ValueError('Unknown score transformation')

def select_candidates(folder,per,times,scopes):
    """Fixed small grid; preserve exact baseline calibration event IDs in both scopes."""
    diagnosis=read(folder/'diagnosis.json');base=diagnosis['rule']
    rows=[];candidates=[]
    # Include the unchanged rule as a fallback, not merely an optimized candidate.
    candidates.append(dict(family='raw',threshold=base['threshold'],merge_gap=int(base['merge_gap']),min_dur=int(base['min_dur']),original=True))
    for family in ('raw','channel_q99'):
        score=transformed(per,family,diagnosis['normalization_q99'])
        values=list(np.percentile(score,[99,99.5,99.75,99.9,99.95,99.975,99.99,99.995,100]))
        if family=='raw':values += [base['threshold']*v for v in (1,1.1,1.25,1.5,2)]
        for threshold in np.unique(values):
            for duration in (1,8,32):
                candidates.append(dict(family=family,threshold=float(threshold),merge_gap=0,min_dur=duration,original=False))
    for index,c in enumerate(candidates):
        score=transformed(per,c['family'],diagnosis['normalization_q99'])
        pred=postprocess(score>c['threshold'],c['merge_gap'],c['min_dur'])
        metrics={s:ev.score(pred) for s,ev in scopes.items()}
        ids={s:detected_ids(ev,pred) for s,ev in scopes.items()}
        preserves=all(set(diagnosis['detected_ids'][s]).issubset(ids[s]) for s in scopes)
        system=metrics['system_all_channels'];local=metrics['input_channels_40_47']
        feasible=preserves and pred.mean()<=.01 and system['FPe']<=diagnosis['metrics']['system_all_channels']['FPe'] and system['false_positive_seconds']<=diagnosis['metrics']['system_all_channels']['false_positive_seconds']+1e-8
        rows.append(dict(candidate=index,**c,preserves_ids=preserves,feasible=feasible,alarm_fraction=float(pred.mean()),system_f05=system['EW_F_0.50'],local_f05=local['EW_F_0.50'],false_alarm_seconds=system['false_positive_seconds'],false_alarm_events=system['FPe'],system_detected=system['TPe'],local_detected=local['TPe']))
    grid=pd.DataFrame(rows);grid.to_csv(folder/'calibration_candidates.csv',index=False)
    eligible=grid[grid.feasible].sort_values(['system_f05','false_alarm_seconds','original','candidate'],ascending=[False,True,False,True],kind='stable')
    if eligible.empty:raise ValueError('Baseline unexpectedly infeasible')
    winner=eligible.iloc[0].to_dict()
    write_json(folder/'frozen_rule.json',dict(candidate=winner,normalization_q99=diagnosis['normalization_q99'],baseline=base,scope='system F0.5; preserve exact calibration detected IDs in both scopes',calibration_candidates_sha256=digest(folder/'calibration_candidates.csv')))
    print(folder.name+' frozen rule '+json.dumps(winner),flush=True)

def assess(output):
    freeze=read(output/'selection_complete.json')
    for method,h in freeze['rules'].items():
        if digest(output/method/'frozen_rule.json')!=h:raise ValueError('Frozen decision changed')
    rows=[];events=[]
    times=pd.date_range('2007-01-01','2014-01-01',freq='30s',inclusive='left').asi8
    for method in METHODS:
        folder=output/method;rule=read(folder/'frozen_rule.json')
        per=channel_scores(method,folder,'testing')
        raw=per.max(axis=0)
        if not np.array_equal(raw,np.load(SOURCE/method/'testing_score.npy')):raise ValueError('Original model score replay differs')
        for variant in ('original','refined'):
            r=rule['baseline'] if variant=='original' else rule['candidate']
            score=raw if variant=='original' else transformed(per,r['family'],rule['normalization_q99'])
            pred=postprocess(score>r['threshold'],int(r['merge_gap']),int(r['min_dur']))
            if variant=='original' and not np.array_equal(pred,np.load(SOURCE/method/'testing_prediction.npy')):raise ValueError('Original predictions differ')
            save_npy(folder/(variant+'_prediction.npy'),pred)
            annual,pooled=report_predictions(times,pred)
            table=pd.concat([annual,pooled],ignore_index=True);table.insert(0,'variant',variant);table.insert(0,'method',method);rows.append(table)
            for scope,ev in assessment_scopes(times).items():
                ids=set(detected_ids(ev,pred))
                events.extend(dict(method=method,variant=variant,scope=scope,event_id=aid,detected=aid in ids) for aid in sorted(ev.by_id))
        del per,raw;gc.collect()
    pd.concat(rows,ignore_index=True).to_csv(output/'all_results.csv',index=False)
    pd.DataFrame(events).to_csv(output/'event_detection.csv',index=False)
    verify_frozen(ROOT)
    write_json(output/'complete.json',dict(needs_independent_verification=True,files={str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file() and p.name!='complete.json'}))
    print('Both refinements assessed. Independent verification required.',flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--phase',choices=['diagnose','finish'],required=True);args=parser.parse_args()
    lock=open(ROOT/'.false-alarm-reduction.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(2)
    if not torch.cuda.is_available():raise ValueError('CUDA unavailable')
    prepare(args.output)
    if args.phase=='diagnose':
        nominal,times,scopes=calibration_context()
        for method in METHODS:
            folder=args.output/method;folder.mkdir(exist_ok=True)
            per=channel_scores(method,folder,'calibration');diagnose(method,folder,per,nominal,times,scopes);del per
        write_json(args.output/'diagnosis_complete.json',dict(files={str(p.relative_to(args.output)):digest(p) for p in args.output.rglob('*') if p.is_file() and p.name not in ('diagnosis_complete.json','protocol.json')}))
        print('Calibration diagnoses complete. Review before authorizing finish phase.',flush=True)
    else:
        if not verify_receipt(args.output,'diagnosis_complete.json'):raise ValueError('Diagnosis receipt missing')
        if not (args.output/'research_review.json').exists():raise ValueError('Calibration research review missing')
        if not (args.output/'selection_complete.json').exists():
            _,times,scopes=calibration_context()
            for method in METHODS:
                folder=args.output/method
                if not verify_receipt(folder,'calibration_receipt.json'):raise ValueError('Calibration scores missing')
                per=np.load(folder/'calibration_per_channel.npy');select_candidates(folder,per,times,scopes);del per
            write_json(args.output/'selection_complete.json',dict(rules={m:digest(args.output/m/'frozen_rule.json') for m in METHODS},research_review_sha256=digest(args.output/'research_review.json')))
        assess(args.output)

if __name__=='__main__':main()
