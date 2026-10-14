"""User-requested full-period short-no-roll assessment, without test selection."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from . import config
from .calibration_audit import digest,verify_recorded
from .development import development_spec,run_fold,QUANTILES
from .development_recovery import check_protocol,prepare_fold
from .long_baseline import BOUNDARIES
from .masked_pilot import assessment_scopes
from .thresholds import postprocess

MANIFEST_SHA='a49a8c42c23a5d72d91a48596f485d762a805cc6101ad037c01a36c8037bf099'

def verify_frozen(root):
    base=root/'data/frozen/mission1-reference-v1'
    if digest(base/'manifest.json')!=MANIFEST_SHA:raise ValueError('Frozen manifest changed')
    manifest=json.loads((base/'manifest.json').read_text())
    if set(manifest['files'])!={'train.csv','calibration.csv','testing.csv','labels.csv.gz'}:raise ValueError('Unexpected inventory')
    for n,r in manifest['files'].items():
        p=base/n
        if p.is_symlink() or p.stat().st_size!=r['bytes'] or digest(p)!=r['sha256']:raise ValueError('Frozen data changed: '+n)
    return manifest

def report_predictions(times,prediction):
    # Annual slices retain continuous predictions; postprocessing is never restarted at January.
    pooled=[dict(period='2007-2013',scope=s,**ev.score(prediction)) for s,ev in assessment_scopes(times).items()]
    annual=[]
    for year in range(2007,2014):
        keep=(times>=pd.Timestamp(f'{year}-01-01').value)&(times<pd.Timestamp(f'{year+1}-01-01').value)
        for s,ev in assessment_scopes(times[keep]).items():annual.append(dict(period=str(year),scope=s,**ev.score(prediction[keep])))
    return pd.DataFrame(annual),pd.DataFrame(pooled)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    root=Path(__file__).resolve().parents[1];base=root/'data/frozen/mission1-reference-v1'
    if not torch.cuda.is_available():raise ValueError('CUDA unavailable')
    probe=torch.ones((256,256),device='cuda');assert torch.isfinite(probe@probe).all();torch.cuda.synchronize();del probe
    print('Verifying frozen splits and original annotations...',flush=True)
    manifest=verify_frozen(root);annotations=verify_recorded(root,[config.ANNOTATIONS_FILE,config.ANOMALY_TYPES_FILE])
    package=Path(__file__).parent
    plan=dict(experiment='full-short-no-roll-20261014-v1',train=['2000-01-01','2006-01-01'],calibration=['2006-01-01','2007-01-01'],assessment=['2007-01-01','2014-01-01'],features=[f'channel_{i}' for i in range(40,48)],epochs=40,batch_size=64,seed=42,device='cuda',max_windows=0,augmentation='short_no_roll',manifest_sha256=MANIFEST_SHA,annotations=annotations,
        selection='2006 corrected event F0.5 every five epochs and final epoch; earliest tie; original 1% total-alarm cap',quantiles=QUANTILES,merge_gaps=[0,16,64],min_durs=[1,8,32],
        limitations='Explicit user-requested extension despite unstable development results. Previously inspected benchmark, not independent confirmation. Single seed; historical legacy baseline is context, not a matched short-roll-only ablation.',
        settings={k:str(v) for k,v in vars(config).items() if k.isupper()},source_sha256={str(f.relative_to(package)):digest(f) for f in package.rglob('*.py')})
    check_protocol(a.output,plan,a.resume);folder=a.output/'full_period';result=prepare_fold(folder,a.resume,False)
    if result is None:
        result=run_fold(development_spec('baseline'),BOUNDARIES,folder,epochs=40,device='cuda',batch_size=64,seed=42,max_windows=0,training_path=base/'train.csv',calibration_path=base/'calibration.csv',assessment_path=base/'testing.csv',augmentation_mode='short_no_roll')
    rule=json.loads((folder/'frozen_rule.json').read_text())['rule'];scores=np.load(folder/'assessment_score.npy',allow_pickle=False);pred=np.load(folder/'assessment_prediction.npy',allow_pickle=False)
    if not np.array_equal(pred,postprocess(scores>rule['threshold'],int(rule['merge_gap']),int(rule['min_dur']))):raise ValueError('Prediction replay differs')
    times=pd.date_range('2007-01-01','2014-01-01',inclusive='left',freq='30s').asi8
    if len(times)!=len(pred):raise ValueError('Incomplete assessment')
    annual,pooled=report_predictions(times,pred)
    system=pooled[pooled.scope=='system_all_channels'].iloc[0]
    for k in ('EW_F_0.50','EW_precision','EW_recall','TPe','FPe','FNe'):
        if not np.isclose(system[k],result[k],rtol=1e-11,atol=1e-9):raise ValueError('Pooled replay mismatch')
    verify_frozen(root)
    if verify_recorded(root,[config.ANNOTATIONS_FILE,config.ANOMALY_TYPES_FILE])!=annotations:raise ValueError('Annotations changed')
    annual.to_csv(a.output/'yearly_results.csv',index=False);pooled.to_csv(a.output/'pooled_results.csv',index=False)
    combined=pd.concat([annual,pooled],ignore_index=True);combined.to_csv(a.output/'all_results.csv',index=False)
    lines=['# Full-period short-roll removal results','', 'Train 2000–2005; calibrate 2006; inputs 40–47; seed 42. One frozen model/rule across 2007–2013. Previously inspected benchmark, not unseen validation.','', '| Period | All-channel event F0.5 | Input-channel event F0.5 |','|---|---:|---:|']
    pivot=combined.pivot(index='period',columns='scope',values='EW_F_0.50')
    for period in [str(y) for y in range(2007,2014)]+['2007-2013']:
        row=pivot.loc[period];lines.append(f'| {period} | {row["system_all_channels"]:.6f} | {row["input_channels_40_47"]:.6f} |')
    lines+=['','The pooled row is rescored over the continuous seven-year predictions, not averaged from yearly F0.5. No yearly tuning. All values use the corrected ESA event evaluator.']
    (a.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    artifacts={str(f.relative_to(a.output)):digest(f) for f in a.output.rglob('*') if f.is_file() and f.name!='complete.json'}
    (a.output/'complete.json').write_text(json.dumps(dict(verified=True,artifacts=artifacts,selected_epoch=result['epoch'],calibration_f05=result['calibration_f05']),indent=2)+'\n')
    print('\n'.join(lines),flush=True)
if __name__=='__main__':main()
