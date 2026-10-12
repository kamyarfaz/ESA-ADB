"""Parameter-matched diagonal channel-attention comparison; frozen joint control."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

from . import config, inner_validation
from .diagnose_channels import digest
from .channel_isolation import ChannelIsolatedAE
from .loss_weight_comparison import verify_control


@contextmanager
def isolated_model():
    previous = inner_validation.MultivariateAE
    try:
        inner_validation.MultivariateAE = ChannelIsolatedAE
        yield
    finally:
        inner_validation.MultivariateAE = previous


def compare(control, candidate):
    old = json.loads((control/'protocol.json').read_text())
    new = json.loads((candidate/'protocol.json').read_text())
    settings = dict(new['settings'])
    if settings != old['settings']: raise ValueError('Unexpected training setting change')
    for key in ['folds','seeds','features','epochs','max_train_windows','batch_size','augmentation',
                'device','threads','policies','quantiles','merge_gaps','min_durs','alarm_cap',
                'annotations','train_sha256','runtime']:
        if old[key] != new[key]: raise ValueError(f'Protocol mismatch: {key}')
    if new['settings']['PA_LAMBDA'] != 1.: raise ValueError('Wrong loss weight')
    rows = []
    for fold in old['folds']:
        for seed in old['seeds']:
            left, right = control/fold/f'seed{seed}', candidate/fold/f'seed{seed}'
            for name in ['training_starts.npy','validation_timestamps.npy']:
                np.testing.assert_array_equal(np.load(left/name),np.load(right/name))
            for key in ['scaler_median','scaler_iqr','training_samples','calibration_samples','nominal_windows']:
                if json.loads((left/'preflight.json').read_text())[key] != json.loads((right/'preflight.json').read_text())[key]:
                    raise ValueError(f'Preprocessing mismatch: {key}')
            for arm,path in [('joint',left),('isolated',right)]:
                frame=pd.read_csv(path/'results.csv')
                rows.extend(frame[frame.policy=='calibration_selected'].assign(arm=arm).to_dict('records'))
    frame=pd.DataFrame(rows);frame['fold']=frame['fold'].astype(str)
    frame.to_csv(candidate/'isolation_results.csv',index=False)
    summary=frame.groupby(['fold','arm','scope']).agg(f05_mean=('EW_F_0.50','mean'),f05_sd=('EW_F_0.50','std'),fp_mean=('FPe','mean'),recall_mean=('EW_recall','mean'),nominal_false_alarm_rate_mean=('nominal_false_alarm_rate','mean')).reset_index()
    summary.to_csv(candidate/'isolation_summary.csv',index=False)
    system=summary[summary.scope=='system_all_channels'].set_index(['fold','arm'])
    passed=all(system.loc[(f,'isolated'),'f05_mean']>system.loc[(f,'joint'),'f05_mean'] and system.loc[(f,'isolated'),'fp_mean']<=system.loc[(f,'joint'),'fp_mean'] for f in old['folds'])
    inner_validation.write_json(candidate/'isolation_decision.json',dict(advancement_gate_passed=bool(passed),promoted=False,primary_policy='calibration_selected'))
    print(summary.to_string(index=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    old,hashes=verify_control(a.control)
    plan=dict(experiment='inner-channel-isolation-v1',control=str(a.control.resolve()),candidate_architecture='ChannelIsolatedAE', loss_weight=1.0,
              primary_policy='calibration_selected',change='Diagonal masks on channel encoder, channel decoder self-attention and cross-attention; same parameters and initial weights',
              control_hashes=hashes,seeds=old['seeds'],folds=old['folds'],
              advancement='Higher mean system F0.5 in both folds with no mean false-event increase; ties retain control',
              limitations='Previously inspected internal development periods; no new outer assessment; not robust target achievement',
              secondary='fixed_epoch5 outputs retained by unchanged runner; not used to select the primary result')
    print(json.dumps(plan,indent=2),flush=True)
    if a.dry_run:return
    sidecar=a.output.with_name(a.output.name+'.comparison.json')
    if a.resume:
        if json.loads(sidecar.read_text())!=plan:raise ValueError('Comparison protocol changed')
    else:
        if a.output.exists() or sidecar.exists():raise ValueError('Use a new output')
        sidecar.parent.mkdir(parents=True,exist_ok=True);inner_validation.write_json(sidecar,plan)
    previous=sys.argv
    try:
        sys.argv=['inner-validation','--output',str(a.output),'--device',old['device'],'--threads',str(old['threads'])]+(['--resume'] if a.resume else [])
        with isolated_model():inner_validation.main()
    finally:
        sys.argv=previous
    if hashes!={f:digest(f) for f in hashes}:raise ValueError('Control manifests changed')
    compare(a.control,a.output)
    inner_validation.write_json(a.output/'comparison_protocol.json',plan)


if __name__=='__main__':main()
