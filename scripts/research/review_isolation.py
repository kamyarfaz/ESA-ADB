"""Matched primary-policy report after both campaign reviews have passed."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from esa_thesis.diagnose_channels import digest
from esa_thesis.loss_weight_comparison import verify_control
from esa_thesis.isolation_comparison import compare


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--candidate-review',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Use a new review output')
    plan,hashes=verify_control(a.control)
    review=json.loads((a.candidate_review/'verification.json').read_text())
    assert review['verified_metric_rows']==24 and review['verified_epochs']==30
    assert review['source_protocol_sha256']==digest(a.candidate/'protocol.json')
    candidate_plan=json.loads((a.candidate/'protocol.json').read_text())
    assert all(digest(Path('esa_thesis')/name)==sha for name,sha in candidate_plan['source_sha256'].items())
    histories=[]
    for fold in plan['folds']:
        np.testing.assert_array_equal(np.load(a.control/fold/'calibration_timestamps.npy'),
                                      np.load(a.candidate/fold/'calibration_timestamps.npy'))
    for root,weight in [(a.control,1.),(a.candidate,1.)]:
        for fold in plan['folds']:
            for seed in plan['seeds']:
                folder=root/fold/f'seed{seed}'
                manifest=json.loads((folder/'complete.json').read_text())
                assert all(digest(folder/name)==sha for name,sha in manifest.items())
                h=pd.read_csv(folder/'history.csv')
                np.testing.assert_allclose(h.loss,h.normal_mse+weight*h.pseudo_penalty,rtol=0,atol=2e-6)
                histories.append(h.assign(fold=fold,seed=seed,weight=weight,arm='joint' if root==a.control else 'isolated'))
    compare(a.control,a.candidate)
    a.output.mkdir(parents=True)
    frame=pd.read_csv(a.candidate/'isolation_results.csv')
    summary=pd.read_csv(a.candidate/'isolation_summary.csv')
    frame.to_csv(a.output/'results.csv',index=False);summary.to_csv(a.output/'summary.csv',index=False)
    pd.concat(histories,ignore_index=True).to_csv(a.output/'training_history.csv',index=False)
    decision=json.loads((a.candidate/'isolation_decision.json').read_text())
    (a.output/'verification.json').write_text(json.dumps(dict(status='complete',decision=decision,
        candidate_review=review,control_protocol_sha256=digest(a.control/'protocol.json'),
        verified_loss_decompositions=60,matched_windows_scalers_timestamps=True,
        interpretation='Primary calibration-selected checkpoints only; fixed-epoch5 not used for model choice'),indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    system=frame[frame.scope=='system_all_channels']
    fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    for col,fold in enumerate((2001,2002)):
        for i,arm in enumerate(('joint','isolated')):
            d=system[(system.fold==fold)&(system.arm==arm)].sort_values('seed')
            axes[0,col].plot(d.seed,d['EW_F_0.50'],marker='o',label=arm)
            axes[1,col].bar(np.arange(3)+(i-.5)*.35,d.FPe,width=.35,label=arm)
        axes[0,col].set_ylim(-.02,1);axes[0,col].set_xticks([42,43,44]);axes[0,col].set_ylabel('System event F0.5')
        axes[0,col].set_title(f'Real July–December {fold}');axes[0,col].legend()
        axes[1,col].set_xticks(np.arange(3),['42','43','44']);axes[1,col].set_xlabel('Seed');axes[1,col].set_ylabel('False alarm events');axes[1,col].legend()
    fig.savefig(a.output/'comparison.png',dpi=150);plt.close(fig)
    print(summary.to_string(index=False));print(decision)


if __name__=='__main__':main()
