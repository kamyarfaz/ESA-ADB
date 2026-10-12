"""Verify the completed internal-selection campaign and build comparison plots."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

from esa_thesis.diagnose_channels import digest
from esa_thesis.inner_validation import FOLDS, POLICIES, choose_decisions
from esa_thesis.masked_pilot import assessment_scopes
from esa_thesis.protocol import evaluator_for
from esa_thesis.thresholds import postprocess, select_rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists(): p.error('Preserve prior reviews: use a new output')
    plan = json.loads((args.source/'protocol.json').read_text())
    rows, quiet, calibration = [], [], []
    count = 0
    for fold in FOLDS:
        cal_times = np.load(args.source/fold/'calibration_timestamps.npy')
        cal_ev = evaluator_for(cal_times)
        months = pd.to_datetime(cal_times,utc=True).strftime('%Y-%m')
        nominal = np.ones(len(cal_times), bool)
        for lo, hi in cal_ev.all_intervals:
            nominal &= ~((cal_times >= lo)&(cal_times <= hi))
        for seed in plan['seeds']:
            folder = args.source/fold/f'seed{seed}'
            manifest = json.loads((folder/'complete.json').read_text())
            assert all(digest(folder/f) == h for f,h in manifest.items())
            last = torch.load(folder/'last.pt', map_location='cpu', weights_only=False)
            assert len(last['history']) == 5
            decisions = json.loads((folder/'frozen_decisions.json').read_text())
            assert decisions == choose_decisions(last['history'])
            for h in last['history']:
                epoch = h['epoch']; ep = folder/f'epoch_{epoch}'
                assert all(digest(ep/f) == sha for f,sha in h['sha256'].items())
                rule = select_rows(pd.read_csv(ep/'calibration_candidates.csv')).iloc[0]
                for key in ['threshold','merge_gap','min_dur','val_esa_f05']:
                    assert np.isclose(rule[key],h['rule'][key],rtol=0,atol=1e-12)
                scores = np.load(ep/'calibration_score.npy')
                pred = postprocess(scores>h['rule']['threshold'],int(rule.merge_gap),int(rule.min_dur))
                assert abs(cal_ev.score(pred)['EW_F_0.50']-h['rule']['val_esa_f05']) < 1e-9
                for month in np.unique(months):
                    values = scores[(months==month)&nominal]
                    calibration.append(dict(fold=fold,seed=seed,epoch=epoch,month=month,
                                            nominal_definition='Outside all raw annotation intervals',
                                            median=float(np.median(values)),q99=float(np.percentile(values,99)),
                                            maximum=float(values.max())))
            times = np.load(folder/'validation_timestamps.npy')
            assert times[0] == pd.Timestamp(FOLDS[fold][1],tz='UTC').value
            assert times[-1] == pd.Timestamp(FOLDS[fold][2],tz='UTC').value-30_000_000_000
            assert np.all(np.diff(times)==30_000_000_000)
            saved_rows = pd.read_csv(folder/'results.csv')
            for policy in POLICIES:
                rule = decisions[policy]['rule']; pred = np.load(folder/f'{policy}_validation_prediction.npy')
                score = np.load(folder/f'{policy}_validation_score.npy')
                np.testing.assert_array_equal(pred,postprocess(score>rule['threshold'],int(rule['merge_gap']),int(rule['min_dur'])))
                for scope,ev in assessment_scopes(times).items():
                    metrics = ev.score(pred)
                    row = saved_rows[(saved_rows.policy==policy)&(saved_rows.scope==scope)].iloc[0]
                    for key,value in metrics.items():
                        if isinstance(value,str): assert value == row[key]
                        else: assert np.isclose(value,row[key],rtol=0,atol=1e-9), (key,value,row[key])
                    rows.append(row.to_dict()); count += 1
                if fold == '2002':
                    keep = times >= pd.Timestamp('2002-10-01',tz='UTC').value
                    ev = evaluator_for(times[keep]); metrics = ev.score(pred[keep])
                    assert metrics['num_true_events'] == 0
                    quiet.append(dict(seed=seed,policy=policy,period='2002Q4',**metrics,
                                      interpretation='No selected true events: event F0.5 is not a detection-performance measure'))
    args.output.mkdir(parents=True)
    frame = pd.DataFrame(rows)
    frame['fold'] = frame['fold'].astype(str)
    frame.to_csv(args.output/'verified_results.csv',index=False)
    pd.DataFrame(quiet).to_csv(args.output/'quiet_quarter.csv',index=False)
    pd.DataFrame(calibration).to_csv(args.output/'calibration_monthly_scores.csv',index=False)
    summary = frame.groupby(['fold','policy','scope']).agg(f05_mean=('EW_F_0.50','mean'),f05_sd=('EW_F_0.50','std'),fp_mean=('FPe','mean'),recall_mean=('EW_recall','mean')).reset_index()
    summary.to_csv(args.output/'summary.csv',index=False)
    system = summary[summary.scope=='system_all_channels'].set_index(['fold','policy'])
    gate = all(system.loc[(f,'fixed_epoch5'),'f05_mean'] > system.loc[(f,'calibration_selected'),'f05_mean'] and system.loc[(f,'fixed_epoch5'),'fp_mean'] <= system.loc[(f,'calibration_selected'),'fp_mean'] for f in FOLDS)
    assert bool(gate) == json.loads((args.source/'decision.json').read_text())['advancement_gate_passed']
    (args.output/'verification.json').write_text(json.dumps(dict(verified_metric_rows=count,verified_epochs=30,advancement_gate_passed=bool(gate),source_protocol_sha256=digest(args.source/'protocol.json')),indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for ax,fold in zip(axes,FOLDS):
        data=frame[(frame.fold.astype(str)==fold)&(frame.scope=='system_all_channels')]
        for policy in POLICIES:
            values=data[data.policy==policy].sort_values('seed')
            ax.plot(values.seed,values['EW_F_0.50'],marker='o',label=policy)
        ax.set_title(f'Real Jul–Dec {fold}, inputs 40–47');ax.set_ylim(0,1);ax.set_xticks([42,43,44]);ax.set_xlabel('Seed');ax.set_ylabel('System event F0.5');ax.legend(fontsize=8)
    fig.savefig(args.output/'policy_comparison.png',dpi=150);plt.close(fig)
    print(summary.to_string(index=False));print('Advancement gate:',bool(gate))


if __name__=='__main__':main()
