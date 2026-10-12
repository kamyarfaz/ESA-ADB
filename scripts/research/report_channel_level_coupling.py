"""Summarize the frozen-model intervention, without event metrics or tuning."""
import argparse
import json
import shutil
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert json.loads((a.source/'verification.json').read_text())['status']=='complete'
    if a.output.exists():p.error('Use new output')
    a.output.mkdir(parents=True)
    for f in ('monthly_interventions.csv','training_level_sweep.csv','training_references.csv','verification.json'):shutil.copy2(a.source/f,a.output/f)
    d=pd.read_csv(a.source/'monthly_interventions.csv');s=pd.read_csv(a.source/'training_level_sweep.csv')
    # Same untouched outputs 41–46 for both interventions (excludes ch40 and ch47).
    common=d[d.channel.isin([f'channel_{i}' for i in range(41,47)])]
    summary=common.groupby(['arm','seed','month','intervention']).mean_mse.mean().unstack('intervention').reset_index()
    summary['ch47_recenter_reduction']=1-summary.recenter_ch47/summary.original
    summary['ch40_recenter_reduction']=1-summary.recenter_ch40/summary.original
    summary.to_csv(a.output/'unchanged_outputs_summary.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for arm,style in [('lambda1','--'),('lambda0.1','-')]:
        for seed,color in zip((42,43,44),('tab:blue','tab:orange','tab:green')):
            sub=s[(s.arm==arm)&(s.seed==seed)]
            axes[0].plot(sub.added_level,sub.other_channels_mean_mse,style,color=color,label=f'{arm} seed{seed}',marker='o')
            sub=summary[(summary.arm==arm)&(summary.seed==seed)&(summary.month>='2002-01')]
            axes[1].plot(sub.month.str[-2:],sub.ch47_recenter_reduction*100,style,color=color,marker='o')
    axes[0].set_xlabel('Synthetic ch47 level above training reference')
    axes[0].set_ylabel('Unchanged channels 40–46: mean MSE');axes[0].set_title('Training-window level intervention');axes[0].legend(fontsize=7)
    axes[1].set_xlabel('Calibration month (2002)');axes[1].set_ylabel('Channels 41–46 MSE reduction (%)');axes[1].set_title('Recenter ch47; preserve other inputs')
    fig.savefig(a.output/'coupling.png',dpi=150);plt.close(fig)
    print(summary[summary.month=='2002-06'].to_string(index=False))
    print(s[s.added_level.isin([0,4])].to_string(index=False))


if __name__=='__main__':main()
