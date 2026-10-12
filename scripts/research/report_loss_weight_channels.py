"""Export compact tables and plots from the completed frozen channel replay."""
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
    if a.output.exists():p.error('Use a new output')
    a.output.mkdir(parents=True)
    for f in ('channel_monthly.csv','replay.csv','verification.json'):shutil.copy2(a.source/f,a.output/f)
    d=pd.read_csv(a.source/'channel_monthly.csv')
    fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    for r,arm in enumerate(('lambda1','lambda0.1')):
        for c,seed in enumerate((42,43,44)):
            ax=axes[r,c];s=d[(d.arm==arm)&(d.seed==seed)]
            for ch,rows in s.groupby('channel'):
                ax.plot(pd.to_datetime(rows.month),rows.mse_median,label=ch.replace('channel_','ch'))
            ax.axvline(pd.Timestamp('2002-01-01'),color='grey',linestyle=':')
            ax.axvline(pd.Timestamp('2002-07-01'),color='grey',linestyle='--')
            ax.set_title(f'{arm}, seed {seed}');ax.set_ylabel('Median normal-window channel MSE')
            ax.tick_params(axis='x',rotation=30)
    axes[0,0].legend(ncol=2,fontsize=7)
    fig.savefig(a.output/'channel_errors.png',dpi=140);plt.close(fig)
    print(d[d.month.isin(['2001-12','2002-06','2002-12'])].sort_values('mse_median').groupby(['arm','seed','month']).tail(1)[['arm','seed','month','channel','scaled_window_mean','reconstruction_window_mean','mse_median','winner_fraction','clipping_fraction']].to_string(index=False))


if __name__=='__main__':main()
