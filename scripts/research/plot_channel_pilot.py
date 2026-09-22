"""Presentation figures from the recorded training-only audit and result snapshot."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[2];base=root/'research_workspace/channel_study_2026-09-22'
out=root/'outputs/esa-thesis-review-20260922';out.mkdir(exist_ok=True,parents=True)
plt.rcParams.update({'font.size':10,'figure.dpi':150})
rows=pd.DataFrame(json.loads((base/'results_snapshot.json').read_text())['rows'])
fig,axes=plt.subplots(1,2,figsize=(12,4.5))
for name,label,color in [('Baseline AE + pseudo','8 channels + pseudo','#2563a6'),('Expanded AE + pseudo','11 channels + pseudo','#d98724'),('Reconstruction-only AE','8 channels, reconstruction only','#44856d')]:
 d=rows[rows.model==name].sort_values('year')
 axes[0].plot(d.year,d.assessment_f05,'o-',label=label,color=color)
 axes[1].plot(d.year,d.FPe,'o-',label=label,color=color)
axes[0].axhline(.85,color='gray',ls='--',lw=1,label='Target 0.85');axes[0].set_ylim(0,1)
axes[0].set_ylabel('ESA event-wise F0.5');axes[1].set_ylabel('Wholly false alarm events');axes[1].set_ylim(bottom=0)
for ax in axes:ax.set_xticks([2003,2004,2005]);ax.set_xlabel('Assessment year (April–December)');ax.grid(alpha=.2)
axes[0].legend(fontsize=8,loc='lower left');fig.suptitle('Completed AE comparisons • seed 42 • calibration-selected rules')
fig.tight_layout();fig.savefig(out/'assessment_comparison.png');plt.close(fig)
corr=pd.read_csv(base/'pilot/pearson.csv',index_col=0);channels=sorted([c for c in corr if c.startswith('channel_')],key=lambda x:int(x.split('_')[1]));v=corr.loc[channels,channels]
fig,ax=plt.subplots(figsize=(10,9));im=ax.imshow(v,vmin=-1,vmax=1,cmap='RdBu_r');ticks=np.arange(0,len(channels),4)
ax.set_xticks(ticks,[channels[i].replace('channel_','') for i in ticks],rotation=90);ax.set_yticks(ticks,[channels[i].replace('channel_','') for i in ticks]);ax.set_xlabel('Channel ID');ax.set_ylabel('Channel ID');ax.set_title('Training-only Pearson correlation\n2000–2002 hourly nominal sample; constant sampled inputs omitted');fig.colorbar(im,ax=ax,label='Pearson r',shrink=.8);fig.tight_layout();fig.savefig(out/'training_correlation.png');plt.close(fig)
sample=pd.read_csv(base/'pilot/nominal_hourly_sample.csv');sample['timestamp']=pd.to_datetime(sample.timestamp,utc=True)
fig,axes=plt.subplots(2,1,figsize=(11,6))
for c in ['channel_14','channel_21','channel_29']:
 monthly=sample.set_index('timestamp')[c].resample('MS').median();axes[0].plot(monthly.index,monthly,label=c)
axes[0].set_ylabel('Prepared value (anonymized units)');axes[0].legend();axes[0].set_title('Monthly median levels • hourly nominal training sample')
axes[1].scatter(sample.channel_14.iloc[::5],sample.channel_21.iloc[::5],s=3,alpha=.2);axes[1].set_xlabel('channel_14');axes[1].set_ylabel('channel_21');axes[1].set_title('Sampled relationship; high correlation does not establish redundancy')
fig.tight_layout();fig.savefig(out/'training_channel_examples.png');plt.close(fig)
print('Wrote three scientific figures')
