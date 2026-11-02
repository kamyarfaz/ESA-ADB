"""Render matrix figures and prepare typed scientific results for workbook export."""
import argparse,json,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def records(frame):
    return json.loads(frame.to_json(orient='records',double_precision=15))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    pairs=pd.read_csv(args.run/'all_pairs.csv');groups=pd.read_csv(args.run/'stable_groups.csv')
    inventory=pd.read_csv(args.run/'channel_inventory.csv')
    cm=plt.get_cmap('RdBu_r').copy();cm.set_bad('#D1D5DB')
    payload={'matrices':[],'groups':records(groups),'inventory':records(inventory),
             'pairs':records(pairs[pairs.high_pooled]),'all_pair_count':len(pairs),
             'counts':{},'periods':{1:'2000–2005',2:'2000'}}
    for m in (1,2):
        x=pairs[pairs.mission==m]
        payload['counts'][m]={'all_pairs':len(x),'high_pooled':int(x.high_pooled.sum()),
                              'stable':int(x.stable_high.sum()),
                              'stable_targets':int((x.stable_high & x.both_targets).sum()),
                              'stable_positive':int((x.stable_high & (x.pearson>0)).sum()),
                              'stable_negative':int((x.stable_high & (x.pearson<0)).sum())}
    with PdfPages(out/'ESA_all_channel_correlation_matrices.pdf') as pdf:
        for m in (1,2):
            for key,title in [('pearson','Pearson: signal levels'),('delta','Pearson: five-minute changes'),('spearman','Spearman: signal ranks')]:
                r=pd.read_csv(args.run/f'mission{m}_{key}.csv',index_col=0)
                labels=[str(i) for i in range(1,len(r)+1)]
                payload['matrices'].append({'mission':m,'kind':key,'title':title,'channels':labels,
                                            'values':json.loads(r.to_json(orient='values',double_precision=15))})
                fig,ax=plt.subplots(figsize=(15,14),constrained_layout=True)
                im=ax.imshow(r.to_numpy(),vmin=-1,vmax=1,cmap=cm,interpolation='nearest')
                ax.set_xticks(range(len(r)),labels,rotation=90,fontsize=7)
                ax.set_yticks(range(len(r)),labels,fontsize=7)
                ax.set_xlabel('Channel number');ax.set_ylabel('Channel number')
                ax.set_title(f'Mission {m} — {title}\nAll {len(r)} channels · training {payload["periods"][m]} · nominal samples only',fontsize=16,pad=18)
                fig.colorbar(im,ax=ax,shrink=.78,label='Correlation coefficient (−1 to +1)')
                fig.text(.5,.002,'Gray = undefined (categorical, constant or insufficient samples). Raw counters retained and flagged in the workbook.',ha='center',fontsize=9)
                fig.savefig(out/f'mission{m}_{key}_all_channels.png',dpi=180)
                pdf.savefig(fig);plt.close(fig)
        for m,cs in [(1,list(range(40,48))),(2,[58,59,75,76,77,83,84,85])]:
            fig,axes=plt.subplots(1,2,figsize=(13,6),constrained_layout=True)
            for ax,key,title in zip(axes,['pearson','delta'],['Signal levels','Five-minute changes']):
                names=[f'channel_{c}' for c in cs]
                r=pd.read_csv(args.run/f'mission{m}_{key}.csv',index_col=0).loc[names,names].to_numpy()
                im=ax.imshow(r,vmin=-1,vmax=1,cmap=cm)
                ax.set_xticks(range(8),cs);ax.set_yticks(range(8),cs);ax.set_title(title)
                for i in range(8):
                    for j in range(8):
                        v=r[i,j]
                        ax.text(j,i,f'{v:.2f}' if np.isfinite(v) else 'n.a.',ha='center',va='center',fontsize=9,
                                color='white' if np.isfinite(v) and abs(v)>.65 else '#17202A')
            fig.suptitle(f'Mission {m}: '+('original channels 40–47' if m==1 else 'proposed eight-channel candidate')+f'\nTraining {payload["periods"][m]} · within-mission correlations',fontsize=15)
            fig.colorbar(im,ax=axes,shrink=.7,label='Pearson r')
            fig.savefig(out/f'mission{m}_focused_correlations.png',dpi=160)
            pdf.savefig(fig);plt.close(fig)
    (out/'workbook-data.json').write_text(json.dumps(payload,allow_nan=False)+'\n')
    with zipfile.ZipFile(out/'ESA_all_channel_correlation_csv.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(args.run.glob('*.csv')):z.write(p,p.name)
        for name in ['frozen.json','complete.json','independent-verification.json']:
            if (args.run/name).exists():z.write(args.run/name,name)
    (out/'summary-counts.json').write_text(json.dumps(payload['counts'],indent=2)+'\n')
    print(json.dumps(payload['counts'],indent=2),flush=True)


if __name__=='__main__':main()
