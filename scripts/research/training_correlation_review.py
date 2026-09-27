"""Full-resolution, nominal-only Pearson review for 2000–2005; no model imports."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage,fcluster
from scipy.spatial.distance import squareform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Moments:
    def __init__(self,k):
        self.n=0;self.mean=np.zeros(k);self.m2=np.zeros((k,k))
        self.lo=np.full(k,np.inf);self.hi=np.full(k,-np.inf)
    def add(self,x):
        if not len(x):return
        n=len(x);mean=x.mean(axis=0);z=x-mean;delta=mean-self.mean
        self.m2+=z.T@z+np.outer(delta,delta)*(self.n*n/(self.n+n))
        self.mean+=delta*(n/(self.n+n));self.n+=n
        self.lo=np.minimum(self.lo,x.min(axis=0));self.hi=np.maximum(self.hi,x.max(axis=0))
    def corr(self):
        scale=np.sqrt(np.maximum(np.diag(self.m2),0));den=np.outer(scale,scale)
        with np.errstate(divide='ignore',invalid='ignore'):r=self.m2/den
        r=np.clip(r,-1,1);constant=self.lo==self.hi;r[constant,:]=np.nan;r[:,constant]=np.nan
        return r

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    out=p.parse_args().output;out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[2]
    source=root/'data/preprocessed/multivariate/ESA-Mission1-semi-supervised/84_months.train.csv'
    before=source.stat();header=pd.read_csv(source,nrows=0).columns
    features=sorted([c for c in header if c.startswith('channel_')],key=lambda c:int(c.split('_')[1]))
    labels=[c for c in header if c.startswith('is_anomaly_')]
    assert len(features)==76 and len(labels)==87
    years=list(range(2000,2006));mom={y:Moments(76) for y in years};delta={y:Moments(76) for y in years};total=Moments(76)
    rows=0;previous=None;samples=[]
    for frame in pd.read_csv(source,usecols=['timestamp']+features+labels,chunksize=50000):
        t=pd.to_datetime(frame.timestamp);keep=t<'2006-01-01';frame=frame.loc[keep];t=t[keep]
        if frame.empty:break
        x=frame[features].to_numpy(dtype=np.float64);lab=frame[labels].to_numpy();normal=(lab==0).all(axis=1)
        assert np.isfinite(x).all() and np.isin(lab,[0,1,2,3]).all()
        ns=t.astype('int64').to_numpy();yr=t.dt.year.to_numpy();assert (np.diff(ns)==30000000000).all()
        if previous is not None:assert ns[0]-previous[0]==30000000000
        for y in np.unique(yr):
            selected=normal&(yr==y);mom[int(y)].add(x[selected]);total.add(x[selected])
            valid=normal[1:]&normal[:-1]&(yr[1:]==y)&(yr[:-1]==y)
            delta[int(y)].add((x[1:]-x[:-1])[valid])
        if previous is not None and previous[2] and normal[0] and previous[3]==yr[0]:delta[int(yr[0])].add((x[0]-previous[1])[None,:])
        previous=(ns[-1],x[-1].copy(),normal[-1],yr[-1])
        sample=((np.arange(len(x))+rows)%120==0)&normal
        s=pd.DataFrame(x[sample],columns=features);s.insert(0,'timestamp',t[sample].to_numpy());samples.append(s)
        rows+=len(x)
        if rows%500000==0:print('Read',rows,flush=True)
        if not keep.all():break
    assert rows==int((pd.Timestamp('2006-01-01')-pd.Timestamp('2000-01-01')).total_seconds()/30)
    after=source.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    matrices=[]
    for y in years:
        r=mom[y].corr();matrices.append(r)
        pd.DataFrame(r,index=features,columns=features).to_csv(out/f'pearson_{y}.csv')
        pd.DataFrame(delta[y].corr(),index=features,columns=features).to_csv(out/f'delta_pearson_{y}.csv')
    full=total.corr();pd.DataFrame(full,index=features,columns=features).to_csv(out/'pearson_full.csv')
    sample=pd.concat(samples,ignore_index=True)
    sample[features].corr(method='spearman').to_csv(out/'spearman_hourly.csv')
    # All pairs retained; missing correlation means undefined, never zero evidence.
    pairs=[];stack=np.stack(matrices)
    for i,a in enumerate(features):
        for j in range(i+1,len(features)):
            v=stack[:,i,j];complete=np.isfinite(v).all();same=complete and ((v>0).all() or (v<0).all())
            r=dict(feature_a=a,feature_b=features[j],pearson_full=full[i,j],defined_years=int(np.isfinite(v).sum()),same_sign_all_years=bool(same),min_abs_yearly=float(np.abs(v).min()) if complete else None)
            r.update({f'pearson_{y}':v[k] for k,y in enumerate(years)});pairs.append(r)
    pairs=pd.DataFrame(pairs);pairs.to_csv(out/'all_pairs.csv',index=False)
    stable=np.isfinite(stack).all(axis=0)&((stack>0).all(axis=0)|(stack<0).all(axis=0))
    strength=np.where(stable,np.nan_to_num(np.abs(stack),nan=0).min(axis=0),0)
    distance=1-strength;distance=(distance+distance.T)/2;np.fill_diagonal(distance,0)
    groups=[]
    for cutoff in [.90,.95]:
        ids=fcluster(linkage(squareform(distance),method='complete'),t=1-cutoff+1e-12,criterion='distance')
        for group in sorted(set(ids)):
            members=[features[i] for i in np.flatnonzero(ids==group)]
            groups.append(dict(min_abs_pearson_each_year=cutoff,group=int(group),size=len(members),members=','.join(members),purpose='diagnostic only; singleton channels retained'))
    pd.DataFrame(groups).to_csv(out/'stable_groups.csv',index=False)
    pd.DataFrame([dict(year=y,nominal_rows=mom[y].n,adjacent_nominal_pairs=delta[y].n,constant_channels=','.join(np.array(features)[mom[y].lo==mom[y].hi])) for y in years]).to_csv(out/'coverage.csv',index=False)
    fig,axes=plt.subplots(2,3,figsize=(14,9),constrained_layout=True)
    for ax,y,r in zip(axes.flat,years,matrices):
        im=ax.imshow(r,vmin=-1,vmax=1,cmap='RdBu_r');ax.set_title(str(y));ax.set_xticks([0,19,39,59,75],[1,20,40,60,76]);ax.set_yticks([0,19,39,59,75],[1,20,40,60,76])
    fig.colorbar(im,ax=axes.ravel().tolist(),label='Pearson correlation; blank = undefined',shrink=.7)
    fig.suptitle('All 76 telemetry channels • full-resolution globally nominal training data\n2000–2005 only; channel IDs on axes; no calibration/test data')
    fig.savefig(out/'yearly_correlation.png',dpi=150);plt.close(fig)
    protocol=dict(interval='[2000-01-01,2006-01-01)',rows=rows,nominal_rows=total.n,telemetry=76,label_columns=87,
        nominal='All 87 paired labels equal zero',pearson='All eligible 30-second samples; stable centered covariance aggregation',
        differences='Adjacent 30-second samples both nominal, same calendar year; no bridging excluded periods',
        spearman='Exploratory fixed hourly nominal sample only',sample_rows=len(sample),
        clustering='Complete linkage on 1-min(abs(yearly Pearson)), same sign and defined in all six years; diagnostic 0.90 and 0.95 cuts; retain all singletons',
        limitations='Prepared-data relationships; inherited imputation; Pearson is linear association, not utility/causality. No automatic input selection or training.',
        source_size=before.st_size,source_mtime_ns=before.st_mtime_ns,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');print(json.dumps(protocol,indent=2),flush=True)

if __name__=='__main__':main()
