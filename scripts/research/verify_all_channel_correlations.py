"""Independent correlation reconstruction, counts, group and previous-subset checks."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def independent_pearson(x):
    valid=np.isfinite(x).astype(float)
    count=valid.T@valid
    means=np.nansum(x,axis=0)/np.maximum(valid.sum(axis=0),1)
    z=np.where(np.isfinite(x),x-means,0)
    sums=z.T@valid
    ss=(z*z).T@valid
    with np.errstate(divide='ignore',invalid='ignore'):
        cov=z.T@z-sums*sums.T/count
        var=np.maximum(0,ss-sums*sums/count)
        r=cov/np.sqrt(var*var.T)
    r[count<100]=np.nan
    # Pairwise deletion can leave a channel constant even when it varies over
    # its full column. The matrix subtraction then suffers cancellation. Replay
    # those pairs directly, independently of the stored/pandas coefficients.
    delicate=(var<=1e-12*np.abs(ss)) | (var.T<=1e-12*np.abs(ss.T)) | ~np.isfinite(r)
    for i,j in np.argwhere(np.triu(delicate & (count>=100))):
        ok=np.isfinite(x[:,i]) & np.isfinite(x[:,j])
        a,b=x[ok,i],x[ok,j]
        value=np.corrcoef(a,b)[0,1] if np.ptp(a)>0 and np.ptp(b)>0 else np.nan
        r[i,j]=r[j,i]=value
    return r,count


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run',type=Path,required=True)
    args=ap.parse_args();out=args.run
    root=Path(__file__).resolve().parents[2]
    receipt=json.loads((out/'complete.json').read_text())
    for f in receipt['files']:assert sha(out/f['path'])==f['sha256']
    assert sha(root/'docs/research/all-channel-correlation-protocol-2026-11-02.md')==receipt['frozen']['protocol_sha256']
    assert sha(root/'scripts/research/all_channel_correlation_review.py')==receipt['frozen']['script_sha256']
    pairs=pd.read_csv(out/'all_pairs.csv');groups=pd.read_csv(out/'stable_groups.csv')
    pooled_count=0;quarter_count=0;spearman_count=0;subset_cells=0
    for m,n in [(1,76),(2,100)]:
        z=np.load(out/f'mission{m}_training_grid.npz');t=z['times'];x=z['values'];channels=z['channels']
        assert x.shape==(len(t),n) and len(channels)==n
        assert t.min()==pd.Timestamp('2000-01-01').value
        assert t.max()<pd.Timestamp('2006-01-01' if m==1 else '2001-01-01').value
        assert np.all(np.diff(t)==300*10**9)
        b=np.load(out/f'mission{m}_block_matrices.npz')
        q=pd.to_datetime(t).to_period('Q').astype(str).to_numpy()
        for key in ['pearson','spearman','delta']:
            a=b[key]
            np.testing.assert_allclose(a,a.transpose(0,2,1),atol=1e-10,equal_nan=True)
            assert np.nanmax(np.abs(a))<=1+1e-10
        r,count=independent_pearson(x)
        np.testing.assert_allclose(r,b['pearson'][0],atol=1e-8,rtol=1e-8,equal_nan=True)
        np.testing.assert_equal(count,b['pairs'][0])
        d=np.diff(x,axis=0);d[q[1:]!=q[:-1]]=np.nan
        dr,dc=independent_pearson(d)
        np.testing.assert_allclose(dr,b['delta'][0],atol=1e-8,rtol=1e-8,equal_nan=True)
        np.testing.assert_equal(dc,b['delta_pairs'][0]);pooled_count+=n*(n-1)//2
        own=pairs[pairs.mission==m]
        assert len(own)==n*(n-1)//2
        positions={c:i for i,c in enumerate(channels)}
        for row in own.itertuples():
            i,j=positions[row.a],positions[row.b]
            np.testing.assert_allclose([row.pearson,row.spearman,row.delta_pearson],
                                        [b['pearson'][0,i,j],b['spearman'][0,i,j],b['delta'][0,i,j]],equal_nan=True,atol=1e-10)
            vals=b['pearson'][1:,i,j]
            expected=bool(np.isfinite(vals).all() and (np.abs(vals)>=.9).all() and
                          ((vals>0).all() or (vals<0).all()) and (b['pairs'][1:,i,j]>=500).all())
            assert row.stable_high==expected
        stable=own[own.stable_high]
        for block in b['blocks'][1:]:
            block_i=list(b['blocks']).index(block)
            xx=x[q==block]
            dd=np.diff(xx,axis=0)
            for row in stable.itertuples():
                i,j=positions[row.a],positions[row.b]
                for data,key in [(xx,'pearson'),(dd,'delta')]:
                    ok=np.isfinite(data[:,i])&np.isfinite(data[:,j])
                    a,c=data[ok,i],data[ok,j]
                    value=np.corrcoef(a,c)[0,1] if len(a)>=100 and np.ptp(a)>0 and np.ptp(c)>0 else np.nan
                    np.testing.assert_allclose(value,b[key][block_i,i,j],atol=1e-8,rtol=1e-8,equal_nan=True)
                quarter_count+=1
        rng=np.random.default_rng(730+m)
        for index in rng.choice(len(own),size=20,replace=False):
            row=own.iloc[index];i,j=positions[row.a],positions[row.b]
            mask=np.isfinite(x[:,i])&np.isfinite(x[:,j])
            a,c=x[mask,i],x[mask,j]
            v=spearmanr(a,c).statistic if len(a)>=100 and np.ptp(a)>0 and np.ptp(c)>0 else np.nan
            np.testing.assert_allclose(v,b['spearman'][0,i,j],atol=1e-9,equal_nan=True);spearman_count+=1
        prior=root/'results_longrun/mission_cross_channel_20261102_v1'
        old=np.load(prior/f'mission{m}_training_diagnostic_grid.npz')
        idx=[positions[c] for c in old['channels']]
        np.testing.assert_equal(x[:,idx],old['values'])
        for key in ['pearson','spearman','delta']:
            oldr=pd.read_csv(prior/f'mission{m}_{key}.csv',index_col=0).to_numpy()
            np.testing.assert_allclose(b[key][0][np.ix_(idx,idx)],oldr,atol=1e-9,equal_nan=True)
            subset_cells+=oldr.size
        for g in groups[groups.mission==m].itertuples():
            cs=[f'channel_{c}' for c in g.members.split(',')]
            sub=own[own.a.isin(cs)&own.b.isin(cs)]
            assert len(sub)==len(cs)*(len(cs)-1)//2 and sub.stable_high.all()
        print(f'Mission{m} verified',flush=True)
    result={'verified':True,'pooled_unique_pairs_pearson_and_changes':pooled_count,
            'stable_pair_quarter_replays':quarter_count,'sampled_pooled_spearman_pairs':spearman_count,
            'prior_subset_matrix_cells':subset_cells,'prior_training_grids_exactly_preserved':True,
            'hashes_verified':len(receipt['files']),'assessment_used':False}
    (out/'independent-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
