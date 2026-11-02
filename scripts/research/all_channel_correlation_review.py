"""Build all-channel within-mission training correlation matrices and group lists."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mission_cross_channel_review import held_values, sha, timestamp_ns

ROOT = Path(__file__).resolve().parents[2]
PRIOR = ROOT / 'results_longrun/mission_cross_channel_20261102_v1'
PROTOCOL = ROOT / 'docs/research/all-channel-correlation-protocol-2026-11-02.md'


def matrices(x, quarter):
    frame = pd.DataFrame(x)
    delta = frame.diff()
    delta.iloc[np.flatnonzero(np.r_[True, quarter[1:] != quarter[:-1]])] = np.nan
    valid, dv = frame.notna().to_numpy(dtype=float), delta.notna().to_numpy(dtype=float)
    return (frame.corr(min_periods=100).to_numpy(),
            frame.corr(method='spearman', min_periods=100).to_numpy(),
            delta.corr(min_periods=100).to_numpy(),
            np.rint(valid.T @ valid).astype(np.int64),
            np.rint(dv.T @ dv).astype(np.int64))


def stable_matrix(stack, count, cutoff):
    return (np.isfinite(stack).all(axis=0) & (np.abs(stack) >= cutoff).all(axis=0)
            & ((stack > 0).all(axis=0) | (stack < 0).all(axis=0)) & (count >= 500).all(axis=0))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    prior_receipt = json.loads((PRIOR / 'complete.json').read_text())
    integrity_path = ROOT / 'docs/research/tables/mission-cross-channel-2026-11-02/raw-integrity.json'
    integrity = json.loads(integrity_path.read_text())
    fingerprints = {f['path']: f['sha256'] for m in integrity['missions'] for f in m['extracted_files']}
    frozen = {'protocol_sha256':sha(PROTOCOL),'script_sha256':sha(__file__),
              'sampling_source_sha256':sha(Path(__file__).with_name('mission_cross_channel_review.py')),
              'integrity_sha256':sha(integrity_path),'periods':{'1':['2000-01-01','2006-01-01'],
              '2':['2000-01-01','2001-01-01']},'minimum_correlation_pairs':100,
              'minimum_stability_pairs_per_quarter':500,'assessment_used':False}
    (out/'frozen.json').write_text(json.dumps(frozen,indent=2)+'\n')
    all_inventory, all_pairs, all_groups = [], [], []
    for mission, count in [(1,76),(2,100)]:
        base = ROOT / f'data/ESA-Mission{mission}'
        for filename in ['channels.csv','labels.csv']:
            assert sha(base/filename)==fingerprints[str((base/filename).relative_to(ROOT))]
        metadata = pd.read_csv(base/'channels.csv').set_index('Channel')
        labels = pd.read_csv(base/'labels.csv')
        labels['lo'], labels['hi'] = timestamp_ns(labels.StartTime), timestamp_ns(labels.EndTime)
        start,end = map(pd.Timestamp, frozen['periods'][str(mission)])
        labels = labels[(labels.lo<end.value)&(labels.hi>=start.value)]
        grid = pd.date_range(start,end,freq='5min',inclusive='left')
        quarter = grid.to_period('Q').astype(str).to_numpy()
        previous = PRIOR / f'mission{mission}_training_diagnostic_grid.npz'
        assert sha(previous)==next(f['sha256'] for f in prior_receipt['files'] if f['path']==previous.name)
        with np.load(previous) as z:
            assert np.array_equal(z['times'],grid.asi8)
            cached={c:z['values'][:,i] for i,c in enumerate(z['channels'])}
        channels=[f'channel_{n}' for n in range(1,count+1)]
        x=np.full((len(grid),count),np.nan)
        inventory=[]
        for i,c in enumerate(channels):
            path=base/'channels'/f'{c}.zip'
            assert sha(path)==fingerprints[str(path.relative_to(ROOT))]
            before=path.stat()
            if c in cached:
                values=cached[c]
                numeric=True; dtype='numeric; verified previous grid'
            else:
                raw=pd.read_pickle(path)
                raw=raw.loc[(raw.index>=start)&(raw.index<end)]
                assert len(raw.columns)==1
                dtype=str(raw.iloc[:,0].dtype)
                numeric=pd.api.types.is_numeric_dtype(raw.iloc[:,0])
                if numeric and len(raw):
                    intervals=labels.loc[labels.Channel==c,['lo','hi']].to_numpy()
                    values=held_values(timestamp_ns(raw.index),raw.iloc[:,0].to_numpy(),grid.asi8,intervals)
                else:values=np.full(len(grid),np.nan)
                del raw
            x[:,i]=values
            finite=values[np.isfinite(values)]
            const=bool(len(finite) and np.ptp(finite)==0)
            counter=(4<=i+1<=11) if mission==1 else (29<=i+1<=46)
            inventory.append(dict(mission=mission,channel=c,**metadata.loc[c].to_dict(),dtype=dtype,
                                  numeric=bool(numeric),known_counter=counter,constant=const,
                                  eligible_samples=int(len(finite)),coverage=float(len(finite)/len(grid)),
                                  status='categorical: Pearson N/A' if not numeric else
                                  ('no eligible samples' if not len(finite) else ('constant: Pearson N/A' if const else 'numeric'))))
            assert (before.st_size,before.st_mtime_ns)==(path.stat().st_size,path.stat().st_mtime_ns)
            print(f'M{mission} {c} {inventory[-1]["status"]} coverage={inventory[-1]["coverage"]:.3%}',flush=True)
        inv=pd.DataFrame(inventory);all_inventory.extend(inventory)
        np.savez_compressed(out/f'mission{mission}_training_grid.npz',times=grid.asi8,channels=np.array(channels),values=x)
        blocks=['pooled']+list(dict.fromkeys(quarter))
        stats=[]
        for b in blocks:
            mask=np.ones(len(grid),bool) if b=='pooled' else quarter==b
            stats.append(matrices(x[mask],quarter[mask]))
            print(f'M{mission} matrix {b} complete',flush=True)
        arrays=[np.stack([s[k] for s in stats]) for k in range(5)]
        np.savez_compressed(out/f'mission{mission}_block_matrices.npz',blocks=np.array(blocks),
                            channels=np.array(channels),pearson=arrays[0],spearman=arrays[1],delta=arrays[2],
                            pairs=arrays[3],delta_pairs=arrays[4])
        for name,a in zip(['pearson','spearman','delta','counts','delta_counts'],arrays):
            pd.DataFrame(a[0],index=channels,columns=channels).to_csv(out/f'mission{mission}_{name}.csv')
        stable=stable_matrix(arrays[0][1:],arrays[3][1:],.9)
        strong_changes=stable_matrix(arrays[2][1:],arrays[4][1:],.5)
        minimum=np.min(np.abs(arrays[0][1:]),axis=0)
        minimum_delta=np.min(np.abs(arrays[2][1:]),axis=0)
        pairs=[]
        for i,a in enumerate(channels):
            for j in range(i+1,count):
                b=channels[j]
                r=arrays[0][0,i,j]
                row=dict(mission=mission,a=a,b=b,pearson=r,spearman=arrays[1][0,i,j],
                         delta_pearson=arrays[2][0,i,j],pooled_pairs=int(arrays[3][0,i,j]),
                         min_quarter_pairs=int(arrays[3][1:,i,j].min()),
                         min_abs_quarter_pearson=minimum[i,j],min_abs_quarter_delta=minimum_delta[i,j],
                         high_pooled=bool(np.isfinite(r) and abs(r)>=.9),stable_high=bool(stable[i,j]),
                         stable_changes=bool(strong_changes[i,j]),
                         both_targets=bool(metadata.loc[a,'Target']=='YES' and metadata.loc[b,'Target']=='YES'),
                         counter_pair=bool(inventory[i]['known_counter'] or inventory[j]['known_counter']))
                pairs.append(row)
        all_pairs.extend(pairs)
        # Complete linkage guarantees every pair, rather than only chain neighbours, qualifies.
        for target_only in (False,True):
            selected=np.flatnonzero((inv.Target=='YES').to_numpy()) if target_only else np.arange(count)
            strength=np.where(stable,minimum,0)[np.ix_(selected,selected)]
            distance=1-strength;distance=(distance+distance.T)/2;np.fill_diagonal(distance,0)
            ids=fcluster(linkage(squareform(distance),method='complete'),t=.1+1e-12,criterion='distance')
            for group in sorted(set(ids)):
                indices=selected[ids==group]
                if len(indices)<2:continue
                tri=np.triu_indices(len(indices),1)
                pool=arrays[0][0][np.ix_(indices,indices)][tri]
                delt=arrays[2][0][np.ix_(indices,indices)][tri]
                all_groups.append(dict(mission=mission,scope='targets only' if target_only else 'all channels',
                                       members=','.join(str(i+1) for i in indices),size=len(indices),
                                       min_abs_quarter_r=float(minimum[np.ix_(indices,indices)][tri].min()),
                                       pooled_r_min=float(pool.min()),pooled_r_max=float(pool.max()),
                                       change_r_min=float(delt.min()),change_r_max=float(delt.max()),
                                       all_changes_stable=bool(strong_changes[np.ix_(indices,indices)][tri].all()),
                                       contains_counter=bool(inv.iloc[indices].known_counter.any())))
        print(f'M{mission}: all pairs and groups complete',flush=True)
        del x,cached,stats,arrays
    pd.DataFrame(all_inventory).to_csv(out/'channel_inventory.csv',index=False)
    pairs=pd.DataFrame(all_pairs)
    pairs['abs_pearson']=pairs.pearson.abs()
    pairs=pairs.sort_values(['mission','stable_high','abs_pearson'],ascending=[True,False,False])
    pairs.to_csv(out/'all_pairs.csv',index=False)
    pairs[pairs.stable_high].to_csv(out/'stable_high_pairs.csv',index=False)
    pd.DataFrame(all_groups).to_csv(out/'stable_groups.csv',index=False)
    assert sha(PROTOCOL)==frozen['protocol_sha256'] and sha(__file__)==frozen['script_sha256']
    receipt={'complete':True,'channels':176,'unique_pairs':len(pairs),'frozen':frozen,
             'files':[{'path':p.name,'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(out.iterdir()) if p.is_file()]}
    (out/'complete.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
