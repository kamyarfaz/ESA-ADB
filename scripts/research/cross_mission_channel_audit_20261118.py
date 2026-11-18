"""Training-only behavioral mapping and event coverage; no neural inference."""
import json
import sys
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mission_cross_channel_review import sha, profile, QKEYS, AKEYS, DKEYS

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results_longrun/cross_mission_transfer_20261118_v1/research'
PROTOCOL = ROOT/'docs/research/cross-mission-transfer-protocol-2026-11-18.md'
CORR = ROOT/'results_longrun/all_channel_correlations_20261102_v1'
INTEGRITY = ROOT/'docs/research/tables/mission-cross-channel-2026-11-02/raw-integrity.json'
SOURCE = [f'channel_{n}' for n in (*range(40,48),33,49,14,21,29)]
KEYS = QKEYS+AKEYS+DKEYS


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def assign(unary, allowed, source_matrices, target_matrices):
    """Fixed deterministic multistart constrained assignment, never duplicate inputs."""
    n,k=unary.shape
    if np.any(~np.isfinite(unary[allowed])):raise ValueError('Missing compatible profiles')
    tri=np.triu_indices(n,1)
    def parts(a):
        vals=[float(unary[np.arange(n),a].mean())]
        for s,t in zip(source_matrices,target_matrices):
            vals.append(float(np.abs(s[tri]-t[np.ix_(a,a)][tri]).mean()/2))
        return vals
    def cost(a):return float(np.mean(parts(a)))
    rng=np.random.default_rng(20261118); solutions=[]
    for start in range(8):
        price=np.where(allowed,unary if start==0 else rng.random((n,k)),1e6)
        r,a=linear_sum_assignment(price)
        if len(r)!=n or not allowed[r,a].all():raise ValueError('No distinct compatible assignment')
        now=cost(a)
        for move in range(50):
            candidate=None; value=now
            used=set(a.tolist())
            for i in range(n):
                for j in np.flatnonzero(allowed[i]):
                    trial=a.copy()
                    if j in used:
                        other=int(np.flatnonzero(a==j)[0])
                        if other<=i or not allowed[other,a[i]]:continue
                        trial[i],trial[other]=trial[other],trial[i]
                    else:trial[i]=j
                    score=cost(trial)
                    if score<value-1e-12:
                        value=score;candidate=trial
            if candidate is None:break
            a,now=candidate,value
        solutions.append((now,tuple(int(v) for v in a),parts(a),move+1))
    solutions.sort(key=lambda x:(x[0],x[1]))
    return list(solutions[0][1]), {'objective':solutions[0][0], 'cost_components':solutions[0][2],
                                'start_objectives':[x[0] for x in solutions]}


def pair_medians(matrices, indices, omit):
    blocks=matrices['blocks'][1:]
    keep=np.array([omit==0 or int(str(b)[-1])!=omit for b in blocks])
    return [np.nanmedian(matrices[key][1:][keep],axis=0)[np.ix_(indices,indices)] for key in ('pearson','delta')]


def coverage(mission, panels):
    base=ROOT/f'data/ESA-Mission{mission}'
    labels=pd.read_csv(base/'labels.csv').merge(pd.read_csv(base/'anomaly_types.csv'),on='ID',validate='many_to_one')
    stop=pd.Timestamp('2006-01-01' if mission==1 else '2001-01-01',tz='UTC')
    labels=labels[(pd.to_datetime(labels.StartTime,utc=True)<stop)&(pd.to_datetime(labels.EndTime,utc=True)>=pd.Timestamp('2000-01-01',tz='UTC'))]
    rows=[]
    for definition,categories in [('anomaly_and_rare',['Anomaly','Rare Event']),('anomaly_only',['Anomaly'])]:
        selected=labels[labels.Category.isin(categories)]
        system=set(selected.ID)
        for name,channels in panels.items():
            seen=set(selected.loc[selected.Channel.isin(channels),'ID'])
            rows.append(dict(mission=mission,panel=name,event_definition=definition,channels=len(channels),
                             system_events=len(system),events_with_annotation_on_input=len(seen),
                             events_without_annotation_on_input=len(system-seen),covered_ids=sorted(seen),uncovered_ids=sorted(system-seen)))
    return rows


def main():
    if (OUT/'complete.json').exists():raise SystemExit('Research already complete; verify, do not overwrite.')
    OUT.mkdir(parents=True,exist_ok=True)
    frozen=dict(started=datetime.now(timezone.utc).isoformat(),script_sha256=sha(__file__),protocol_sha256=sha(PROTOCOL),
                integrity_sha256=sha(INTEGRITY),source_channels=SOURCE,assessment_used=False,calibration_used=False)
    write(OUT/'frozen.json',frozen)
    receipt=json.loads(INTEGRITY.read_text());raw={}
    assert receipt['verified'] and all(m['official_md5_matched'] for m in receipt['missions'])
    for m in receipt['missions']:
        for item in m['extracted_files']:
            p=ROOT/item['path']
            if sha(p)!=item['sha256']:raise ValueError('Raw integrity mismatch '+str(p))
            raw[item['path']]=item['sha256']
    print('Verified',len(raw),'raw files against official-v1 receipt',flush=True)
    prior=json.loads((CORR/'complete.json').read_text()); hashes={f['path']:f['sha256'] for f in prior['files']}
    dependencies={}
    for m in (1,2):
        for suffix in ('training_grid.npz','block_matrices.npz'):
            name=f'mission{m}_{suffix}'
            if sha(CORR/name)!=hashes[name]:raise ValueError('Training cache changed '+name)
            dependencies[str((CORR/name).relative_to(ROOT))]=hashes[name]
    write(OUT/'integrity.json',dict(passed=True,raw=raw,training_cache=dependencies))
    meta={m:pd.read_csv(ROOT/f'data/ESA-Mission{m}/channels.csv').fillna('').set_index('Channel') for m in (1,2)}
    pairs={tuple(meta[1].loc[c,['Subsystem','Physical Unit']]) for c in SOURCE}
    candidates=[c for c,row in meta[2].iterrows() if row.Target=='YES' and tuple(row[['Subsystem','Physical Unit']]) in pairs]
    candidates.sort(key=lambda c:int(c.split('_')[1]))
    profiles=[]; matrices={};indices={};inventory=[]
    for m,selected in ((1,SOURCE),(2,candidates)):
        with np.load(CORR/f'mission{m}_training_grid.npz',allow_pickle=False) as z:
            times=z['times'];channels=list(z['channels']);values=z['values']
        end=pd.Timestamp('2006-01-01' if m==1 else '2001-01-01').value
        if times.min()!=pd.Timestamp('2000-01-01').value or times.max()>=end:raise ValueError('Nontraining cache')
        quarters=pd.DatetimeIndex(times).to_period('Q').astype(str).to_numpy()
        indices[m]=[channels.index(c) for c in selected]
        with np.load(CORR/f'mission{m}_block_matrices.npz',allow_pickle=False) as z:
            matrices[m]={k:z[k] for k in ('blocks','pearson','delta')}
            assert list(z['channels'])==channels
        for c in selected:
            col=values[:,channels.index(c)]
            for q in sorted(set(quarters)):
                profiles.append(dict(mission=m,channel=c,quarter=q,season=int(q[-1]),**profile(col[quarters==q])))
        print('Profiles complete mission',m,flush=True)
        del values
        for c,row in meta[m].iterrows():
            inventory.append(dict(mission=m,channel=c,**row.to_dict(),known_physical_identity=False))
    pd.DataFrame(inventory).to_csv(OUT/'channel_metadata.csv',index=False)
    prof=pd.DataFrame(profiles);prof.to_csv(OUT/'quarter_profiles.csv',index=False)
    allowed=np.array([[tuple(meta[1].loc[a,['Subsystem','Physical Unit']])==tuple(meta[2].loc[b,['Subsystem','Physical Unit']]) for b in candidates] for a in SOURCE])
    maps=[];primary=None;allcost=[]
    for omit in range(5):
        rows=prof if omit==0 else prof[prof.season!=omit]
        med=rows.groupby(['mission','channel'])[list(KEYS)].median()
        u=med.loc[[(1,c) for c in SOURCE]].to_numpy();v=med.loc[[(2,c) for c in candidates]].to_numpy()
        allv=np.concatenate([u,v]);sc=np.maximum(np.nanpercentile(allv,75,axis=0)-np.nanpercentile(allv,25,axis=0),.1)
        distances=np.minimum(np.abs(u[:,None,:]-v[None,:,:])/sc,3)/3
        lens=[len(QKEYS),len(AKEYS),len(DKEYS)];start=0;fam=[]
        for n in lens:fam.append(distances[:,:,start:start+n].mean(axis=2));start+=n
        unary=np.mean(fam,axis=0)
        sm=pair_medians(matrices[1],indices[1],omit);tm=pair_medians(matrices[2],indices[2],omit)
        a,info=assign(unary,allowed,sm,tm)
        mapping={c:candidates[j] for c,j in zip(SOURCE,a)}
        maps.append(dict(omitted_season=omit,mapping=mapping,**info))
        if omit==0:
            primary=mapping
            for i,c in enumerate(SOURCE):
                for j,b in enumerate(candidates):
                    if allowed[i,j]:allcost.append(dict(source=c,target=b,distribution_cost=float(fam[0][i,j]),autocorrelation_cost=float(fam[1][i,j]),changes_cost=float(fam[2][i,j]),unary_cost=float(unary[i,j]),selected=mapping[c]==b))
            for label,s,t in zip(('level','changes'),sm,tm):
                rows2=[]
                for i,c in enumerate(SOURCE):
                    for j in range(i+1,len(SOURCE)):
                        rows2.append(dict(source_a=c,source_b=SOURCE[j],target_a=mapping[c],target_b=mapping[SOURCE[j]],source_correlation=float(s[i,j]),target_correlation=float(t[a[i],a[j]]),absolute_error=float(abs(s[i,j]-t[a[i],a[j]]))))
                pd.DataFrame(rows2).to_csv(OUT/f'mapped_{label}_relationships.csv',index=False)
        print('Mapping omit-season',omit,'objective',info['objective'],flush=True)
    write(OUT/'mapping.json',dict(primary=primary,sensitivity=maps,physical_equivalence_established=False,selection_used_assessment=False))
    pd.DataFrame(allcost).to_csv(OUT/'compatible_pair_costs.csv',index=False)
    stability=[dict(source=c,target=primary[c],same_in_leave_season_out=sum(x['mapping'][c]==primary[c] for x in maps[1:]),sensitivity_runs=4) for c in SOURCE]
    pd.DataFrame(stability).to_csv(OUT/'mapping_stability.csv',index=False)
    cov=[]
    for m in (1,2):
        targets=meta[m].index[meta[m].Target=='YES'].tolist()
        oldbase=[f'channel_{i}' for i in (range(40,48) if m==1 else (58,59,75,76,77,83,84,85))]
        oldexpanded=SOURCE if m==1 else oldbase+[f'channel_{i}' for i in (90,86,89)]
        panels={'historical_base':oldbase,'historical_expanded':oldexpanded,'all_targets':targets,
                'compatible_families':[c for c in targets if tuple(meta[m].loc[c,['Subsystem','Physical Unit']]) in pairs],
                'mapped_10':SOURCE[:10] if m==1 else [primary[c] for c in SOURCE[:10]],
                'mapped_13':SOURCE if m==1 else [primary[c] for c in SOURCE]}
        cov.extend(coverage(m,panels))
    write(OUT/'training_event_coverage.json',cov)
    pd.DataFrame([{k:v for k,v in r.items() if not k.endswith('_ids')} for r in cov]).to_csv(OUT/'training_event_coverage.csv',index=False)
    if sha(__file__)!=frozen['script_sha256'] or sha(PROTOCOL)!=frozen['protocol_sha256']:raise ValueError('Research source changed')
    write(OUT/'complete.json',dict(completed=datetime.now(timezone.utc).isoformat(),research_complete=True,
         files={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='complete.json'}))
    print('RESEARCH COMPLETE',flush=True)

if __name__=='__main__':main()
