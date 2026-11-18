"""Independent replay of research profiles, mapping costs and annotation coverage."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from esa_thesis.calibration_audit import digest
from esa_thesis.full_similarity_mosad import verify_receipt

OUT=ROOT/'results_longrun/cross_mission_transfer_20261118_v1/research'
CORR=ROOT/'results_longrun/all_channel_correlations_20261102_v1'


def main():
    assert verify_receipt(OUT,'complete.json')
    prof=pd.read_csv(OUT/'quarter_profiles.csv');checked=0;mat={}
    for mission in (1,2):
        with np.load(CORR/f'mission{mission}_training_grid.npz',allow_pickle=False) as z:
            values=z['values'];channels=list(z['channels']);seasons=pd.DatetimeIndex(z['times']).to_period('Q').astype(str).to_numpy()
        with np.load(CORR/f'mission{mission}_block_matrices.npz',allow_pickle=False) as z:mat[mission]={k:z[k] for k in z.files}
        for row in prof[prof.mission==mission].itertuples(index=False):
            s=pd.Series(values[seasons==row.quarter,channels.index(row.channel)])
            quant=s.dropna().quantile([.01,.1,.25,.5,.75,.9,.99]).to_numpy();scale=quant[4]-quant[2]
            for k,v in zip(('q01','q10','q25','q75','q90','q99'),(quant[[0,1,2,4,5,6]]-quant[3])/scale):
                np.testing.assert_allclose(getattr(row,k),v,rtol=0,atol=2e-10);checked+=1
            for lag in (1,6,12,72,288):
                keep=s.rolling(lag+1,min_periods=lag+1).count()==lag+1
                value=s[keep].corr(s.shift(lag)[keep])
                np.testing.assert_allclose(getattr(row,f'acf_{lag}'),value,rtol=0,atol=2e-10);checked+=1
            delta=s.diff().abs().dropna()
            for k,v in [('log_delta_median',np.log1p(delta.median()/scale)),('log_delta_p95',np.log1p(delta.quantile(.95)/scale)),('delta_zero_fraction',(delta==0).mean())]:
                np.testing.assert_allclose(getattr(row,k),v,rtol=0,atol=2e-10);checked+=1
        del values
    maps=json.loads((OUT/'mapping.json').read_text());meta={m:pd.read_csv(ROOT/f'data/ESA-Mission{m}/channels.csv').fillna('').set_index('Channel') for m in (1,2)}
    fields=[['q01','q10','q25','q75','q90','q99'],['acf_1','acf_6','acf_12','acf_72','acf_288'],['log_delta_median','log_delta_p95','delta_zero_fraction']]
    keys=sum(fields,[]);candidates=prof[prof.mission==2].channel.drop_duplicates().tolist();source=list(maps['primary'])
    cost_checks=[]
    for record in maps['sensitivity']:
        omit=record['omitted_season'];mapping=record['mapping'];assert set(mapping)==set(source) and len(set(mapping.values()))==len(source)
        for a,b in mapping.items():assert tuple(meta[1].loc[a,['Subsystem','Physical Unit']])==tuple(meta[2].loc[b,['Subsystem','Physical Unit']])
        subset=prof if omit==0 else prof[prof.season!=omit]
        med=subset.groupby(['mission','channel'])[keys].median()
        u=med.loc[[(1,c) for c in source]].to_numpy();v=med.loc[[(2,c) for c in candidates]].to_numpy()
        pooled=np.r_[u,v];scale=np.maximum(np.quantile(pooled,.75,axis=0)-np.quantile(pooled,.25,axis=0),.1)
        mapped=np.array([v[candidates.index(mapping[c])] for c in source]);d=np.clip(np.abs(u-mapped)/scale,0,3)/3
        unary=np.mean([d[:,[keys.index(k) for k in f]].mean() for f in fields])
        costs=[float(unary)]
        for key in ('pearson','delta'):
            matrices=[]
            for m,cs in [(1,source),(2,[mapping[c] for c in source])]:
                indices=[list(mat[m]['channels']).index(c) for c in cs]
                arr=mat[m][key][1:][:,indices][:,:,indices]
                keep=np.array([not omit or int(str(q)[-1])!=omit for q in mat[m]['blocks'][1:]])
                matrices.append(np.median(arr[keep],axis=0))
            tri=np.triu_indices(len(source),1);costs.append(float(np.mean(np.abs(matrices[0][tri]-matrices[1][tri]))/2))
        np.testing.assert_allclose(record['cost_components'],costs,atol=1e-12,rtol=0)
        np.testing.assert_allclose(record['objective'],np.mean(costs),atol=1e-12,rtol=0)
        cost_checks.append(dict(omitted_season=omit,objective=float(np.mean(costs))))
    coverage=json.loads((OUT/'training_event_coverage.json').read_text())
    for m in (1,2):
        base=ROOT/f'data/ESA-Mission{m}'
        ann=pd.read_csv(base/'labels.csv').merge(pd.read_csv(base/'anomaly_types.csv'),on='ID')
        start=pd.to_datetime(ann.StartTime,utc=True);end=pd.to_datetime(ann.EndTime,utc=True)
        stop=pd.Timestamp('2006-01-01' if m==1 else '2001-01-01',tz='UTC')
        ann=ann[(start<stop)&(end>=pd.Timestamp('2000-01-01',tz='UTC'))]
        for row in [r for r in coverage if r['mission']==m]:
            cats=['Anomaly'] if row['event_definition']=='anomaly_only' else ['Anomaly','Rare Event']
            ids=set(ann[ann.Category.isin(cats)].ID)
            assert set(row['covered_ids']).isdisjoint(row['uncovered_ids'])
            assert set(row['covered_ids'])|set(row['uncovered_ids'])==ids
            assert len(ids)==row['system_events'] and len(row['covered_ids'])==row['events_with_annotation_on_input']
            oldbase=[f'channel_{i}' for i in (range(40,48) if m==1 else (58,59,75,76,77,83,84,85))]
            historical_expanded=source if m==1 else oldbase+['channel_90','channel_86','channel_89']
            targets=meta[m].index[meta[m].Target=='YES'].tolist()
            families={tuple(meta[1].loc[c,['Subsystem','Physical Unit']]) for c in source}
            panels={'historical_base':oldbase,'historical_expanded':historical_expanded,'all_targets':targets,
                    'compatible_families':[c for c in targets if tuple(meta[m].loc[c,['Subsystem','Physical Unit']]) in families],
                    'mapped_10':source[:10] if m==1 else [maps['primary'][c] for c in source[:10]],
                    'mapped_13':source if m==1 else list(maps['primary'].values())}
            covered=set(ann[ann.Category.isin(cats)&ann.Channel.isin(panels[row['panel']])].ID)
            assert covered==set(row['covered_ids'])
    prior=OUT.with_name('research-attempt1-utc-error')/'mapping.json'
    assert json.loads(prior.read_text())==maps
    receipt=dict(passed=True,profile_values_replayed=checked,mapping_costs=cost_checks,coverage_rows_checked=len(coverage),
        first_attempt_failure='UTC-aware/naive annotation comparison; retained. Repaired bounds only; all mapping outputs identical.',
        research_sha256=digest(OUT/'complete.json'),verifier_sha256=digest(__file__),
        limitations='Profiles replay verified training caches; no claim of recovered physical sensor identities or globally optimal assignment.')
    (OUT.parent/'research-verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
