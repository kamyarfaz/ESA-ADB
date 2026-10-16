"""Recompute pre-2006 annotation coverage and summarize verified internal score audits."""
import json
from pathlib import Path
import pandas as pd
from esa_thesis.calibration_audit import verify_recorded, load_annotations
from esa_thesis.evaluation.esa import CATEGORIES

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/research/tables/development-evidence-2026-10-16'
if OUT.exists(): raise ValueError('Preserve existing evidence')
paths=[ROOT/'data/ESA-Mission1/labels.csv',ROOT/'data/ESA-Mission1/anomaly_types.csv',ROOT/'docs/research/tables/inner-event-audit-2026-10-08/events.csv',ROOT/'docs/research/tables/calibration-diversity-2026-10-13/periods.csv']
# October 13's diversity table postdates the original migration manifest;
# recompute coverage from original-hash annotations rather than trust that table.
hashes=verify_recorded(ROOT,paths[:3])
labels=load_annotations(*paths[:2])
labels=labels[labels.Category.isin(CATEGORIES)]
inputs={f'channel_{i}' for i in range(40,48)}
rows=[]; events=[]
for name,start,end in [('2000-2005','2000-01-01','2006-01-01')]+[(str(y),f'{y}-01-01',f'{y+1}-01-01') for y in range(2000,2006)]:
    part=labels[(labels.StartTime<pd.Timestamp(end,tz='UTC'))&(labels.EndTime>=pd.Timestamp(start,tz='UTC'))]
    local=set(part[part.Channel.isin(inputs)].ID)
    ids=set(part.ID)
    rows.append(dict(period=name,all_event_ids=len(ids),input_annotated_ids=len(local),outside_input_only_ids=len(ids-local)))
    for aid,g in part.groupby('ID'):
        events.append(dict(period=name,event_id=aid,category=g.Category.iloc[0],has_input_annotation=aid in local,channels=';'.join(sorted(g.Channel.unique()))))
annual=pd.DataFrame(rows)
prior=pd.read_csv(paths[3],dtype={'period':str})
for r in rows[1:]:
    p=prior[prior.period==r['period']].set_index('scope')
    assert r['all_event_ids']==p.loc['system','ids']
    assert r['input_annotated_ids']==p.loc['inputs_40_47','ids']
audit=pd.read_csv(paths[2])
local=audit[(audit.arm=='joint')&(audit.period=='validation')&audit.local]
assert set(local.fold)=={2001,2002} and set(local.seed)=={42,43,44}
assert not local.detected.any()
summary=local.groupby('fold').agg(event_seed_rows=('event_id','size'),unique_event_ids=('event_id','nunique'),min_peak_ratio=('peak_ratio','min'),max_peak_ratio=('peak_ratio','max'),min_nominal_fraction_ge_peak=('nominal_fraction_at_least_event_peak','min'),max_nominal_fraction_ge_peak=('nominal_fraction_at_least_event_peak','max')).reset_index()
OUT.mkdir(parents=True)
annual.to_csv(OUT/'training_period_coverage.csv',index=False)
pd.DataFrame(events).to_csv(OUT/'training_period_event_channels.csv',index=False)
summary.to_csv(OUT/'internal_score_summary.csv',index=False)
(OUT/'verification.json').write_text(json.dumps(dict(dependency_sha256=hashes,checks=['Annual coverage reproduced prior diversity audit','Pooled IDs deduplicated across 2000–2005','All six internal joint-model event-score groups included'],scope='No 2006 calibration or 2007–2013 predictions, scores or telemetry read; raw annotations filtered to periods before 2006 for coverage',limits='Event-score summary reuses original-hash verified prior audit, not new model inference; these development periods were already inspected'),indent=2)+'\n')
print(annual.to_string(index=False));print(summary.to_string(index=False))
