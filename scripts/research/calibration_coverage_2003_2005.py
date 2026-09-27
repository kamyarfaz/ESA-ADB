"""Inventory baseline calibration annotations only; never read assessment outputs."""
import json
from pathlib import Path
import pandas as pd
from esa_thesis import config
from esa_thesis.diagnose_channels import digest
from esa_thesis.masked_pilot import assessment_scopes

out=Path('docs/research/tables/calibration-coverage-2026-09-26')
out.mkdir(parents=True,exist_ok=False)
inputs=[config.ANNOTATIONS_FILE,config.ANOMALY_TYPES_FILE]
labels=pd.read_csv(config.ANNOTATIONS_FILE)
summary=[];events=[]
for year in (2003,2004,2005):
 root=Path('results_longrun/development')/('ae_baseline_2003_seed42' if year==2003 else 'ae_baseline_remaining_seed42')/str(year)
 path=root/'frozen_rule.json';inputs.append(path)
 frozen=json.loads(path.read_text());rule=frozen['rule']
 start=pd.Timestamp(f'{year}-01-01',tz='UTC');end=pd.Timestamp(f'{year}-04-01',tz='UTC')
 scopes=assessment_scopes(pd.date_range(start,end,freq='30s',inclusive='left').asi8)
 system=scopes['system_all_channels'];local=scopes['input_channels_40_47']
 assert len(system.by_id)==rule['val_num_true_events']
 summary.append(dict(year=year,system_events=len(system.by_id),input_events=len(local.by_id),outside_input_events=len(set(system.by_id)-set(local.by_id)),selected_epoch=frozen['epoch'],calibration_f05=rule['val_esa_f05'],system_TP=rule['val_TPe'],system_FP=rule['val_FPe'],system_FN=rule['val_FNe']))
 for eid,intervals in system.by_id.items():
  raw=labels[labels.ID==eid].copy()
  raw['StartTime']=pd.to_datetime(raw.StartTime,utc=True);raw['EndTime']=pd.to_datetime(raw.EndTime,utc=True)
  raw=raw[(raw.EndTime>=start)&(raw.StartTime<end)]
  channels=sorted(set(raw.Channel)&{f'channel_{i}' for i in range(40,48)})
  events.append(dict(year=year,event_id=eid,annotated_on_inputs=eid in local.by_id,input_channels=';'.join(channels),clipped_union_hours=sum(int(hi)-int(lo) for lo,hi in intervals)/3.6e12,starts_before_calibration=bool((raw.StartTime<start).any()),extends_beyond_calibration=bool((raw.EndTime>=end).any()),event_detection_status='not inferred from aggregate counts'))
pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
pd.DataFrame(events).to_csv(out/'events.csv',index=False)
(out/'provenance.json').write_text(json.dumps(dict(input_sha256={str(p):digest(p) for p in inputs},policy='Annotations and saved calibration rules only; no assessment access; aggregate TP counts cannot identify detected IDs'),indent=2)+'\n')
print(pd.DataFrame(summary).to_string(index=False))
print(pd.DataFrame(events).to_string(index=False))
