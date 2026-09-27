"""Pre2006 local annotation duration/support inventory; no later-period access."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from esa_thesis import config
from esa_thesis.diagnose_channels import digest
from esa_thesis.evaluation.esa import _merge
out=Path('docs/research/tables/local-event-shapes-2026-09-27');out.mkdir(parents=True,exist_ok=False)
files=[config.ANNOTATIONS_FILE,config.ANOMALY_TYPES_FILE];before={str(p):digest(p) for p in files}
df=pd.read_csv(files[0]).merge(pd.read_csv(files[1]),on='ID',validate='many_to_one')
for k in ['StartTime','EndTime']:df[k]=pd.to_datetime(df[k],utc=True).astype('int64')
lo=pd.Timestamp('2000-01-01',tz='UTC').value;hi=pd.Timestamp('2006-01-01',tz='UTC').value
features=[f'channel_{i}' for i in range(40,48)]
df=df[df.Channel.isin(features)&df.Category.isin(['Anomaly','Rare Event'])&(df.EndTime>=lo)&(df.StartTime<hi)].copy()
df.StartTime=df.StartTime.clip(lower=lo);df.EndTime=df.EndTime.clip(upper=hi)
rows=[]
for eid,g in df.groupby('ID'):
 intervals=_merge(list(zip(g.StartTime,g.EndTime)))
 for i,(start,end) in enumerate(intervals):
  v=g[(g.EndTime>=start)&(g.StartTime<=end)]
  # Peak positive-duration overlap of unique annotated channels. Point-only
  # overlaps are not assigned positive duration or counted as coordinated bursts.
  boundaries=sorted(set(v.StartTime)|set(v.EndTime));peak=0;wide_ns=0
  for a,b in zip(boundaries,boundaries[1:]):
   active=v[(v.StartTime<b)&(v.EndTime>a)].Channel.nunique()
   peak=max(peak,int(active))
   if active>4:wide_ns+=b-a
  rows.append(dict(event_id=eid,episode=i,year=pd.Timestamp(start).year,start=str(pd.Timestamp(start)),end=str(pd.Timestamp(end)),duration_seconds=(int(end)-int(start))/1e9,channels=';'.join(sorted(v.Channel.unique())),peak_simultaneous_channels=peak,seconds_over_four_channels=wide_ns/1e9,category=v.Category.iloc[0]))
a=pd.DataFrame(rows);a['shorter_than_synthetic_min']=a.duration_seconds<360;a['wider_than_synthetic_max']=a.peak_simultaneous_channels>4
a.to_csv(out/'episodes.csv',index=False)
summary=dict(event_ids=int(a.event_id.nunique()),episodes=len(a),short_episodes=int(a.shorter_than_synthetic_min.sum()),short_event_ids=int(a[a.shorter_than_synthetic_min].event_id.nunique()),wide_episodes=int(a.wider_than_synthetic_max.sum()),wide_event_ids=int(a[a.wider_than_synthetic_max].event_id.nunique()),short_and_wide_episodes=int((a.shorter_than_synthetic_min&a.wider_than_synthetic_max).sum()),short_and_wide_event_ids=sorted(a[a.shorter_than_synthetic_min&a.wider_than_synthetic_max].event_id.unique()),short_ids_excluding_id98=sorted(a[a.shorter_than_synthetic_min&(a.event_id!='id_98')].event_id.unique()))
assert before=={str(p):digest(p) for p in files}
(out/'summary.json').write_text(json.dumps(dict(**summary,input_sha256=before,scope='2000–2005 raw annotations on40–47; Anomaly+RareEvent; descriptive only',limitations='Union episodes and annotations are not independent physical events; multiple IDs may share a mechanism; broad all-channel annotation support may not mean equal signal change. Pre2006 includes previously inspected development periods. No physical cause or efficacy proven.'),indent=2)+'\n')
print(json.dumps(summary,indent=2))
