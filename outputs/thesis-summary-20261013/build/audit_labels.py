"""Read-only source audit; export binary labels without changing research inputs."""
import csv,gzip,hashlib,json
from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np
ROOT=Path('/home/k_faz/projects/ESA-ADB')
OUT=ROOT/'outputs/thesis-summary-20261013'
DATA=ROOT/'data/preprocessed/multivariate/ESA-Mission1-semi-supervised'
EXPECTED={'train':'fe6812dd5d324fe2eb347a703894551f0a03f40253742a3cd35f42d02fdb80ad','test':'9990c46dba0101074a19b7e58382bee787ea9a2051230a5b1716d207d8315f6c'}
final=OUT/'deliverables/ESA_Mission1_binary_labels.csv.gz'
assert not final.exists()
tmp=OUT/'build/binary_labels.partial.csv.gz'
counts={s:Counter() for s in ['train','calibration','testing']}
sources=[]; previous=None; emitted=0
with gzip.open(tmp,'wt',encoding='ascii',newline='',compresslevel=6) as output:
 output.write('timestamp,split,is_anomaly,is_ignored\n')
 for kind in ('train','test'):
  path=DATA/f'84_months.{kind}.csv'; before=path.stat(); digest=hashlib.sha256(); total=0; excluded=0; batch=[]; stamps=[]; first=None; last=None; codes=Counter()
  def flush():
   global previous
   if not stamps:return
   ts=pd.DatetimeIndex(pd.to_datetime(stamps,utc=True)).as_unit('ns').asi8
   assert np.all(np.diff(ts)==30_000_000_000)
   if previous is not None: assert ts[0]-previous==30_000_000_000
   previous=int(ts[-1]);output.writelines(batch);batch.clear();stamps.clear()
  with path.open('rb') as f:
   raw=f.readline();digest.update(raw);header=raw.decode('ascii').strip().split(',')
   nlabels=sum(c.startswith('is_anomaly_') for c in header)
   assert nlabels==87
   labels=header[-nlabels:]; assert all(x.startswith('is_anomaly_channel_') for x in labels[:76])
   assert {int(x.rsplit('_',1)[1]) for x in labels[:76]}==set(range(1,77))
   assert labels[76:]==[x for x in labels if x.startswith('is_anomaly_telecommand_')]
   for raw in f:
    digest.update(raw);line=raw.rstrip(b'\r\n');ts=line.split(b',',1)[0].decode('ascii');total+=1
    if first is None:first=ts
    last=ts
    parts=line.rsplit(b',',nlabels);assert len(parts)==nlabels+1
    vals=parts[1:];assert set(vals).issubset({b'0',b'1',b'2',b'3'}),f'Invalid label at {ts}'
    # Do not turn communication gaps into anomaly positives.
    channel_vals=set(vals[:76]);codes.update(channel_vals)
    if not ('2000-01-01 00:00:00'<=ts<'2014-01-01 00:00:00') or (kind=='train' and ts>='2007-01-01 00:00:00'):
     excluded+=1;continue
    split='train' if ts<'2006-01-01 00:00:00' else 'calibration' if ts<'2007-01-01 00:00:00' else 'testing'
    y=int(bool(channel_vals & {b'1',b'2'}));ignored=int(b'3' in channel_vals and not y)
    counts[split].update(rows=1,positive=y,ignored=ignored,normal=int(not y and not ignored))
    batch.append(f'{ts},{split},{y},{ignored}\n');stamps.append(ts);emitted+=1
    if len(batch)==100000:flush()
   flush()
  assert digest.hexdigest()==EXPECTED[kind],f'Checksum mismatch: {path}'
  after=path.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
  sources.append(dict(path=str(path.relative_to(ROOT)),sha256=digest.hexdigest(),rows=total,first=first,last=last,excluded_boundary_rows=excluded,channels=76))
  print(kind,sources[-1],flush=True)
assert emitted==int((pd.Timestamp('2014-01-01')-pd.Timestamp('2000-01-01')).total_seconds()/30)
annotations=pd.read_csv(ROOT/'data/ESA-Mission1/labels.csv').merge(pd.read_csv(ROOT/'data/ESA-Mission1/anomaly_types.csv'),on='ID',validate='many_to_one')
assert annotations.Category.notna().all()
for split,lo,hi in [('train','2000-01-01','2006-01-01'),('calibration','2006-01-01','2007-01-01'),('testing','2007-01-01','2014-01-01')]:
 a=annotations[(pd.to_datetime(annotations.EndTime,utc=True)>=pd.Timestamp(lo,tz='UTC'))&(pd.to_datetime(annotations.StartTime,utc=True)<pd.Timestamp(hi,tz='UTC'))]
 counts[split]['event_ids']=int(a[a.Category.isin(['Anomaly','Rare Event'])].ID.nunique())
 counts[split]['annotated_channels']=int(a[a.Category.isin(['Anomaly','Rare Event'])].Channel.nunique())
report=dict(sources=sources,counts=counts,rows=emitted,label_definition='1 if any of 76 prepared channel labels equals 1 (Anomaly) or 2 (Rare Event); 0 otherwise',ignored_definition='1 if any channel has code 3 and no channel has a positive target label; exclude from pointwise nominal interpretation',scope='All 76 channels; telecommands excluded',alignment='Existing prepared 30-second timestamps; source resampling labels retained, not reconstructed from intervals',intervals='train [2000,2006), calibration [2006,2007), testing [2007,2014)',event_metric='Raw annotation IDs and exact intervals remain authoritative for esa-ew-id-duration-v1; binary labels are not a substitute',sha256=hashlib.sha256(tmp.read_bytes()).hexdigest())
tmp.rename(final)
(OUT/'build/data-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(counts,indent=2),flush=True)
