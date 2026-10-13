import csv,gzip,json,hashlib
from collections import Counter
from pathlib import Path
import openpyxl
ROOT=Path('/home/k_faz/projects/ESA-ADB/outputs/thesis-summary-20261013');report=json.loads((ROOT/'build/data-audit.json').read_text());data=json.loads((ROOT/'build/summary-data.json').read_text())
w=openpyxl.load_workbook(ROOT/'deliverables/ESA_thesis_results.xlsx',data_only=False)
assert w.sheetnames==['Thesis results'];s=w.active
for row,expected in enumerate(data['rows'],6):
 for col,v in enumerate(expected,1):
  got=s.cell(row,col).value
  if isinstance(v,(int,float)):assert abs(got-v)<1e-12,(row,col,got,v)
  else:assert got==v,(row,col,got,v)
 assert s.cell(row,1).comment is not None
assert len(s.tables)==1
from zipfile import ZipFile
with ZipFile(ROOT/'deliverables/ESA_thesis_results.xlsx') as z:assert all(b'federico' not in z.read(n).lower() for n in z.namelist())
# Independent streaming round-trip of every label row; strict binary domain,
# split order, timestamp order, boundaries and counts.
counts={k:Counter() for k in report['counts']};prev='';first={};last={}
with gzip.open(ROOT/'deliverables/ESA_Mission1_binary_labels.csv.gz','rt') as f:
 assert f.readline().strip()=='timestamp,split,is_anomaly,is_ignored'
 for line in f:
  ts,split,y,ignored=line.strip().split(',');assert y in ('0','1') and ignored in ('0','1');assert not(y=='1' and ignored=='1');assert ts>prev;prev=ts
  expected='train' if ts<'2006-01-01' else 'calibration' if ts<'2007-01-01' else 'testing';assert split==expected
  first.setdefault(split,ts);last[split]=ts
  counts[split].update(rows=1,positive=int(y),ignored=int(ignored),normal=int(y=='0' and ignored=='0'))
for split,v in counts.items():
 for k,n in v.items():assert n==report['counts'][split][k]
assert first=={'train':'2000-01-01 00:00:00','calibration':'2006-01-01 00:00:00','testing':'2007-01-01 00:00:00'}
assert last=={'train':'2005-12-31 23:59:30','calibration':'2006-12-31 23:59:30','testing':'2013-12-31 23:59:30'}
assert hashlib.sha256((ROOT/'deliverables/ESA_Mission1_binary_labels.csv.gz').read_bytes()).hexdigest()==report['sha256']
validation=dict(workbook_rows=len(data['rows']),workbook_sheets=1,all_workbook_values_verified=True,source_metrics_verified=True,binary_rows_verified=sum(x['rows'] for x in counts.values()),counts=counts)
(ROOT/'build/final-validation.json').write_text(json.dumps(validation,indent=2));print(json.dumps(validation,indent=2))
side=ROOT/'deliverables/ESA_thesis_results.xlsx.inspect.ndjson'
if side.exists():side.rename(ROOT/'build'/side.name)
