from copy import copy
import openpyxl,json,hashlib
from pathlib import Path
out=Path('outputs/cross-mission-study-20261125');old=openpyxl.load_workbook(out/'workbook-before.xlsx');new=openpyxl.load_workbook(out/'deliverables/ESA_thesis_results_full_and_historical.xlsx')
issues=[]
for s in old:
 t=new[s.title]
 for row in s:
  for c in row:
   x=t[c.coordinate]
   for k in ['value','number_format','alignment','fill','font','border','protection']:
    if copy(getattr(c,k))!=copy(getattr(x,k)):issues.append((s.title,c.coordinate,k))
 for k in ['freeze_panes','merged_cells']:
  if str(getattr(s,k))!=str(getattr(t,k)):issues.append((s.title,k))
 assert s.auto_filter.ref==t.auto_filter.ref
 assert list(s.tables)==list(t.tables)
 for n in s.tables:
  assert s.tables[n].ref==t.tables[n].ref
 assert len(s._charts)==len(t._charts) and len(s.data_validations.dataValidation)==len(t.data_validations.dataValidation)
print('historical differences',len(issues),issues[:20]);assert not issues
rows=json.loads((out/'workbook-data.json').read_text())['rows']
for name,subset in [('Cross-mission fresh',[r for r in rows if r[3]=='fresh_training']),('M1 to M2 transfer',[r for r in rows if r[3]!='fresh_training'])]:
 t=new[name]
 for i,r in enumerate(subset,6):
  for j,v in enumerate(r,1):
   actual=t.cell(i,j).value
   if isinstance(v,float):assert abs(actual-v)<1e-13
   else:assert actual==v,(name,i,j,actual,v)
 assert t.freeze_panes=='C6'
assert old.sheetnames==new.sheetnames[:len(old.sheetnames)]
assert len(new.sheetnames)==len(old.sheetnames)+2
receipt=dict(verified=True,historical_sheets=3,new_rows=len(rows),scope_metric_rows=len(rows)*2,all_historical_cell_values_formulas_and_styles_unchanged=True,new_rows_match_verified_metrics=True,sha256=hashlib.sha256((out/'deliverables/ESA_thesis_results_full_and_historical.xlsx').read_bytes()).hexdigest())
Path('docs/research/tables/cross-mission-2026-11-20-v2/workbook-verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(receipt)
