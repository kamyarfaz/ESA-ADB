from copy import copy
import openpyxl,json,hashlib
from pathlib import Path
out=Path('outputs/mission2-study-20261115');old=openpyxl.load_workbook(out/'workbook-before.xlsx');new=openpyxl.load_workbook(out/'deliverables/ESA_thesis_results_full_and_historical.xlsx')
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
rows=json.loads((out/'workbook-data.json').read_text())['rows'];t=new['Mission 2']
for i,r in enumerate(rows,6):
 for j,v in enumerate(r,1):
  actual=t.cell(i,j).value
  if isinstance(v,float):assert abs(actual-v)<1e-13
  else:assert actual==v,(i,j,actual,v)
receipt=dict(verified=True,historical_sheets=2,mission2_rows=len(rows),scope_metric_rows=len(rows)*2,all_historical_cell_values_formulas_and_styles_unchanged=True,new_rows_match_verified_metrics=True,sha256=hashlib.sha256((out/'deliverables/ESA_thesis_results_full_and_historical.xlsx').read_bytes()).hexdigest())
Path('docs/research/tables/mission2-three-technique-2026-11-14/workbook-verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(receipt)
