"""Read-only verification of the exported scientific workbook, including native panes."""
import json,math
from pathlib import Path
import openpyxl

out=Path(__file__).resolve().parents[1]
data=json.loads((out/'workbook-data.json').read_text())
book=openpyxl.load_workbook(out/'ESA_Mission1_Mission2_correlation_matrices.xlsx',data_only=False)
assert len(book.sheetnames)==9
checked=0
for m in data['matrices']:
    label={'pearson':'Levels','delta':'Changes','spearman':'Spearman'}[m['kind']]
    s=book[f'M{m["mission"]} {label}']
    assert s.freeze_panes=='B6',s.freeze_panes
    assert len(s.conditional_formatting)>=1
    for i,c in enumerate(m['channels']):
        assert s.cell(5,i+2).value==int(c)
        assert s.cell(i+6,1).value==int(c)
        for j,v in enumerate(m['values'][i]):
            value=s.cell(i+6,j+2).value
            if v is None:assert value=='n.a.'
            else:assert isinstance(value,(int,float)) and math.isclose(value,v,rel_tol=0,abs_tol=1e-12)
            checked+=1
pair=book['High correlation pairs'];assert pair.freeze_panes=='D6'
assert len(pair.tables)==1 and len(book['Channels'].tables)==1 and len(book['Groups and guide'].tables)==1
for i,p in enumerate(data['pairs'],6):
    assert pair.cell(i,1).value==p['mission']
    assert pair.cell(i,2).value==int(p['a'].split('_')[1])
    assert pair.cell(i,3).value==int(p['b'].split('_')[1])
    assert math.isclose(pair.cell(i,4).value,p['pearson'],abs_tol=1e-12)
    assert pair.cell(i,8).value==('Yes' if p['stable_high'] else 'No')
for s in book:
    assert s.sheet_view.showGridLines==False
    for row in s:
        for cell in row:assert cell.data_type!='e',(s.title,cell.coordinate,cell.value)
result={'verified':True,'sheets':len(book.sheetnames),'matrix_cells_verified':checked,
        'high_pair_rows_verified':len(data['pairs']),'native_freeze_panes_verified':True,
        'tables_and_conditional_formats_preserved':True,'error_cells':0}
(out/'workbook-verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
