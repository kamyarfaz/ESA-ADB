import json,csv,hashlib,statistics
from pathlib import Path
from collections import OrderedDict
ROOT=Path('/home/k_faz/projects/ESA-ADB');OUT=ROOT/'outputs/thesis-summary-20261013/build'
d=json.loads((ROOT/'outputs/federico-meeting-20261013/build/october_data.json').read_text())
source_map={v:k for k,v in d['source_map'].items()}
groups=OrderedDict()
# One row per technique and assessment period, averaged over all available seeds.
for r in d['records']:
 r['channels']=r['channels'].replace('40, 41, 42, 43, 44, 45, 46, 47','40–47')
 if r['scope']!='system_all_channels':continue
 if r['campaign'] in ('Full-period baseline','Timing comparison') and r['period']!='2007–2013 pooled':continue
 if r['campaign'] in ('Loss weight internal','Channel isolation internal') and 'fixed_epoch5' in r['method']:continue
 key=tuple(str(r.get(k,'')) for k in ['campaign','method','period','channels','epochs','window_cap','train_start','train_end','cal_start','cal_end','assessment_start','assessment_end'])
 groups.setdefault(key,[]).append(r)
# Do not duplicate the exactly reused short-coordinated control.
for k in list(groups):
 if k[0]=='No short rolls paired' and k[1]=='Short coordinated generator':
  other=[v for q,v in groups.items() if q[0]=='Augmentation paired' and q[1]==k[1] and q[2:]==k[2:]]
  assert len(other)==1
  assert all(abs(a['EW_F_0.50']-b['EW_F_0.50'])<1e-12 and a['seed']==b['seed'] for a,b in zip(sorted(groups[k],key=lambda x:x['seed']),sorted(other[0],key=lambda x:x['seed'])))
  del groups[k]
rename={'Transformer AE baseline':'Baseline Transformer AE','Transformer AE + original augmentation':'Baseline Transformer AE','Short generator without rolls':'Remove short rolls','Short coordinated generator':'Short synthetic anomalies','Original generator':'Original synthetic anomalies','Joint AE / calibration_selected':'Joint AE control','Joint AE / fixed_epoch5':'Fixed fifth epoch','Loss weight 0.1 / calibration_selected':'Synthetic loss weight 0.1','Isolated AE / calibration_selected':'Channel-isolated AE','Observation-age context':'Add observation-age context'}
comparison={'Full-period baseline':'Reference, 40 epochs','Timing comparison':'Timing, 5 epochs','Historical 40-epoch development':'Development, 40 epochs','OR combination':'Development, 40 epochs','Augmentation paired':'Synthetic anomalies, 5 epochs','No short rolls paired':'Synthetic anomalies, 5 epochs','Masking paired':'Masking, 5 epochs','Masked overlap paired':'Masked scoring','Threshold policy':'Threshold selection','Forecasting':'Forecasting','Alarm confirmation':'Forecast confirmation','Checkpoint policy internal':'Internal, 5 epochs','Loss weight internal':'Internal, 5 epochs','Channel isolation internal':'Internal, 5 epochs','Score-change internal':'Internal, frozen model','Direct telemetry internal':'Internal, statistical','Historical scoring pair':'Historical scoring'}
priority={'Full-period baseline':0,'No short rolls paired':2,'Augmentation paired':1,'Checkpoint policy internal':3,'Loss weight internal':4,'Channel isolation internal':5,'Score-change internal':6,'Direct telemetry internal':7}
def date_span(start,end):
 if start is None or end is None:return 'Historical split'
 from datetime import date,timedelta
 e=date.fromisoformat(end)-timedelta(days=1);s=date.fromisoformat(start)
 if s.month==1 and s.day==1 and e.month==12 and e.day==31:return str(s.year) if s.year==e.year else f'{s.year}–{e.year}'
 return f'{s:%Y-%m}–{e:%Y-%m}'
rows=[];meta=[]
for key,rs in sorted(groups.items(),key=lambda kv:(priority.get(kv[0][0],10),kv[0][0],kv[0][2],kv[0][1])):
 r=rs[0];values=[float(x['EW_F_0.50']) for x in rs]
 assert len({x['seed'] for x in rs})==len(rs),(key,len(rs))
 sources=sorted({source_map[x['source']] for x in rs})
 for x in rs:
  p=ROOT/source_map[x['source']]
  source_rows=list(csv.DictReader(p.open()))
  assert any(abs(float(z.get('EW_F_0.50',z.get('test_EW_F_0.50','nan')))-x['EW_F_0.50'])<1e-12 and abs(float(z.get('EW_precision',z.get('test_EW_precision','nan')))-x['EW_precision'])<1e-12 and abs(float(z.get('EW_recall',z.get('test_EW_recall','nan')))-x['EW_recall'])<1e-12 for z in source_rows),(x['id'],str(p))
 channel=r['channels'].replace('40, 41, 42, 43, 44, 45, 46, 47','40–47')
 rows.append([rename.get(r['method'],r['method']),comparison[r['campaign']],channel,date_span(r['train_start'],r['train_end']),date_span(r['cal_start'],r['cal_end']),r['period'],len(rs),statistics.mean(x['EW_precision'] for x in rs),statistics.mean(x['EW_recall'] for x in rs),statistics.mean(values),statistics.stdev(values) if len(rs)>1 else None,statistics.mean(x['FPe'] for x in rs)])
 meta.append(dict(excel_row=len(rows)+5,sources=sources,record_ids=[x['id'] for x in rs],epochs=r['epochs'],window_cap=r['window_cap'],source_hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}))
(OUT/'summary-data.json').write_text(json.dumps(dict(rows=rows,provenance=meta),indent=2))
print('Summary rows:',len(rows))
