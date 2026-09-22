"""Collect immutable completed metrics for the supervisor workbook; no model selection."""
import json,hashlib
from pathlib import Path
import pandas as pd
root=Path(__file__).resolve().parents[2]
out=root/'research_workspace/channel_study_2026-09-22';out.mkdir(exist_ok=True)
rows=[];sources=[]
campaigns=[('Baseline AE + pseudo','40–47',8,'ae_baseline_2003_seed42',['2003']),
 ('Baseline AE + pseudo','40–47',8,'ae_baseline_remaining_seed42',['2004','2005']),
 ('Expanded AE + pseudo','40–47, 14, 21, 29',11,'ae_expanded_seed42',['2003','2004','2005']),
 ('Reconstruction-only AE','40–47',8,'ae_reconstruction_seed42',['2003','2004','2005']),
 ('Specialist AE + pseudo','14, 21, 29',3,'ae_specialist_seed42_verified',['2003','2004','2005'])]
for model,subset,n,campaign,folds in campaigns:
 for year in folds:
  run=root/'results_longrun/development'/campaign/year
  complete=(run/'results.json').is_file()
  result=json.loads((run/'results.json').read_text()) if complete else {}
  frozen=json.loads((run/'frozen_rule.json').read_text()) if complete else {}
  rule=frozen.get('rule',{})
  row={'year':int(year),'model':model,'channels':subset,'n_channels':n,
       'calibration_f05':result.get('calibration_f05'),'assessment_f05':result.get('EW_F_0.50'),
       'precision':result.get('EW_precision'),'recall':result.get('EW_recall'),
       'TPe':result.get('TPe'),'FPe':result.get('FPe'),'FNe':result.get('FNe'),
       'false_positive_seconds':result.get('false_positive_seconds'),'epoch':result.get('epoch'),
       'seed':42,'status':'Completed fold' if complete else 'Pending; not zero',
       'source_id':campaign+'/'+year,'threshold':rule.get('threshold'),'merge_gap':rule.get('merge_gap'),
       'min_dur':rule.get('min_dur'),'calibration_precision':rule.get('val_EW_precision'),
       'calibration_recall':rule.get('val_EW_recall'),'calibration_TPe':rule.get('val_TPe'),
       'calibration_FPe':rule.get('val_FPe'),'calibration_FNe':rule.get('val_FNe')}
  rows.append(row)
  if complete:
   for f in ['results.json','frozen_rule.json']:
    sources.append({'file':str((run/f).relative_to(root)),'sha256':hashlib.sha256((run/f).read_bytes()).hexdigest()})
# Forecasts are separate designs; retain them as evidence rather than compare as architecture-only.
for model,key,campaign in [('Persistence forecast','persistence','forecast_2003_seed42'),('MLP forecast','mlp','forecast_2003_seed42'),('Transformer forecast','transformer','forecast_2003_seed42'),('Residual Transformer forecast','transformer_residual','forecast_residual_2003_seed42')]:
 run=root/'results_longrun/development'/campaign/'2003'/key
 r=json.loads((run/'results.json').read_text());f=json.loads((run/'frozen_rule.json').read_text());rule=f['rule']
 rows.append({'year':2003,'model':model,'channels':'40–47','n_channels':8,'calibration_f05':r['calibration_f05'],
  'assessment_f05':r['EW_F_0.50'],'precision':r['EW_precision'],'recall':r['EW_recall'],
  'TPe':r['TPe'],'FPe':r['FPe'],'FNe':r['FNe'],'false_positive_seconds':r['false_positive_seconds'],
  'epoch':r['epoch'],'seed':42,'status':'Completed fold','source_id':campaign+'/2003/'+key,
  'threshold':rule['threshold'],'merge_gap':rule['merge_gap'],'min_dur':rule['min_dur'],
  'calibration_precision':rule.get('val_EW_precision'),'calibration_recall':rule.get('val_EW_recall'),
  'calibration_TPe':rule.get('val_TPe'),'calibration_FPe':rule.get('val_FPe'),'calibration_FNe':rule.get('val_FNe')})
 for name in ['results.json','frozen_rule.json']:
  sources.append({'file':str((run/name).relative_to(root)),'sha256':hashlib.sha256((run/name).read_bytes()).hexdigest()})
(out/'results_snapshot.json').write_text(json.dumps({'rows':rows,'sources':sources},indent=2))
pd.DataFrame(rows).to_csv(out/'results_snapshot.csv',index=False)
print('Captured',len(rows),'experiment rows;',sum(r['status']=='Completed fold' for r in rows),'completed')
