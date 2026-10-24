"""Publish verified historical baselines, never pretend these are new training."""
import hashlib,json
from pathlib import Path
import pandas as pd
import wandb
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wandb-20261024'
ENTITY='kamyarfaz-politecnico-di-torino'
PROJECT='esa-thesis-finetuning'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 receipt_path=OUT/'baseline-import.json'
 done=json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
 sources=[
 ('short_no_roll','short_no_roll_full_20261014_seed42_v1','full_period',None),
 ('pretrain_finetune','combinations_full_20261021_seed42_v1','pretrain_finetune','pretrain_finetune'),
 ('expanded_short_no_roll','combinations_full_20261021_seed42_v1','expanded_short_no_roll','expanded_short_no_roll')]
 for method,source,sub,filter_method in sources:
  if method in done:continue
  root=ROOT/'results_longrun'/source
  receipts=json.loads((root/'complete.json').read_text())
  hashes=receipts.get('files',receipts.get('artifacts'))
  for name in ['all_results.csv',sub+'/history.csv',sub+'/frozen_rule.json']:
   assert sha(root/name)==hashes[name],name
  df=pd.read_csv(root/'all_results.csv',dtype={'period':str})
  if filter_method:df=df[df.method==filter_method]
  assert len(df)==16
  history=pd.read_csv(root/sub/'history.csv')
  rule=json.loads((root/sub/'frozen_rule.json').read_text())
  rid='baseline-'+method+'-'+sha(root/'complete.json')[:8]
  with wandb.init(entity=ENTITY,project=PROJECT,id=rid,resume='allow',name='original-'+method,
    group='historical-baselines',job_type='historical-import',mode='online',
    config=dict(method=method,historical_import=True,training_started=False,seed=42,
     train_years='2000–2005',calibration_years='2006',assessment_years='2007–2013',
     input_channels=list(range(40,48))+([14,21,29] if method=='expanded_short_no_roll' else []),
     source_receipt_sha256=sha(root/'complete.json'),selected_epoch=rule['epoch']),
    settings=wandb.Settings(disable_git=True,save_code=False)) as run:
   run.define_metric('epoch');run.define_metric('train/*',step_metric='epoch');run.define_metric('calibration/*',step_metric='epoch')
   for _,row in history.iterrows():
    metrics={'epoch':int(row.epoch),'train/loss':float(row.loss)}
    if pd.notna(row.calibration_f05):metrics['calibration/system_f05']=float(row.calibration_f05)
    run.log(metrics)
   run.log({'assessment/annual_and_pooled':wandb.Table(dataframe=df)})
   run.summary['calibration/selected_system_f05']=rule['rule']['val_esa_f05']
   for _,row in df[df.period=='2007-2013'].iterrows():
    scope='system' if row.scope=='system_all_channels' else 'channels40_47'
    for name,column in [('f05','EW_F_0.50'),('precision','EW_precision'),('event_tpr','EW_recall'),('time_fpr','nominal_false_alarm_rate'),('detected_events','TPe')]:
     run.summary['assessment/'+scope+'/'+name]=float(row[column])
   done[method]={'url':run.url,'run_id':rid}
  receipt_path.write_text(json.dumps(done,indent=2)+'\n')
 print(json.dumps(done,indent=2))
if __name__=='__main__':main()
