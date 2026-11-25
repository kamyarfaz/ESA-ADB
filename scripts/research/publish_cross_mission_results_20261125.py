"""Publish all 36 verified model/normalization assessments; download and compare."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import wandb
ROOT=Path(__file__).resolve().parents[2]
T=ROOT/'docs/research/tables/cross-mission-2026-11-20-v2'
R=ROOT/'results_longrun/cross_mission_transfer_20261120_v2'
O=ROOT/'outputs/cross-mission-study-20261125'
PROJECT='kamyarfaz-politecnico-di-torino/esa-thesis-finetuning'
METRICS=['EW_F_0.50','EW_precision','EW_recall','nominal_false_alarm_rate','TPe','FPe','FNe','num_true_events']

def main():
    receipt=json.loads((T/'verification.json').read_text());assert receipt['verified']
    assert json.loads((T/'all-preparation-verification.json').read_text())['passed']
    df=pd.read_csv(T/'verified_results.csv',dtype={'period':str})
    joint=json.loads((R/'selection_complete.json').read_text());training=[];work=[]
    api=wandb.Api(timeout=60)
    for i,selection in enumerate(joint['selections'],1):
        job=selection['job'];rid=f'xm1120v2-{i:02d}-m{job["mission"]}-s{job["seed"]}'
        run=api.run(PROJECT+'/'+rid)
        assert run.state=='finished' and run.summary['completed'] is True
        for k,v in job.items():assert run.config[k]==v
        assert run.config['assessment_used_for_selection'] is False and run.summary['selected_epoch']==selection['rule']['epoch']
        training.append(dict(id=rid,url=run.url,state=run.state,epoch=run.summary['epoch'],selected_epoch=run.summary['selected_epoch']))
        work.append((job,'fresh_training',selection['rule'],dict(training_run_id=rid)))
    (T/'wandb-training-verification.json').write_text(json.dumps(dict(verified=True,runs=training),indent=2)+'\n')
    for source in joint['sources']:
        job={**source['job'],'mission':2,'arm':'source_context_transfer'}
        for condition in ('frozen_source_scaling','target_training_normalization'):
            work.append((job,condition,source['rule'],dict(source_model_sha256=source['model_sha256'],mapping=joint['mapping'],mapped_features=source['mapped_features'],target_threshold_tuning=False)))
    path=T/'wandb-evaluation.json';done=json.loads(path.read_text()) if path.exists() else {}
    for i,(job,condition,selection,extra) in enumerate(work,1):
        key=job['name']+'/'+condition;frame=df[(df.name==job['name'])&(df.condition==condition)].reset_index(drop=True)
        if key not in done:
            rid=f'xm1120v2-eval-{i:02d}-s{job["seed"]}'
            with wandb.init(entity=PROJECT.split('/')[0],project=PROJECT.split('/')[1],id=rid,resume='allow',name='verified-'+key,
                group='cross-mission-whole-system-20261120-v2-evaluation',job_type='verified-evaluation',mode='online',dir=str(O),
                config={**job,**extra,'condition':condition,'selected_epoch':selection['epoch'],'frozen_rule':selection['rule'],
                        'assessment_period':'2007–2013' if job['mission']==1 else '2001-10-01 to 2003-06-30',
                        'selection_scope':'system_all_channels' if condition=='fresh_training' else 'historical_source40_47',
                        'reused_benchmark':True,'assessment_used_for_selection':False,'independently_verified':True,
                        'verification_receipt':receipt['complete_receipt_sha256'],'score_pooling':'max over all model input reconstruction errors'},
                settings=wandb.Settings(disable_git=True,save_code=False)) as run:
                run.log({'assessment/annual_and_pooled':wandb.Table(dataframe=frame)})
                for _,row in frame[frame.period=='pooled'].iterrows():
                    prefix='assessment/'+row.event_definition+'/'+row.scope+'/'
                    for k in METRICS:run.summary[prefix+k]=float(row[k])
                run.summary['metric_rows']=len(frame)
                info=dict(id=rid,url=run.url,name=job['name'],condition=condition,metric_rows=len(frame))
            done[key]=info;path.write_text(json.dumps(done,indent=2)+'\n')
    checked=[];api=wandb.Api(timeout=60)
    for job,condition,selection,extra in work:
        record=done[job['name']+'/'+condition];run=api.run(PROJECT+'/'+record['id'])
        assert run.state=='finished' and run.config['independently_verified'] is True and run.config['assessment_used_for_selection'] is False
        for k,v in {**job,**extra,'condition':condition}.items():assert run.config[k]==v
        cloud=O/'cloud-check'/record['id'];run.file('wandb-summary.json').download(root=str(cloud),replace=True)
        summary=json.loads((cloud/'wandb-summary.json').read_text());table=summary['assessment/annual_and_pooled']
        run.file(table['path']).download(root=str(cloud),replace=True)
        downloaded=json.loads((cloud/table['path']).read_text());actual=pd.DataFrame(downloaded['data'],columns=downloaded['columns'])
        expected=df[(df.name==job['name'])&(df.condition==condition)].reset_index(drop=True)
        pd.testing.assert_frame_equal(actual,expected,check_dtype=False,rtol=1e-12,atol=1e-12)
        assert summary['metric_rows']==len(expected)
        for _,row in expected[expected.period=='pooled'].iterrows():
            prefix='assessment/'+row.event_definition+'/'+row.scope+'/'
            for k in METRICS:assert np.isclose(summary[prefix+k],row[k],rtol=1e-12,atol=1e-12)
        checked.append(dict(**record,state=run.state));print(record['id'],'downloaded table and summaries verified',flush=True)
    assert len(checked)==36 and sum(r['metric_rows'] for r in checked)==720
    (T/'wandb-cloud-verification.json').write_text(json.dumps(dict(verified=True,runs=36,downloaded_metric_rows=720,checked=checked),indent=2)+'\n')

if __name__=='__main__':main()
