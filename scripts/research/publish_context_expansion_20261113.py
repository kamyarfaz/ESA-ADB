"""Publish all verified seeds to separate evaluation runs and download-check."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import wandb

ROOT=Path(__file__).resolve().parents[2]
TABLE=ROOT/'docs/research/tables/context-expansion-2026-11-04'
RUN=ROOT/'results_longrun/context_expansion_20261104_v1'
PROJECT='kamyarfaz-politecnico-di-torino/esa-thesis-finetuning'


def main():
    receipt=json.loads((TABLE/'verification.json').read_text());assert receipt['verified']
    df=pd.read_csv(TABLE/'verified_results.csv',dtype={'period':str})
    jobs=json.loads((RUN/'protocol.json').read_text())['jobs']
    selections=json.loads((RUN/'selection_complete.json').read_text())['selections']
    path=TABLE/'wandb-evaluation.json'
    done=json.loads(path.read_text()) if path.exists() else {}
    api=wandb.Api(timeout=60)
    training=[]
    for index,job in enumerate(jobs,1):
        rid=f'ctx1104-{index:02d}-s{job["seed"]}'
        run=api.run(PROJECT+'/'+rid)
        assert run.state=='finished'
        for k,v in job.items():assert run.config[k]==v
        assert run.config['assessment_used_for_selection'] is False
        assert run.summary['completed'] is True
        assert run.summary['selected_epoch']==selections[index-1]['rule']['epoch']
        training.append(dict(id=rid,url=run.url,state=run.state,epoch=run.summary['epoch'],selected_epoch=run.summary['selected_epoch']))
    (TABLE/'wandb-training-verification.json').write_text(json.dumps(dict(verified=True,runs=training),indent=2)+'\n')
    for index,(job,selection) in enumerate(zip(jobs,selections),1):
        frame=df[df.name==job['name']].reset_index(drop=True);assert len(frame)==16
        if job['name'] not in done:
            rid=f'ctx1104-eval-{index:02d}-s{job["seed"]}'
            with wandb.init(entity=PROJECT.split('/')[0],project=PROJECT.split('/')[1],id=rid,resume='allow',
                    name='verified-'+job['name'],group='context-expansion-20261104-v1-evaluation',job_type='verified-evaluation',mode='online',
                    dir=str(ROOT/'outputs/context-expansion-20261113'),
                    config={**job,'training_run_id':f'ctx1104-{index:02d}-s{job["seed"]}',
                            'selected_epoch':selection['rule']['epoch'],'frozen_rule':selection['rule']['rule'],
                            'train_years':'2000-2005','calibration_years':'2006','assessment_years':'2007-2013',
                            'selection_scope':'input_channels_40_47','reused_benchmark':True,'assessment_used_for_selection':False,
                            'score_pooling':'max over all input-channel reconstruction errors',
                            'independently_verified':True,'verification_receipt':receipt['complete_receipt_sha256']},
                    settings=wandb.Settings(disable_git=True,save_code=False)) as run:
                run.log({'assessment/annual_and_pooled':wandb.Table(dataframe=frame)})
                for _,row in frame[frame.period=='2007-2013'].iterrows():
                    scope='system' if row.scope=='system_all_channels' else 'channels40_47'
                    for k in ['EW_F_0.50','EW_precision','EW_recall','nominal_false_alarm_rate','TPe','FPe','FNe','num_true_events']:
                        run.summary[f'assessment/{scope}/{k}']=float(row[k])
                run.summary['metric_rows']=16
                info=dict(id=rid,url=run.url,training_id=f'ctx1104-{index:02d}-s{job["seed"]}')
            done[job['name']]=info
            path.write_text(json.dumps(done,indent=2)+'\n')
    checked=[]
    api=wandb.Api(timeout=60)
    for job in jobs:
        record=done[job['name']];run=api.run(PROJECT+'/'+record['id']);assert run.state=='finished'
        assert run.config['assessment_used_for_selection'] is False and run.config['independently_verified'] is True
        for k,v in job.items():assert run.config[k]==v
        cloud=ROOT/'outputs/context-expansion-20261113/cloud-check'/record['id']
        run.file('wandb-summary.json').download(root=str(cloud),replace=True)
        summary=json.loads((cloud/'wandb-summary.json').read_text())
        table=summary['assessment/annual_and_pooled']
        run.file(table['path']).download(root=str(cloud),replace=True)
        downloaded=json.loads((cloud/table['path']).read_text())
        actual=pd.DataFrame(downloaded['data'],columns=downloaded['columns'])
        expected=df[df.name==job['name']].reset_index(drop=True)
        pd.testing.assert_frame_equal(actual,expected,check_dtype=False,rtol=1e-12,atol=1e-12)
        assert summary['metric_rows']==16
        for _,row in expected[expected.period=='2007-2013'].iterrows():
            scope='system' if row.scope=='system_all_channels' else 'channels40_47'
            for k in ['EW_F_0.50','EW_precision','EW_recall','nominal_false_alarm_rate','TPe','FPe','FNe','num_true_events']:
                assert np.isclose(summary[f'assessment/{scope}/{k}'],row[k],rtol=1e-12,atol=1e-12)
        checked.append(dict(name=job['name'],id=record['id'],rows=len(actual),state=run.state))
        print(job['name'],'downloaded cloud table and summary verified',flush=True)
    (TABLE/'wandb-cloud-verification.json').write_text(json.dumps(dict(verified=True,runs=18,downloaded_metric_rows=288,checked=checked),indent=2)+'\n')


if __name__=='__main__':main()
