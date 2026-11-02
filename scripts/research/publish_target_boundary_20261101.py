"""Publish only verified evaluation; no training, telemetry or weight upload."""
import json
import pandas as pd
import wandb
from scripts.research.target_boundary_common import ROOT, OUTPUT, TABLE


def main():
    assert json.loads((TABLE / 'verification.json').read_text())['verified']
    df = pd.read_csv(TABLE / 'verified_results.csv', dtype={'period':str})
    frozen = json.loads((OUTPUT / 'selection_complete.json').read_text())
    receipt = TABLE / 'wandb-evaluation.json'
    done = json.loads(receipt.read_text()) if receipt.exists() else {}
    for method in df.method.drop_duplicates():
        if method in done:
            continue
        frame = df[df.method == method]
        rid = 'targetboundary20261101-' + method
        with wandb.init(entity='kamyarfaz-politecnico-di-torino', project='esa-thesis-finetuning',
                        id=rid, resume='allow', name='verified-target-boundary-' + method,
                        group='target-boundary-20261101', job_type='verified-evaluation', mode='online',
                        config=dict(method=method, train_years='2000-2005', calibration_years='2006',
                                    assessment_years='2007-2013', seed=42, reused_benchmark=True,
                                    assessment_used_for_selection=False, no_new_training=True,
                                    frozen_arms={a:s for a,s in frozen.items() if s['method'] == method}),
                        settings=wandb.Settings(disable_git=True, save_code=False)) as run:
            run.log({'assessment/annual_and_pooled':wandb.Table(dataframe=frame)})
            for _, row in frame[frame.period == '2007-2013'].iterrows():
                scope = 'system' if row.scope == 'system_all_channels' else 'channels40_47'
                for key, col in [('f05','EW_F_0.50'),('precision','EW_precision'),('event_tpr','EW_recall'),
                                 ('time_fpr','nominal_false_alarm_rate'),('detected_events','TPe'),('total_events','num_true_events')]:
                    run.summary[f'assessment/{row.arm}/{row.variant}/{scope}/{key}'] = float(row[col])
            done[method] = dict(id=rid, url=run.url)
        receipt.write_text(json.dumps(done, indent=2) + '\n')
    api = wandb.Api(timeout=30)
    for method, details in done.items():
        run = api.run('kamyarfaz-politecnico-di-torino/esa-thesis-finetuning/' + details['id'])
        assert run.state == 'finished'
        cloud = ROOT / 'outputs/target-boundary-20261101/cloud-check' / method
        run.file('wandb-summary.json').download(root=str(cloud), replace=True)
        summary = json.loads((cloud / 'wandb-summary.json').read_text())
        table = summary['assessment/annual_and_pooled']
        run.file(table['path']).download(root=str(cloud), replace=True)
        downloaded = json.loads((cloud / table['path']).read_text())
        actual = pd.DataFrame(downloaded['data'], columns=downloaded['columns'])
        expected = df[df.method == method].reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False, rtol=1e-12, atol=1e-12)
        for _, row in expected[expected.period == '2007-2013'].iterrows():
            scope = 'system' if row.scope == 'system_all_channels' else 'channels40_47'
            for key, col in [('f05','EW_F_0.50'),('event_tpr','EW_recall'),('time_fpr','nominal_false_alarm_rate')]:
                assert abs(summary[f'assessment/{row.arm}/{row.variant}/{scope}/{key}'] - row[col]) < 1e-12
    (TABLE / 'wandb-cloud-verification.json').write_text(json.dumps(dict(verified=True, runs=len(done),
        downloaded_metric_rows=len(df), checks=['Finished run states', 'All downloaded table values',
                                                'Pooled summary F0.5, eventTPR and timeFPR']), indent=2) + '\n')
    print('Published and cloud-verified three evaluation runs and336metric rows')


if __name__ == '__main__':
    main()
