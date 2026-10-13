import csv, json, hashlib
from pathlib import Path
ROOT=Path('/home/k_faz/projects/ESA-ADB')
OUT=ROOT/'outputs/federico-meeting-20261013/build'
d=json.loads((ROOT/'outputs/federico-meeting-20260930/build/all_results_data.json').read_text())
old_count=len(d['records']); old_sources=len(d['sources']); old_history=len(d['histories']); old_protocols=len(d['protocols'])
def num(v):
    if v in ('',None): return None
    if v in ('True','False'): return v=='True'
    try:
        x=float(v)
        return int(x) if x.is_integer() else x
    except (ValueError,TypeError): return v
def rows(path):
    return [{k:num(v) for k,v in r.items()} for r in csv.DictReader((ROOT/path).open())]
def source(path):
    path=str(path)
    if path not in d['source_map']:
        sid=f'S{len(d["sources"])+1:03d}'
        d['source_map'][path]=sid
        p=ROOT/path
        d['sources'].append([sid,p.parent.name,p.name,hashlib.sha256(p.read_bytes()).hexdigest()])
    return d['source_map'][path]
def add_record(r,campaign,method,run,internal=True):
    year=int(r['fold']); deterministic=r['seed']=='deterministic'
    rec={k:None for k in d['records'][0]}
    rec.update(r)
    rec.update(id=f'R{len(d["records"])+1:03d}',campaign=campaign,method=method,run=run,channels='40–47',
        train_start='2000-01-01',train_end=f'{year}-01-01',cal_start=f'{year}-01-01',
        cal_end=f'{year}-07-01' if internal else f'{year}-04-01',
        assessment_start=f'{year}-07-01' if internal else f'{year}-04-01',assessment_end=f'{year+1}-01-01',
        boundary='[start,end)',period=f'{year} Jul–Dec' if internal else f'{year} Apr–Dec',
        epochs=0 if deterministic else 5,window_cap=0 if deterministic else 25000,
        data='Real prepared telemetry + real annotations',
        notes=('Internal development; not untouched validation. Primary: calibration_selected. Fixed epoch5 is secondary.' if internal else 'Three seeds complete. Same real assessment, five epochs, 25000 windows; no promotion.'))
    if campaign=='Score-change internal':rec['notes']+=' Reuses joint selected checkpoint; no new training. Model training budget shown.'
    if deterministic:rec['notes']='One deterministic run per fold. No neural training; training scale uses nominal telemetry. Silent on validation.'
    d['records'].append(rec)
for seed in [43,44]:
    run=f'short_no_roll_seed{seed}_v1'; path=f'results_longrun/development/{run}/results.csv'
    for r in rows(path):
        r.update(seed=seed,source=source(path))
        add_record(r,'No short rolls paired','Short coordinated generator' if r['model']=='short_coordinated' else 'Short generator without rolls',run,False)
configs=[('inner_selection_cuda_v2','Checkpoint policy internal','Joint AE'),('inner_lambda01_cuda_v1','Loss weight internal','Loss weight 0.1'),('inner_isolation_cuda_v1','Channel isolation internal','Isolated AE')]
for run,campaign,label in configs:
    path=f'results_longrun/development/{run}/results.csv'
    for r in rows(path):
        r['source']=source(path)
        add_record(r,campaign,label+' / '+r['policy'],run)
for run,campaign,arm,label in [('score_change_v1','Score-change internal','score_change','Score change'),('telemetry_reference_v1','Direct telemetry internal','telemetry_reference','Direct telemetry')]:
    path=f'results_longrun/development/{run}/results.csv'
    for r in rows(path):
        if r['arm']!=arm:continue
        r['source']=source(path)
        add_record(r,campaign,label,run)
new_runs=[f'short_no_roll_seed{x}_v1' for x in [43,44]]+[x[0] for x in configs]+['score_change_v1','telemetry_reference_v1']
for run in new_runs:
    folder=ROOT/'results_longrun/development'/run
    p=json.loads((folder/'protocol.json').read_text()); sid=source((folder/'protocol.json').relative_to(ROOT))
    is_neural=run not in ['score_change_v1','telemetry_reference_v1']
    d['protocols'].append(dict(run=run,campaign=next(r['campaign'] for r in d['records'] if r['run']==run),epochs=5 if is_neural else 0,seed=p.get('seed','42, 43, 44' if run!='telemetry_reference_v1' else 'deterministic'),batch_size=64 if is_neural else None,window_cap=25000 if is_neural else None,channels='40–47',selection='Calibration-selected primary; fixed epoch5 secondary' if run.startswith('inner') else 'Frozen calibration-selected rule',scoring='Training median/IQR; ±10 clipping; window-mean overlap then channel maximum' if is_neural else 'Fixed one-day history reference; see report',limits='Development data; no promotion; do not compare different periods as matched results.',source=sid))
    for path in sorted(folder.glob('**/history.csv')):
        sid=source(path.relative_to(ROOT))
        for r in rows(path.relative_to(ROOT)):d['histories'].append(dict(run=run,arm=' / '.join(path.relative_to(folder).parts[:-1]),source=sid,**r))
diagnostic_files={
 'Calibration audit':'docs/research/tables/calibration-selection-audit-2026-10-08/calibration_tradeoffs.csv',
 'Event audit':'docs/research/tables/inner-event-audit-2026-10-08/events.csv',
 'Channel interventions':'docs/research/tables/channel-level-coupling-2026-10-03/unchanged_outputs_summary.csv'}
d['diagnostics']={}
for name,path in diagnostic_files.items():
    sid=source(path)
    d['diagnostics'][name]=[dict(r,source=sid) for r in rows(path)]
for r in d['records']:
    assert r['metric_version']=='esa-ew-id-duration-v1'
    P,R=r['EW_precision'],r['EW_recall']
    expected=1.25*P*R/(.25*P+R) if P+R else 0
    assert abs(expected-r['EW_F_0.50'])<1e-9,r['id']
    assert r['TPe']+r['FNe']==r['num_true_events'],r['id']
    if r['nominal_seconds'] and r['TPe']+r['FPe']:
        expectedP=r['TPe']/(r['TPe']+r['FPe'])*(1-r['false_positive_seconds']/r['nominal_seconds'])
        assert abs(P-expectedP)<1e-9,r['id']
d['old_counts']={'records':old_count,'sources':old_sources,'histories':old_history,'protocols':old_protocols}
d['validation']={'metric_rows_checked':len(d['records']),'new_assessment_rows':len(d['records'])-old_count,'historical_records_preserved':old_count,'new_training_rows':len(d['histories'])-old_history}
(OUT/'october_data.json').write_text(json.dumps(d,indent=2,allow_nan=False))
print(d['validation']);print('Diagnostics:',{k:len(v) for k,v in d['diagnostics'].items()})
