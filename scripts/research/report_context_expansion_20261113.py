"""Summarize all frozen outcomes; never choose seeds or rules on assessment."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
TABLE=ROOT/'docs/research/tables/context-expansion-2026-11-04'
RUN=ROOT/'results_longrun/context_expansion_20261104_v1'
OUTPUT=ROOT/'outputs/context-expansion-20261113'
METHODS=['short_no_roll','pretrain_finetune','expanded_short_no_roll']
NAMES=dict(zip(METHODS,['Short-no-roll','Masked pretraining + short-no-roll','Short-no-roll retaining 14/21/29']))
SCOPES=['input_channels_40_47','system_all_channels']
METRICS=['EW_F_0.50','EW_precision','EW_recall','nominal_false_alarm_rate','TPe','FPe','FNe','false_positive_seconds']


def main():
    verification=json.loads((TABLE/'verification.json').read_text());assert verification['verified']
    df=pd.read_csv(TABLE/'verified_results.csv',dtype={'period':str})
    ev=pd.read_csv(TABLE/'verified_event_ids.csv',dtype={'period':str})
    cal=pd.read_csv(TABLE/'verified_calibration.csv')
    plan=json.loads((RUN/'protocol.json').read_text())
    paired=[]
    for (method,seed,period,scope),g in df.groupby(['method','seed','period','scope'],sort=False):
        c=g[g.arm=='control'].iloc[0];x=g[g.arm=='context'].iloc[0]
        row=dict(method=method,seed=int(seed),period=period,scope=scope)
        for k in METRICS:row.update({k+'_control':c[k],k+'_context':x[k],k+'_delta':x[k]-c[k]})
        e=ev[(ev.method==method)&(ev.seed==seed)&(ev.period==period)&(ev.scope==scope)]
        a=set(e[(e.arm=='control')&e.detected].event_id);b=set(e[(e.arm=='context')&e.detected].event_id)
        row['gained_ids']=';'.join(sorted(b-a));row['lost_ids']=';'.join(sorted(a-b))
        paired.append(row)
    pairs=pd.DataFrame(paired);pairs.to_csv(TABLE/'paired_changes.csv',index=False)
    summaries=[]
    for (method,period,scope),g in pairs.groupby(['method','period','scope'],sort=False):
        for k in METRICS:
            r=dict(method=method,period=period,scope=scope,metric=k,seeds=3)
            for variant in ['control','context','delta']:
                v=g[k+'_'+variant]
                r.update({variant+'_mean':v.mean(),variant+'_sd':v.std(ddof=1),variant+'_min':v.min(),variant+'_max':v.max()})
            summaries.append(r)
    summary=pd.DataFrame(summaries);summary.to_csv(TABLE/'seed_summary.csv',index=False)
    pooled=df[df.period=='2007-2013'];pp=pairs[pairs.period=='2007-2013']
    # Historical references remain descriptive: their objective was system scope.
    old=[]
    for method in METHODS:
        source=ROOT/('results_longrun/short_no_roll_full_20261014_seed42_v1/all_results.csv' if method=='short_no_roll' else 'results_longrun/combinations_full_20261021_seed42_v1/all_results.csv')
        historic=pd.read_csv(source,dtype={'period':str})
        if method!='short_no_roll':historic=historic[historic.method==method]
        for scope in SCOPES:
            h=historic[(historic.period=='2007-2013')&(historic.scope==scope)].iloc[0]
            for _,r in pooled[(pooled.method==method)&(pooled.scope==scope)].iterrows():
                old.append(dict(method=method,seed=int(r.seed),arm=r.arm,scope=scope,
                                old_seed=42,old_system_selected=True,new_local_selected=True,
                                old_f05=h['EW_F_0.50'],new_f05=r['EW_F_0.50'],delta_f05=r['EW_F_0.50']-h['EW_F_0.50'],
                                old_tpr=h.EW_recall,new_tpr=r.EW_recall,old_fpr=h.nominal_false_alarm_rate,new_fpr=r.nominal_false_alarm_rate))
    pd.DataFrame(old).to_csv(TABLE/'historical_comparison.csv',index=False)
    lines=['# Matched input expansion across three techniques — 13 November 2026','',
      'Adding training-selected channels 33 and 49 increased mean pooled target 40–47 F0.5 in all three families. '
      'The short-no-roll comparison improved in all three seeds. Masked pretraining improved in two seeds and worsened in seed 42. '
      'The family retaining 14/21/29 improved over poor fresh controls, but remained unstable and had substantial nominal alarm duration. '
      'These are results on the reused 2007–2013 benchmark, not independent confirmation or robust achievement of the thesis target.','',
      'All 18 fresh models completed 40 epochs. All selections jointly froze at 11:43:46 UTC on 13 November; assessment finished at 12:19:39 UTC. '
      'There were no failed or omitted training runs. A 9 November host reboot interrupted run 6; the unchanged service resumed its checkpoint. '
      'The reboot cause is unknown. Performance failures are retained below.','',
      '## Design and rationale','',
      'Each family has a fresh matched control and an arm adding 33/49, paired by seeds 42,43,44. '
      'The first two families use 40–47 in the controls (8→10 inputs). The third already retains 14/21/29 (11→13 inputs). '
      'Both arms train 2000–2005 and select checkpoint and alarm rule using corrected 40–47 event F0.5 on 2006 only, with a 1% calibration alarm-fraction cap. '
      'No assessment scores choose channels, thresholds, checkpoints, seeds or models.','',
      'Channels 33 and 49 were selected solely from 2000–2005 training correlations. They had the strongest worst-quarter absolute Spearman '
      'correlation in the prespecified pools for anchors 40 and 47: 0.6673 and 0.5515 over 24 quarters; pooled values 0.99795 and 0.98766. '
      'Their first-difference Pearson correlations were near zero (0.00350 and −0.00260). The rationale was stable shared operating context, '
      'not demonstrated causal usefulness. High level correlations may reflect slow trends. All 225 candidate/quarter coefficients were replayed before launch.','',
      'The baseline is already a Transformer autoencoder. Architecture, nominal-only scaler fitting, training windows and optimizer were fixed. '
      'All runs receive 40 epochs; masked pretraining allocates 20 masked + 20 short-no-roll. Each epoch uses 349,461 windows and 5,461 batches. '
      'The 18 runs require 3,931,920 optimizer steps plus 120 calibration checkpoint evaluations and 18 assessment passes. '
      'The observed execution spans approximately 5 days 23 hours from 7 November 13:06 UTC through assessment completion, including input checks and the reboot. '
      'Adding inputs increases compute even with an equal epoch budget.','',
      'Reconstruction errors are averaged within 256-sample windows, overlapped at stride 32, and maximized over every input channel. '
      'New inputs therefore also become score sources. This study tests input-and-score expansion, not isolated contextual conditioning.','',
      '## Three-seed pooled results','',
      'F0.5 uses corrected event-ID/duration evaluation. Event TPR is detected IDs divided by annotated IDs; time FPR is false-alarm duration '
      'divided by nominal duration. Means and sample standard deviations below summarize three seeds, not confidence intervals. '
      'Pooled metrics are recomputed continuously over all seven years, not averaged from annual metrics.','',
      '| Technique | Scope | Control F0.5 mean ± SD | Added 33/49 mean ± SD | Paired change mean ± SD |','|---|---|---:|---:|---:|']
    for method in METHODS:
        for scope in SCOPES:
            r=summary[(summary.method==method)&(summary.period=='2007-2013')&(summary.scope==scope)&(summary.metric=='EW_F_0.50')].iloc[0]
            lines.append(f'| {NAMES[method]} | {scope} | {r.control_mean:.6f} ± {r.control_sd:.6f} | {r.context_mean:.6f} ± {r.context_sd:.6f} | {r.delta_mean:+.6f} ± {r.delta_sd:.6f} |')
    lines+=['','## Every paired seed, including adverse outcomes','',
            '| Technique | Seed | Scope | F0.5 control → added | Δ F0.5 | Detected control → added | Time FPR (%) control → added |','|---|---:|---|---:|---:|---:|---:|']
    for method in METHODS:
        for _,r in pp[pp.method==method].iterrows():
            lines.append(f'| {NAMES[method]} | {r.seed} | {r.scope} | {r["EW_F_0.50_control"]:.6f} → {r["EW_F_0.50_context"]:.6f} | {r["EW_F_0.50_delta"]:+.6f} | {int(r.TPe_control)} → {int(r.TPe_context)} | {100*r.nominal_false_alarm_rate_control:.5f} → {100*r.nominal_false_alarm_rate_context:.5f} |')
    lines+=['','For short-no-roll, added inputs improve target F0.5 in all seeds, but seed 42/43 also increase nominal alarm duration. '
      'For masked pretraining, seed 42 gains 11 target detections (50→61) while false events rise 2→8, lowering F0.5; its timeFPR also rises. '
      'For the third family, fresh controls produce 141–356false events and target F0.5 of 0.158–0.323. Added inputs reduce false-event counts, '
      'but seeds 43/44 spend approximately 5.9% of nominal time in false alarms. Fewer long false alarms can yield high event F0.5 despite poor alarm duration. '
      'The calibration cap does not guarantee the same cap on assessment. The third family should not be presented as a reliable improvement based only on F0.5.','',
      '## Comparison with unchanged historical runs','',
      'Historical seed 42 models used system-scope calibration selection; this study uses target 40–47 selection and fresh training. '
      'Comparisons below are descriptive and cannot isolate the channel effect. The fresh paired controls above are the primary comparison. '
      'No historical row or selected checkpoint is overwritten.','',
      '| Technique | Historical target F0.5 | Fresh control target mean | Added 33/49 target mean | Historical system F0.5 |','|---|---:|---:|---:|---:|']
    olddf=pd.DataFrame(old)
    for method in METHODS:
        h=olddf[olddf.method==method]
        r=summary[(summary.method==method)&(summary.period=='2007-2013')&(summary.scope==SCOPES[0])&(summary.metric=='EW_F_0.50')].iloc[0]
        lines.append(f'| {NAMES[method]} | {h[h.scope==SCOPES[0]].old_f05.iloc[0]:.6f} | {r.control_mean:.6f} | {r.context_mean:.6f} | {h[h.scope==SCOPES[1]].old_f05.iloc[0]:.6f} |')
    lines+=['','[Every new seed versus the historical reference](tables/context-expansion-2026-11-04/historical_comparison.csv).','',
      '## Frozen calibration selections','',
      '| Technique | Seed | Arm | Epoch | Target calibration F0.5 | Threshold | Gap | Minimum duration |','|---|---:|---|---:|---:|---:|---:|---:|']
    for _,r in cal[cal.selected].iterrows():
        lines.append(f'| {NAMES[r.method]} | {r.seed} | {r.arm} | {r.epoch} | {r.val_esa_f05:.9f} | {r.threshold:.9g} | {r.merge_gap} | {r.min_dur} |')
    lines+=['','Only six target event IDs occur in 2006 calibration. Near-ceiling calibration results therefore provide weak evidence about future behavior. '
      'All 120 checkpoint grids and every rejected candidate are retained. No fallback was substituted after seeing assessment.','',
      '## Event changes and annual evidence','',
      '[All 288 annual/pooled metric rows](tables/context-expansion-2026-11-04/verified_results.csv), '
      '[all detected/missed IDs](tables/context-expansion-2026-11-04/verified_event_ids.csv), '
      '[all 144 paired annual/pooled comparisons](tables/context-expansion-2026-11-04/paired_changes.csv), and '
      '[three-seed summaries](tables/context-expansion-2026-11-04/seed_summary.csv).','']
    for _,r in pp.iterrows():
        lines.append(f'- {r.method}, seed {r.seed}, {r.scope}: gained {r.gained_ids or "none"}; lost {r.lost_ids or "none"}.')
    lines+=['','## Independent verification and limits','',
      f'Verified all {verification["completion_files"]} completion-file hashes, all four prepared input hashes, frozen source/runtime/settings, '
      f'all 18 training/assessment receipts and histories. Independently refitted nominal-only scalers. Replayed all {verification["calibration_grids"]} grids '
      f'({verification["calibration_candidate_rows"]} candidate rows) with independently implemented postprocessing and rule selection. '
      f'All {verification["inference_spots"]} fixed timestamp checks across 120 candidate and 18 selected-checkpoint assessments reproduced float32 scores exactly. '
      'All continuous predictions, 288 metric rows and detected/missed IDs matched. This is sampled neural replay, not a second full inference pass. '
      'The corrected evaluator is shared with production and checked against the bundled independent reference; interval postprocessing, spot-window enumeration '
      'and event-ID enumeration are independently implemented.','',
      'The original 178-test thesis suite and three additional verifier boundary tests passed. '
      'Three seeds are limited variability evidence. Both calibration and assessment benchmarks have been reused; channel pools and methods reflect prior research. '
      'Inherited annotation-informed benchmark preparation, offline overlapping windows with future context, neutral out-of-scope annotations, '
      'additional training/selection compute and expanded score sources limit interpretation. No robust F0.5 ≥ 0.85 or deployment/generalization claim is made. '
      'Scoped input integrity verification does not close unrelated historical migration exceptions.','',
      '[Protocol](context-expansion-protocol-2026-11-04.md), '
      '[verification receipt](tables/context-expansion-2026-11-04/verification.json), '
      '[checkpoint spots](tables/context-expansion-2026-11-04/inference_spots.csv), '
      '[W&B evaluation runs](tables/context-expansion-2026-11-04/wandb-evaluation.json), '
      '[artifact backup Release](https://github.com/kamyarfaz/ESA-ADB/releases/tag/context-expansion-2026-11-13).']
    (ROOT/'docs/research/context-expansion-results-2026-11-13.md').write_text('\n'.join(lines)+'\n')
    # A compact paired plot shows every seed, with no best-seed selection.
    fig,axes=plt.subplots(2,3,figsize=(11,6),sharey=True)
    for j,method in enumerate(METHODS):
        for i,scope in enumerate(SCOPES):
            ax=axes[i,j]
            for seed,color in zip([42,43,44],['#2468a2','#c46d22','#368057']):
                r=pp[(pp.method==method)&(pp.seed==seed)&(pp.scope==scope)].iloc[0]
                ax.plot([0,1],[r['EW_F_0.50_control'],r['EW_F_0.50_context']],marker='o',label=f'Seed {seed}',color=color)
            ax.set_xticks([0,1],['Control','Add 33/49']);ax.set_ylim(0,1)
            ax.grid(axis='y',alpha=.2)
            if i==0:ax.set_title(['Short-no-roll','Masked + short-no-roll','Retain 14/21/29'][j])
            if j==0:ax.set_ylabel(('Target 40–47' if i==0 else 'System')+' F0.5')
    axes[0,0].legend(fontsize=8,loc='lower left')
    fig.suptitle('Matched input expansion: pooled 2007–2013, every seed')
    fig.tight_layout();fig.savefig(TABLE/'paired_f05.png',dpi=180);plt.close(fig)
    # Workbook inputs retain both scope denominators and metrics separately.
    wbrows=[]
    for job in plan['jobs']:
        for period in ['2007-2013',*[str(y) for y in range(2007,2014)]]:
            part=df[(df.name==job['name'])&(df.period==period)].set_index('scope')
            a=part.loc[SCOPES[0]];b=part.loc[SCOPES[1]]
            wbrows.append([NAMES[job['method']],job['seed'],job['arm'],'40–47'+(',14,21,29' if job['method']==METHODS[2] else '')+(',33,49' if job['arm']=='context' else ''),period,
                           a['EW_F_0.50'],a.EW_recall,a.nominal_false_alarm_rate,f'{a.TPe}/{a.num_true_events}',
                           b['EW_F_0.50'],b.EW_recall,b.nominal_false_alarm_rate,f'{b.TPe}/{b.num_true_events}'])
    wbrows=sorted(wbrows,key=lambda r:0 if r[4]=='2007-2013' else int(r[4]))
    OUTPUT.mkdir(parents=True,exist_ok=True)
    (OUTPUT/'workbook-data.json').write_text(json.dumps(dict(rows=wbrows),allow_nan=False))
    print(summary[(summary.period=='2007-2013')&(summary.metric=='EW_F_0.50')].to_string(index=False))


if __name__=='__main__':main()
