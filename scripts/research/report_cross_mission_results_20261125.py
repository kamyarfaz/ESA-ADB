"""Publish the complete descriptive result inventory without selecting a seed."""
import json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
T=ROOT/'docs/research/tables/cross-mission-2026-11-20-v2'
R=ROOT/'results_longrun/cross_mission_transfer_20261120_v2'
O=ROOT/'outputs/cross-mission-study-20261125'
NAMES={'short_no_roll':'Short-no-roll','pretrain_finetune':'Masked + short-no-roll','expanded_short_no_roll':'Expanded-input short-no-roll'}
METRICS=['EW_F_0.50','EW_recall','nominal_false_alarm_rate','TPe','FPe','FNe']

def main():
    v=json.loads((T/'verification.json').read_text());assert v['verified']
    prep=json.loads((T/'all-preparation-verification.json').read_text());assert prep['passed']
    d=pd.read_csv(T/'verified_results.csv',dtype={'period':str});p=d[d.period=='pooled']
    summary=d.groupby(['mission','condition','method','period','event_definition','scope'])[METRICS].agg(['mean','std','min','max'])
    summary.columns=['_'.join(x) for x in summary.columns];summary.reset_index().to_csv(T/'seed_summary.csv',index=False)
    pairs=[]
    for (mission,condition,seed,period,definition,scope),g in d.groupby(['mission','condition','seed','period','event_definition','scope']):
        base=g[g.method=='short_no_roll'].iloc[0]
        for _,r in g[g.method!='short_no_roll'].iterrows():pairs.append(dict(mission=mission,condition=condition,method=r.method,seed=int(seed),period=period,event_definition=definition,scope=scope,**{k+'_delta':r[k]-base[k] for k in METRICS}))
    pd.DataFrame(pairs).to_csv(T/'paired_changes.csv',index=False)
    lines=['# Cross-mission whole-system and transfer results — 25 November 2026','',
      'All 18 fresh models and all 18 predefined transfer conditions completed. The experiment measures whole-system detection using broader mission-specific inputs and separately tests existing Mission 1 weights on Mission 2. All three seeds are retained. Results below use the corrected event evaluator; input-channel coverage is not counted as detection.','',
      'The primary whole-system mean F0.5 values are 0.526498/0.589682/0.690503 for Mission 1 and 0.719673/0.635863/0.789561 for Mission 2, in short-no-roll/masked/expanded order. Mission 2 primary results improve over its previous study, but this is mainly rare-event detection: eight of nine fresh models detect 0/6 anomalies; only masked seed 43 detects 1/6. Mission 1 remains seed-sensitive. Expanded inputs improve primary system F0.5 over the paired baseline only in seed 44 for Mission 2 and seeds 42/44 for Mission 1, so improvement is not consistent across seeds.','',
      'Direct transfer is unsuccessful: all nine frozen-source-scaling models alarm continuously, with 100% nominal-time FPR and F0.5=0. Target normalization leaves system primary F0.5 means 0.081592/0.029058/0.047814 and mean time FPR 76.606%/95.992%/81.051%, respectively. High detected counts under these conditions reflect excessive alarm coverage, not useful transfer.','',
      '## Design and chronology','',
      'Mission 1 trains on 2000–2005, calibrates on 2006 and assesses 2007–2013. Mission 2 trains on 2000, calibrates January–September 2001 and assesses October 2001–June 2003. These are separate time splits, not predictions over training years. All 18 fresh checkpoint/rule selections jointly froze at '+v['joint_freeze_utc']+' before assessment arrays were prepared. Source-transfer models retain their historical Mission 1 2006 rules. No assessment result selected a channel map, seed, checkpoint or threshold.','',
      'The Transformer autoencoder uses 40 epochs and batch 32. Masked training uses 20 masked epochs followed by 20 short-no-roll epochs; other methods use 40 short-no-roll epochs. Seeds 42/43/44 are random training repeats, not channel numbers. Each mission uses 13 training-selected representatives of all target-channel groups; expanded models use 25 Mission 1 or 26 Mission 2 inputs. Nominal windows: 349,383 for Mission 1 and 98,224 for Mission 2, yielding 10,919 and 3,070 batches per epoch. Total fresh optimizer steps: 5,036,040. There are 120 calibration checkpoint evaluations, 18 fresh assessment passes and 18 transfer passes, plus independent auditing.','',
      'Mission 1 uses 30-second and Mission 2 uses 18-second grids; transfer uses 30 seconds to retain source numerical cadence. Raw values are causal and split-contained, with training-only median/IQR scaling. Only Mission 1 channels 62/63 allow 900-second-old readings, as explicitly approved after v1 preparation failed before any of its 18 runs started. All other inputs retain 600 seconds. V1 source, arrays and failure logs remain preserved. V2 completed with no failed training seeds or service restarts. Missing values are imputed from training statistics; existing small-IQR fallback 1 is retained for M1 channels 64/65 and M2 channels 27/26.','',
      'Fresh calibration selects corrected whole-system Anomaly+Rare Event F0.5 under a 1% calibration sample-alarm cap. The cap does not constrain assessment false-alarm duration. Historical source models instead retain the earlier selected 40–47 calibration rules. Scores take the maximum reconstruction error over model inputs, so expanded inputs also expand score sources. Overlapping reconstruction windows use context within the window; this is retrospective scoring, not demonstrated causal online anomaly detection.','',
      '## Channel meaning and transfer limitations','',
      'Public physical channel identities and units are anonymized. We cannot identify temperature or power sensors from correlation alone. The fixed mapping uses compatible anonymous groups and training behavior; none of its 13 assignments is stable across all four leave-season-out checks. It is an experimental correspondence, not established physical equivalence. Each mission has independently shifted/scaled time.','',
      'The nine historical Mission 1 context-arm models are assessed under unchanged source scaling and under target-training normalization fitted only on Mission 2 year 2000. Both preserve weights, mapping and source decision rules. Target normalization is target-data adaptation, not strict zero-shot transfer. Neither condition tunes a target threshold.','',
      'The old Mission 2 eight/eleven-input panels represented annotations for 14/198 training events, including 1/21 anomalies. New 13/26-input panels represent 192/198 and 198/198 events, respectively. Those are training annotation-membership counts, not actual detections or a limit on indirect detection. All system assessment IDs remain in the denominator.','',
      '## Pooled primary results','',
      'Primary events combine Anomaly and Rare Event. F0.5 means and sample standard deviations summarize all three seeds, not confidence intervals. TPR is detected event IDs divided by annotated IDs. Time FPR is false-alarm duration divided by nominal duration. Fixed-base scope means 13 representatives for fresh models, and the mapped original eight source inputs for transfer. Pooled metrics are recomputed continuously rather than averaged from years.','',
      '| Mission | Condition | Technique | Scope | F0.5 mean ± SD | Event TPR mean | Time FPR mean | Detected IDs, seeds 42 /43 /44 | Total IDs |','|---|---|---|---|---:|---:|---:|---|---:|']
    for (mission,condition,method,scope),g in p[p.event_definition=='anomaly_and_rare'].groupby(['mission','condition','method','scope'],sort=False):
        g=g.sort_values('seed');hits=' / '.join(str(int(x)) for x in g.TPe)
        lines.append(f'| M{mission} | {condition} | {NAMES[method]} | {scope} | {g["EW_F_0.50"].mean():.6f} ± {g["EW_F_0.50"].std():.6f} | {g.EW_recall.mean():.2%} | {g.nominal_false_alarm_rate.mean():.4%} | {hits} | {int(g.num_true_events.iloc[0])} |')
    lines+=['','## Every seed, scope and event definition','',
      'Anomaly-only reuses identical predictions and treats rare-event intervals as neutral for false alarms, following the corrected evaluator. It does not establish that the model recognizes rare events as normal.','',
      '| Mission | Condition | Technique | Seed | Definition | Scope | F0.5 | Detected / total | Event TPR | Time FPR |','|---|---|---|---:|---|---|---:|---:|---:|---:|']
    for _,r in p.iterrows():lines.append(f'| M{r.mission} | {r.condition} | {NAMES[r.method]} | {r.seed} | {r.event_definition} | {r.scope} | {r["EW_F_0.50"]:.6f} | {int(r.TPe)} / {int(r.num_true_events)} | {r.EW_recall:.2%} | {r.nominal_false_alarm_rate:.4%} |')
    lines+=['','## Paired changes','',
      'The following fresh-model contrasts pair the same mission, seed, period and scope with short-no-roll. Expanded inputs alter both information and score sources; the masked contrast changes the training procedure. Full paired tables retain both scopes, both definitions and all periods. Transfer contrasts are descriptive comparisons between historical source methods.','',
      '| Mission | Technique | Seed | Scope | Primary Δ F0.5 | Δ detected IDs | Δ time FPR (percentage points) |','|---|---|---:|---|---:|---:|---:|']
    for r in pairs:
        if r['condition']=='fresh_training' and r['period']=='pooled' and r['event_definition']=='anomaly_and_rare':lines.append(f'| M{r["mission"]} | {NAMES[r["method"]]} | {r["seed"]} | {r["scope"]} | {r["EW_F_0.50_delta"]:+.6f} | {r["TPe_delta"]:+.0f} | {100*r["nominal_false_alarm_rate_delta"]:+.5f} |')
    lines+=['','## Historical comparison','',
      'Previous Mission 2 pooled whole-system primary means were 0.137489 short-no-roll,0.218775 masked and 0.189472 expanded. Those models used 8/11 inputs, selected-eight calibration, batch 64 and slightly different training-window preparation. The new campaign uses 13/26 inputs, whole-system calibration and batch 32. Comparisons are descriptive; the change cannot be attributed solely to added channels. The previous Mission 1 context-arm whole-system means were 0.844667/0.853041/0.804662 in short-no-roll/masked/expanded order, all above the new means. Mission 1 also changes from historical restored prepared data to raw causal preparation and uses broader groups. The old selected 40–47 scope is not the new 13-input scope. Historical rows and original results are preserved.','',
      '## Independent verification and limits','',
      f'The audit checked {v["completion_files"]} completion-file hashes, all {v["raw_files"]} raw fingerprints, frozen source/runtime/settings, all 18 terminal histories, independent scaler refits, all {v["calibration_grids"]} calibration grids ({v["calibration_candidate_rows"]} candidate rows), every chosen rule/checkpoint, and {v["inference_spots"]} exact float32 inference spots across 156 checkpoint/condition passes. Full prediction replay matches all 720 annual/pooled metric rows and {v["event_id_rows"]} detected/missed ID records. Independent pandas source lookup checked {prep["raw_source_sample_checks"]} value/age/availability/mask samples across all eight prepared splits, including every assessment preparation. The 203-test suite passed, including bundled-reference event-metric parity. Neural/raw-source replay is sampled; full file hashes are checked. Corrected metric code is reused; postprocessing, rule ranking, windows and ID enumeration are independently implemented.','',
      'Only three neural seeds, reused benchmark years, anonymized physical meaning, unstable transfer mapping, changed preparation/selection/batch, increased score sources, and additional compute limit interpretation. No robust thesis F0.5 ≥0.85 achievement, independent generalization, or deployment claim follows from these results.','',
      '[All720 metric rows](tables/cross-mission-2026-11-20-v2/verified_results.csv), [detected/missed IDs](tables/cross-mission-2026-11-20-v2/verified_event_ids.csv), [paired changes](tables/cross-mission-2026-11-20-v2/paired_changes.csv), [seed summaries](tables/cross-mission-2026-11-20-v2/seed_summary.csv), [audit](tables/cross-mission-2026-11-20-v2/verification.json).','',
      '[Approved protocol](cross-mission-whole-system-protocol-v2-2026-11-20.md), [channel research](cross-mission-channel-research-2026-11-18.md), [W&B evaluation links](tables/cross-mission-2026-11-20-v2/wandb-evaluation.json), [essential artifact Release](https://github.com/kamyarfaz/ESA-ADB/releases/tag/cross-mission-whole-system-2026-11-25).']
    (ROOT/'docs/research/cross-mission-results-2026-11-25.md').write_text('\n'.join(lines)+'\n')
    rows=[]
    for (name,condition,period,definition),g in d.groupby(['name','condition','period','event_definition'],sort=False):
        a=g[g.scope=='fixed_base_channels'].iloc[0];b=g[g.scope=='system_all_channels'].iloc[0]
        rows.append([int(a.mission),NAMES[a.method],int(a.seed),condition,definition,period,a['EW_F_0.50'],a.EW_recall,a.nominal_false_alarm_rate,int(a.TPe),int(a.num_true_events),b['EW_F_0.50'],b.EW_recall,b.nominal_false_alarm_rate,int(b.TPe),int(b.num_true_events)])
    rows.sort(key=lambda r:(r[0],r[3],r[5]!='pooled',r[5],r[4],r[1],r[2]))
    O.mkdir(parents=True,exist_ok=True);(O/'workbook-data.json').write_text(json.dumps(dict(rows=rows),allow_nan=False))
    print(p[(p.scope=='system_all_channels')&(p.event_definition=='anomaly_and_rare')][['mission','condition','method','seed',*METRICS]].to_string(index=False))

if __name__=='__main__':main()
