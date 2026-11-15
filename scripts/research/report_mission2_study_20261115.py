"""Report every Mission 2 seed and scope, without assessment selection."""
import json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
T=ROOT/'docs/research/tables/mission2-three-technique-2026-11-14'
R=ROOT/'results_longrun/mission2_three_technique_20261114_v1'
O=ROOT/'outputs/mission2-study-20261115'
NAMES={'short_no_roll':'Short-no-roll','pretrain_finetune':'Masked + short-no-roll','expanded_short_no_roll':'Expanded-input short-no-roll'}
METRICS=['EW_F_0.50','EW_recall','nominal_false_alarm_rate','TPe','FPe','FNe']

def main():
 v=json.loads((T/'verification.json').read_text());assert v['verified']
 d=pd.read_csv(T/'verified_results.csv',dtype={'period':str});p=d[d.period=='pooled']
 summary=d.groupby(['method','period','event_definition','scope'])[METRICS].agg(['mean','std','min','max'])
 summary.columns=['_'.join(x) for x in summary.columns];summary.reset_index().to_csv(T/'seed_summary.csv',index=False)
 pairs=[]
 for (seed,period,definition,scope),g in d.groupby(['seed','period','event_definition','scope']):
  base=g[g.method=='short_no_roll'].iloc[0]
  for _,r in g[g.method!='short_no_roll'].iterrows():
   pairs.append(dict(method=r.method,seed=int(seed),period=period,event_definition=definition,scope=scope,**{k+'_delta':r[k]-base[k] for k in METRICS}))
 pd.DataFrame(pairs).to_csv(T/'paired_changes.csv',index=False)
 lines=['# Mission 2 three-technique study — 15 November 2026','',
 'The current Mission 2 setup has low detection recall and does not reproduce the strong Mission 1 results. Primary pooled mean selected-scope F0.5 is 0.413341 for short-no-roll, 0.367278 for masked pretraining and 0.481984 for expanded inputs. System-scope means are 0.137489, 0.218775 and 0.189472. Expansion improves selected-scope F0.5 in seeds42/43 but worsens seed44; masked pretraining improves seed42 but worsens seeds43/44. No consistent improvement across seeds is established.','',
 'All nine fresh runs completed 40 epochs and assessment, with no failed or omitted runs and no service restarts. These are Mission 2 results; they are not a continuation of the Mission 1 assessment period or a direct weight transfer.','',
 '## Design','',
 'Train on 2000, calibrate on January–September 2001, and assess continuously from 1 October 2001 through 30 June 2003. The last assessed grid timestamp is '+str(pd.Timestamp(__import__('numpy').load(R/'prepared/assessment/times.npy')[-1]))+'. All nine checkpoint/rule choices jointly froze at '+v['joint_freeze_utc']+' before assessment preparation or inference.','',
 'The baseline is already a Transformer autoencoder. Short-no-roll uses 40 epochs; masked pretraining uses 20 masked epochs followed by 20 short-no-roll epochs. The expanded technique uses 40 short-no-roll epochs with additional inputs. Each technique has fresh seeds 42, 43 and 44; seeds are random initialization/training repeats, not channel IDs. There are 98,228 eligible windows and 1,535 batches per epoch: 552,600 optimizer steps across nine models, plus 60 calibration-checkpoint evaluations and nine assessment passes.','',
 'Base inputs are 58,59,75,76,77,83,84,85. The expanded technique adds 90,86,89, selected only from Mission 2 training-year correlations within compatible anonymized subsystem/unit groups. Their worst-quarter absolute Spearman correlations with anchor59 are 0.98631, 0.98531 and 0.98516. This supports stable similarity, not proven incremental information; redundancy is explicitly allowed. Channel numbers and weights are not transferred from Mission 1.','',
 'Values use an 18-second causal previous-observation grid, a 600-second age limit, no backfill, and training-median missing-value imputation. Scalers use globally unannotated complete training rows. All methods share the same training exclusion mask. Values are not restored using anomaly annotations. A 256-sample window represents 76.8 minutes and reconstruction is retrospective with future context within a window. Scores maximize reconstruction error over all model inputs, so additional inputs also expand score sources.','',
 'Checkpoint and alarm-rule selection uses corrected selected-eight-channel Anomaly + Rare Event F0.5 on calibration only, subject to a 1% calibration sample-alarm cap. Only six selected-scope calibration event IDs are available; that is limited evidence for future behavior. The cap does not guarantee low assessment false-alarm duration.','',
 '## Pooled primary results: Anomaly + Rare Event','',
 'Selected scope always means the same eight base channels, including for the expanded model. System scope includes all Mission 2 channels. TPR is detected event IDs / annotated IDs; time FPR is false-alarm duration / nominal duration. Mean ± sample SD summarizes three seeds, not a confidence interval. Pooled metrics are recomputed continuously, not averaged from annual results.','',
 '| Technique | Scope | F0.5 mean ± SD | TPR mean | Time FPR mean |','|---|---|---:|---:|---:|']
 for (method,scope),g in p[p.event_definition=='anomaly_and_rare'].groupby(['method','scope'],sort=False):
  lines.append(f'| {NAMES[method]} | {scope} | {g["EW_F_0.50"].mean():.6f} ± {g["EW_F_0.50"].std():.6f} | {g.EW_recall.mean():.2%} | {g.nominal_false_alarm_rate.mean():.4%} |')
 lines+=['','## Every seed and both event definitions','',
 'Anomaly-only is a secondary view of identical predictions. Rare-event and out-of-scope annotations remain neutral for false alarms under the corrected evaluator. This does not demonstrate recognition of rare events as normal.','',
 '| Technique | Seed | Definition | Scope | F0.5 | Detected / total | False events | Time FPR |','|---|---:|---|---|---:|---:|---:|---:|']
 for _,r in p.iterrows():
  lines.append(f'| {NAMES[r.method]} | {r.seed} | {r.event_definition} | {r.scope} | {r["EW_F_0.50"]:.6f} | {int(r.TPe)} / {int(r.num_true_events)} | {int(r.FPe)} | {r.nominal_false_alarm_rate:.4%} |')
 lines+=['','## Paired changes against short-no-roll','',
 'These compare the same seed, period and scope. The expanded-versus-base contrast tests added inputs and score sources together. The masked contrast changes the training procedure. No best seed is selected.','',
 '| Technique | Seed | Scope | Primary Δ F0.5 | Δ detected IDs | Δ time FPR (percentage points) |','|---|---:|---|---:|---:|---:|']
 for r in pairs:
  if r['period']=='pooled' and r['event_definition']=='anomaly_and_rare':lines.append(f'| {NAMES[r["method"]]} | {r["seed"]} | {r["scope"]} | {r["EW_F_0.50_delta"]:+.6f} | {r["TPe_delta"]:+.0f} | {100*r["nominal_false_alarm_rate_delta"]:+.5f} |')
 lines+=['','## Frozen calibration choices','', '| Technique | Seed | Epoch | Calibration F0.5 | Threshold | Gap | Minimum duration |','|---|---:|---:|---:|---:|---:|---:|']
 cal=pd.read_csv(T/'verified_calibration.csv')
 for _,r in cal[cal.selected].iterrows():lines.append(f'| {NAMES[r.method]} | {r.seed} | {r.epoch} | {r.val_esa_f05:.6f} | {r.threshold:.8g} | {r.merge_gap} | {r.min_dur} |')
 lines+=['','## Verification and interpretation','',
 f'Independent audit checked {v["completion_files"]} completion-file hashes, raw inputs, source/runtime/settings, all nine final training states and histories, nominal-only scaler refits, all {v["calibration_grids"]} calibration grids ({v["calibration_candidate_rows"]} candidates), all frozen choices, and {v["inference_spots"]} exact float32 inference spots across 69 checkpoints/passes. Full prediction replay, all 144 annual/pooled metric rows and {v["event_id_rows"]} detected/missed ID records match. Independent pandas source lookup additionally checked 204,655 assessment value/availability/age samples. All 189 thesis tests passed, including evaluator-reference parity and verifier boundaries. Neural replay and raw-source replay are sampled, not full second passes. The corrected evaluator is reused; postprocessing, rule ranking, inference-window enumeration and event-ID enumeration are independent.','',
 'Three seeds, six selected calibration IDs, reused benchmark data, anonymized channels, different event composition and temporal resolution, neutral annotations and offline reconstruction limit generalization. Additional channels require additional compute and may be redundant. Differences from Mission 1 cannot be attributed solely to the mission or channel quality because preparation, chronology and event populations differ. No robust thesis F0.5 ≥ 0.85 achievement or deployment claim is made. Historical results are preserved.','',
 '[All 144 metrics](tables/mission2-three-technique-2026-11-14/verified_results.csv), [all detected/missed IDs](tables/mission2-three-technique-2026-11-14/verified_event_ids.csv), [all paired changes](tables/mission2-three-technique-2026-11-14/paired_changes.csv), [seed summaries](tables/mission2-three-technique-2026-11-14/seed_summary.csv), [verification](tables/mission2-three-technique-2026-11-14/verification.json).','',
 '[Protocol](mission2-three-technique-protocol-2026-11-14.md), [W&B runs](tables/mission2-three-technique-2026-11-14/wandb-evaluation.json), [artifact Release](https://github.com/kamyarfaz/ESA-ADB/releases/tag/mission2-three-technique-2026-11-15).']
 (ROOT/'docs/research/mission2-three-technique-results-2026-11-15.md').write_text('\n'.join(lines)+'\n')
 rows=[]
 for (name,period,definition),g in d.groupby(['name','period','event_definition'],sort=False):
  a=g[g.scope=='selected_eight_channels'].iloc[0];b=g[g.scope=='system_all_channels'].iloc[0]
  rows.append([NAMES[a.method],int(a.seed),definition,period,a['EW_F_0.50'],a.EW_recall,a.nominal_false_alarm_rate,int(a.TPe),int(a.num_true_events),b['EW_F_0.50'],b.EW_recall,b.nominal_false_alarm_rate,int(b.TPe),int(b.num_true_events)])
 rows.sort(key=lambda r:(r[3]!='pooled',r[3],r[2],r[0],r[1]))
 O.mkdir(parents=True,exist_ok=True);(O/'workbook-data.json').write_text(json.dumps(dict(rows=rows),allow_nan=False))
 print(p[p.event_definition=='anomaly_and_rare'][['method','seed','scope',*METRICS]].to_string(index=False))
if __name__=='__main__':main()
