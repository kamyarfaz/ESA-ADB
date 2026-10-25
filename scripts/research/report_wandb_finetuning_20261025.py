"""Build academic report and workbook updates from verified records."""
import json,shutil
from pathlib import Path
import pandas as pd
R=Path(__file__).resolve().parents[2];T=R/'docs/research/tables/wandb-finetuning-2026-10-25';RUN=R/'results_longrun/wandb_finetuning_20261024_seed42_v1'
assert json.loads((T/'verification.json').read_text())['verified']
df=pd.read_csv(T/'verified_results.csv',dtype={'period':str});cal=pd.read_csv(T/'calibration_comparison.csv');events=pd.read_csv(T/'event_ids.csv',dtype={'period':str})
sel=json.loads((RUN/'selection_complete.json').read_text())
names={'short_no_roll':'Original short-no-roll','pretrain_finetune':'Masked pretraining + short-no-roll','expanded_short_no_roll':'Short-no-roll + extra inputs'}
lines=['# W&B nine-run fine-tuning results — 25 October 2026','',
'Extra-input short-no-roll improved on the reused 2007–2013 assessment: system event F0.5 rose from 0.824405 to 0.844573, and channels-40–47 event F0.5 from 0.817353 to 0.830900. Target detections increased from 61/67 to 62/67; system detections increased from 76/91 to 80/91. The original short-no-roll and masked-pretraining models were retained by calibration and reproduce their previous predictions exactly.','',
'The extra-input result has the highest system F0.5 among these three selected models, but original short-no-roll still has the highest channels-40–47 F0.5 (0.907140) and substantially lower nominal-time FPR. Greater detection coverage does not imply better precision.','',
'## Fixed protocol and selection','',
'Training: 2000–2005 globally nominal windows. Calibration: 2006. Assessment: 2007–2013. Inputs are channels 40–47 for methods 1 and 2; method 3 additionally uses 14, 21, 29. All local event metrics target 40–47 only. Each existing checkpoint received three separate ten-epoch continuations at initial learning rates 3e-6, 1e-5 and 3e-5. AdamW was reset, cosine scheduling ended at 1e-6, batch size was 64 and seed was 42. Masked pretraining was not repeated. All methods retain their starting architecture, scaler and short-no-roll synthetic anomaly objective.','',
'Candidate epochs 5 and 10 used the frozen threshold/postprocessing grid and a 1% calibration alarm cap. The original model/rule was eligible as epoch 0. Selection used corrected system calibration F0.5 only; ties favor the original/earlier epoch then lower learning rate. All nine runs completed and all three choices froze before assessment. Unselected candidates were not assessed. W&B recorded the fixed local grid; it did not optimize or execute training itself.','',
'| Method | LR | Epoch 0 calibration F0.5 | Epoch 5 | Epoch 10 | Run-selected epoch |','|---|---:|---:|---:|---:|---:|']
for (m,lr),g in cal.groupby(['method','learning_rate'],sort=False):
 v=g.set_index('epoch');ep=int(g[g.run_selected].iloc[0].epoch)
 lines.append(f'| {names[m]} | {lr:g} | {v.loc[0,"calibration_system_f05"]:.9f} | {v.loc[5,"calibration_system_f05"]:.9f} | {v.loc[10,"calibration_system_f05"]:.9f} | {ep} |')
lines+=['','Across rates, methods 1 and 2 retain epoch 0 (the lower-rate run represents the tie). Method 3 selects LR3e-5, epoch10, with calibration system F0.5 0.978000023 versus original 0.977864804. This small calibration gain was not a guarantee of assessment improvement.','', '## Annual and pooled metrics','',
'Precision is the corrected ESA event precision including its nominal-time penalty. Event TPR counts detected target IDs; time FPR measures false-alarm duration divided by nominal duration. Non-target annotations remain neutral. Pooled metrics are recomputed over the continuous seven-year predictions, not averaged from annual values. IDs spanning years can appear in more than one annual count.']
for m in names:
 for scope in ['system_all_channels','input_channels_40_47']:
  lines+=['','### '+names[m]+' — '+scope,'','| Period | Variant | Detected / total | Precision | Event TPR | Time FPR (%) | F0.5 |','|---|---|---:|---:|---:|---:|---:|']
  for period in [str(y) for y in range(2007,2014)]+['2007-2013']:
   for _,r in df[(df.method==m)&(df.scope==scope)&(df.period==period)].iterrows():lines.append(f'| {period} | {r.variant} | {int(r.TPe)}/{int(r.num_true_events)} | {r.EW_precision:.6f} | {r.EW_recall:.6f} | {100*r.nominal_false_alarm_rate:.6f} | {r["EW_F_0.50"]:.6f} |')
lines+=['','## Event-ID changes','']
for m in names:
 for scope in ['system_all_channels','input_channels_40_47']:
  g=events[(events.method==m)&(events.scope==scope)&(events.period=='2007-2013')]
  gained=', '.join(g[g.gained].event_id.astype(str)) or 'none';lost=', '.join(g[g.lost].event_id.astype(str)) or 'none';miss=', '.join(g[~g.selected_detected].event_id.astype(str)) or 'none'
  lines.append(f'- {names[m]}, {scope}: gained {gained}; lost {lost}; selected missed {miss}.')
lines+=['','## Interpretation and limits','',
'For extra-input short-no-roll, false predicted events fell from 11 to 10 and nominal-time FPR decreased slightly (5.930307% to '+f'{100*df[(df.method=="expanded_short_no_roll")&(df.variant=="selected")&(df.period=="2007-2013")].iloc[0].nominal_false_alarm_rate:.6f}'+'%). The result combines more detected events and modestly fewer false alarms, but the long nominal alarm duration remains a weakness. It does not dominate original short-no-roll, which has lower coverage but much higher target precision and F0.5.','',
'The first two methods did not improve under this bounded search. Their original models won calibration, so their unchanged assessment results are fallback outcomes rather than evidence that every continuation has identical behavior. Training-loss reductions alone do not establish improvements in event detection.','',
'One seed was used. The original-short-no-roll LR1e-5 setting repeats the prior continuation and is not independent evidence. Nine runs and two candidate epochs per run increase calibration selection exposure. The benchmark has been reused across studies and guided research interest, so this is not independent confirmatory validation. Model starting histories and total compute differ, and method 3 receives extra information through three additional input channels. No robust achievement of the thesis F0.5 target is claimed.','',
'## Verification and artifacts','',
'Independent verification passed all receipt hashes, frozen source/runtime/settings/data/reference checks, nine ten-epoch histories, 18 full calibration grid replays from saved scores, within-run and cross-rate selection, selected weight/scaler equality, binary assessment replay, all96 corrected annual/pooled metric rows, and annual/pooled event IDs. Original controls match previously completed results. Verification replays stored scores; it does not independently rerun neural inference. All144 thesis tests passed. All nine training runs are finished in W&B.','',
'[Full metrics](tables/wandb-finetuning-2026-10-25/verified_results.csv), [annual and pooled event IDs](tables/wandb-finetuning-2026-10-25/event_ids.csv), [calibration attempts](tables/wandb-finetuning-2026-10-25/calibration_comparison.csv), [verification](tables/wandb-finetuning-2026-10-25/verification.json), [protocol](wandb-finetuning-protocol-2026-10-24.md).','',
'[W&B project](https://wandb.ai/kamyarfaz-politecnico-di-torino/esa-thesis-finetuning). Verified assessment is published in three separate evaluation runs, with original and selected annual/pooled tables.','',
'[GitHub Release](https://github.com/kamyarfaz/ESA-ADB/releases/tag/wandb-finetuning-2026-10-25). Release checksum evidence is recorded beside the metric tables. This additive backup does not replace the full external-drive copy or certify unrelated migration contents.']
(R/'docs/research/wandb-finetuning-results-2026-10-25.md').write_text('\n'.join(lines)+'\n')
for f in RUN.rglob('*'):
 if f.is_file() and f.suffix in ['.json','.csv']:
  target=T/'run-records'/f.relative_to(RUN);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,target)
updates=[]
for m,row in [('short_no_roll',7),('pretrain_finetune',15),('expanded_short_no_roll',16)]:
 g=df[(df.method==m)&(df.variant=='selected')&(df.period=='2007-2013')].set_index('scope');a=g.loc['system_all_channels'];b=g.loc['input_channels_40_47']
 updates.append(dict(row=row,metrics=[a.EW_precision,a.EW_recall,a.nominal_false_alarm_rate,a['EW_F_0.50'],b['EW_F_0.50']],status='W&B: original retained' if sel[m]['epoch']==0 else 'W&B: LR3e-5, epoch10',budget='40 + 3 × 10' if m!='pretrain_finetune' else '20 + 20 + 3 × 10'))
(R/'outputs/wandb-finetuning-20261025/updates.json').write_text(json.dumps(updates))
