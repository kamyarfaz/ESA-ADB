"""Build the descriptive comparison and workbook input from verified results."""
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
TABLE=ROOT/'docs/research/tables/combinations-2026-10-22'
verification=json.loads((TABLE/'verification.json').read_text())
assert verification['verified']
df=pd.read_csv(TABLE/'verified_results.csv',dtype={'period':str})
names={'control':'Short-no-roll reference','masked':'Masked standalone control','fusion':'Approach 1: score fusion','pretrain_finetune':'Approach 2: masked pretraining + short-no-roll','expanded_short_no_roll':'Approach 3: short-no-roll + extra inputs'}
order=['control','masked','fusion','pretrain_finetune','expanded_short_no_roll']
report=['# Frozen three-combination comparison', '',
'All three combinations completed. None improves pooled corrected all-channel event F0.5 over the short-no-roll reference (0.836070). These are descriptive results on a reused benchmark with one seed, not independent validation or evidence that the thesis target has been robustly achieved.', '',
'## Matched periods and deliberate differences', '',
'Every method uses training 2000–2005, calibration 2006, and continuous assessment 2007–2013. Inputs are channels 40–47, except approach 3, which also uses channels 14, 21 and 29 with explicit authorization. The 40–47 event scope remains identical for every method, including approach 3. There is no real training anomaly supervision: training uses nominal windows, with artificial anomalies where specified. MOSAD remains stopped and has no completed assessment.', '',
'Approach 1 normalizes short-no-roll and masked scores using nominal 2006 quantiles, then combines them with calibration-selected masked weight 0.5. It uses two 40-epoch components (80 model-epochs including the reused reference) and five scoring passes. Approach 2 uses 20 epochs of nominal masked pretraining followed by 20 epochs of unmasked short-no-roll fine-tuning. Approach 3 uses 40 epochs of short-no-roll with extra inputs. The masked standalone control uses 40 epochs with its legacy synthetic objective. This control is reported separately below; the workbook retains the original historical masked rows unchanged.', '',
'All arms use seed 42, batch 64, length 256, training stride 16, and all eligible nominal training windows. Calibration selects checkpoints and alarm rules under a 1% total-alarm cap. Selected epochs: '+', '.join(f'{names[k]} {v}' for k,v in verification['selected_epochs'].items())+'. The existing reference selected epoch 10 and retains its original shuffle trajectory. No checkpoint, threshold or combination was selected using assessment results.', '',
'## Pooled comparison', '',
'Precision is corrected ESA event precision; TPR is detected target event IDs divided by all target IDs. Time FPR is false-alarm duration divided by nominal duration. F0.5 is the corrected event score. Values below are fractions except explicitly marked percentages. Out-of-scope annotated intervals remain neutral for false alarms. Pooled metrics are recomputed over the full series, not averaged over annual scores.', '',
'| Method | System F0.5 | 40–47 F0.5 | System precision | System TPR | Time FPR (%) | System events |',
'|---|---:|---:|---:|---:|---:|---:|']
for m in order:
 s=df[(df.method==m)&(df.period=='2007-2013')&(df.scope=='system_all_channels')].iloc[0]
 l=df[(df.method==m)&(df.period=='2007-2013')&(df.scope=='input_channels_40_47')].iloc[0]
 report.append(f'| {names[m]} | {s["EW_F_0.50"]:.6f} | {l["EW_F_0.50"]:.6f} | {s.EW_precision:.6f} | {s.EW_recall:.6f} | {100*s.nominal_false_alarm_rate:.6f} | {int(s.TPe)}/{int(s.num_true_events)} |')
report += ['', '## Interpretation', '',
'Fusion reduces nominal false-alarm time from 0.029194% to 0.025544%, but detects 44 rather than 48 system events. Its F0.5 falls to 0.811670. Masked pretraining and fine-tuning detects 61 events, but precision falls to 0.834696 and F0.5 to 0.795675. Extra inputs detect 76 events, the highest coverage, but nominal false-alarm time reaches 5.930307% and F0.5 is 0.824405. The masked standalone control reaches 0.806278. Better coverage and lower false-alarm duration are different objectives; none of these changes improves the prespecified pooled system F0.5.', '',
'These observations do not isolate causal effects of pretraining, augmentation, or additional channels. Architectures/objectives, information and ensemble computation differ as declared. Historical high scores from other periods cannot establish superiority under this protocol. The 40–47 score has a smaller target-event population and must not be presented as all-system performance.', '',
'## Annual and pooled metrics for both scopes', '',
'Each annual row slices the same continuous predictions. An event crossing a year boundary can appear in more than one annual slice, so annual event counts need not sum to pooled unique IDs. These tables report every method and year, including unsuccessful results.']
for scope,title in [('system_all_channels','All-channel system events'),('input_channels_40_47','Events on channels 40–47')]:
 report+=['','### '+title,'','| Method | Period | Events detected / total | Precision | Event TPR | Time FPR (%) | F0.5 |','|---|---|---:|---:|---:|---:|---:|']
 for m in order:
  for _,r in df[(df.method==m)&(df.scope==scope)].iterrows():
   report.append(f'| {names[m]} | {r.period} | {int(r.TPe)}/{int(r.num_true_events)} | {r.EW_precision:.6f} | {r.EW_recall:.6f} | {100*r.nominal_false_alarm_rate:.6f} | {r["EW_F_0.50"]:.6f} |')
report+=['','## Verification and preservation','',
'The independent verifier checked all recorded run artifacts, frozen data, package sources and reference hashes; complete finite 40-epoch histories; earliest best eligible calibration checkpoints; stored calibration-grid rule choices; fusion weight and nominal normalization; exact prediction replay from saved scores; control-reference identity; and fusion-score identity. All 80 metric rows were independently recomputed from raw annotations using the corrected evaluator. This is score-to-prediction replay, not a second neural inference run. The full thesis suite passed 136 tests, including evaluator parity and exact interrupted-training resume tests.', '',
'Full precision data and verification: [verified_results.csv](tables/combinations-2026-10-22/verified_results.csv), [verification.json](tables/combinations-2026-10-22/verification.json). Frozen design: [protocol](combinations-full-protocol-2026-10-21.md). The updated workbook preserves all 32 original technique rows and adds three combination rows. The original historical masked results remain visible, while its new full-period standalone control is reported in this report and CSV. MOSAD is marked stopped without inventing scores.', '',
'Essential artifacts and the original workbook are backed up in the [GitHub Release](https://github.com/kamyarfaz/ESA-ADB/releases/tag/combinations-full-2026-10-22). Download checksum evidence is recorded in [github-backup.json](tables/combinations-2026-10-22/github-backup.json). This additive backup does not replace the full external-drive snapshot or resolve the previously documented unrelated historical transfer exception. No whole-project destination-copy verification is claimed.']
(ROOT/'docs/research/combinations-full-results-2026-10-22.md').write_text('\n'.join(report)+'\n')
rows=[]
for m,name,features,epochs,status in [('fusion','Score fusion','40–47','40 + 40','Complete. Two models'),('pretrain_finetune','Masked pretrain + short-no-roll','40–47','20 + 20','Complete'),('expanded_short_no_roll','Short-no-roll + extra inputs','40–47, 14, 21, 29',40,'Complete. Extra inputs')]:
 s=df[(df.method==m)&(df.period=='2007-2013')&(df.scope=='system_all_channels')].iloc[0]
 l=df[(df.method==m)&(df.period=='2007-2013')&(df.scope=='input_channels_40_47')].iloc[0]
 rows.append([name,features,'2000–2005','2006','2007–2013',1,epochs,s.EW_precision,s.EW_recall,s.nominal_false_alarm_rate,s['EW_F_0.50'],l['EW_F_0.50'],status])
(ROOT/'outputs/combinations-20261022/build/data.json').write_text(json.dumps(dict(rows=rows),indent=2)+'\n')
print('Report: 80 metric rows; workbook input: three combinations, historical rows retained.')
