"""Publish unchanged selected results and unsuccessful calibration candidates."""
import json,shutil
from pathlib import Path
import pandas as pd
root=Path(__file__).resolve().parents[2]
out=root/'docs/research/tables/short-no-roll-finetune-2026-10-23'
run=root/'results_longrun/short_no_roll_finetune_20261023_seed42_v1'
assert json.loads((out/'verification.json').read_text())['verified']
df=pd.read_csv(out/'verified_results.csv',dtype={'period':str})
cal=pd.read_csv(out/'calibration_comparison.csv')
lines=['# Short-no-roll fine-tuning results — 23 October 2026','',
'Ten additional training epochs did not improve calibration. The original checkpoint and alarm rule were retained, and fresh assessment predictions exactly reproduce the original predictions. No improvement is claimed.','',
'Training uses channels 40–47 and globally nominal windows from 2000–2005, with the original short-no-roll synthetic augmentation objective. The original calibration-selected epoch10 checkpoint was continued for ten epochs at initial learning rate 1e-5, one tenth of its original rate. AdamW and cosine scheduler were reset; seed42 and batch64 were fixed. All decisions used 2006 calibration. The original checkpoint was an eligible fallback, with original/earliest winning ties. No changes were selected from assessment results.','',
'## Calibration selection','',
'| Continuation epoch | Corrected system event F0.5 | Selected |','|---|---:|---|']
for _,r in cal.iterrows():lines.append(f'| {int(r.epoch)} | {r.calibration_f05:.15f} | {"Yes" if r.epoch==0 else "No"} |')
lines+=['','Training loss fell from 0.839924 to 0.773839, but calibration event F0.5 did not improve. Training loss and the event-selection metric measure different objectives. The unsuccessful epoch5 and epoch10 candidate grids and training history are preserved. Unselected models were not assessed on 2007–2013.','',
'## Annual and pooled assessment','',
'Original and selected results are identical, so the following tables show both without duplicating columns. The CSV retains all 32 method/period/scope records.']
for scope in ['system_all_channels','input_channels_40_47']:
 lines+=['','### '+scope,'','| Period | Detected / total | Precision | Event TPR | Time FPR (%) | F0.5 |','|---|---:|---:|---:|---:|---:|']
 for _,r in df[(df.method=='finetuned')&(df.scope==scope)].iterrows():
  lines.append(f'| {r.period} | {int(r.TPe)}/{int(r.num_true_events)} | {r.EW_precision:.6f} | {r.EW_recall:.6f} | {100*r.nominal_false_alarm_rate:.6f} | {r["EW_F_0.50"]:.6f} |')
lines+=['','## Verification and limitations','',
'Independent verification passed frozen data/source/protocol/reference hashes, completion and training receipts, ten-epoch history, candidate-grid selection, exact selected-weight/scaler equality to the original checkpoint, binary prediction replay, and all32 corrected annual/pooled metric rows recomputed from raw annotations. No event IDs were gained or lost. The thesis suite passed141 tests, including interruption recovery and original fallback.','',
'Precision uses the corrected ESA event definition with nominal-time penalty. TPR measures event-ID detection; time FPR measures alarm duration over nominal duration. Annual event counts can include IDs spanning year boundaries; pooled scores are recomputed, not averaged. One seed and a repeatedly inspected benchmark limit generalization claims. Additional training compute did not produce a selected model improvement. This is not independent validation or robust achievement of the thesis target.','',
'[Full metrics](tables/short-no-roll-finetune-2026-10-23/verified_results.csv), [detected and missed event IDs](tables/short-no-roll-finetune-2026-10-23/event_ids.csv), [verification](tables/short-no-roll-finetune-2026-10-23/verification.json), [frozen protocol](short-no-roll-finetune-protocol-2026-10-23.md).','',
'[Artifact Release](https://github.com/kamyarfaz/ESA-ADB/releases/tag/short-no-roll-finetune-2026-10-23). Download checksum evidence is in github-backup.json beside the verified tables. This additive backup does not replace the full external-drive copy or certify unrelated migration contents.']
(root/'docs/research/short-no-roll-finetune-results-2026-10-23.md').write_text('\n'.join(lines)+'\n')
for name in ['history.csv','calibration_candidates_epoch5.csv','calibration_candidates_epoch10.csv','frozen_rule.json']:
 shutil.copy2(run/'finetuned'/name,out/name)
print(df[(df.method=='finetuned')&(df.period=='2007-2013')][['scope','TPe','EW_precision','EW_recall','nominal_false_alarm_rate','EW_F_0.50']].to_string(index=False))
