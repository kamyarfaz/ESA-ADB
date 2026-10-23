"""Publish every original/refined outcome, without test-based experiment selection."""
import json
import shutil
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'results_longrun/false_alarm_reduction_20261023_seed42_v1'
OUT=ROOT/'docs/research/tables/false-alarm-reduction-2026-10-23'
BUILD=ROOT/'outputs/false-alarm-reduction-20261023/build'
assert json.loads((OUT/'verification.json').read_text())['verified']
df=pd.read_csv(OUT/'verified_results.csv',dtype={'period':str})
events=pd.read_csv(OUT/'event_comparison.csv')
names={'pretrain_finetune':'Masked pretraining + fine-tuning','expanded_short_no_roll':'Short-no-roll + extra inputs'}
lines=['# Calibration-constrained false-alarm reduction', '',
'This study reassessed two fixed seed42 models trained on2000–2005, using2006 only for alarm research and selection, then evaluating2007–2013. No neural weights were retrained. Both original and refined outcomes are published. Assessment data were previously inspected, so this is exploratory benchmark reuse, not independent validation.', '',
'Neither refinement improved assessment F0.5. Pretraining retained its original rule; the expanded-input refinement reduced detection and sharply increased false alarm events. Original results remain the better reported outcomes.', '',
'## Calibration research and frozen decisions','',
'The pretrain/fine-tune model had one wholly false calibration alarm interval (960 seconds), but10379.778 seconds of nominal alarm duration overall. About90.75% of nominal alarm duration came from intervals that also touched annotations. The expanded model had no wholly false calibration alarm intervals, but15790.695 seconds of nominal alarm duration. Channel21 dominated349 of524 nominal alarm samples for the expanded model. Sample counts are a diagnostic, not a replacement for duration-based FPR. Long window-mean/overlap alarms can extend beyond short annotated events.', '',
'The prespecified study tested67 candidates per method: raw maximum scores and per-channel nominal2006 q99 normalization, a bounded threshold grid and minimum-duration filtering. Feasibility required retaining the exact original detected event IDs in both calibration scopes, no increase in nominal alarm duration or false alarm event count, and total alarm fraction<=1%. Selection used corrected system F0.5, then nominal alarm duration and fixed tie rules. Both rules were frozen before assessment. Normalized-score candidates did not satisfy the full feasibility constraints for either method.', '',
'Among normalized candidates retaining the original calibration IDs, the best system-F0.5 candidate for pretrain/fine-tune produced37870.695 nominal alarm seconds and2 false alarm events, compared with10379.778 seconds and1 for its original rule. For expanded inputs, the corresponding normalized candidate produced18059.778 nominal alarm seconds, exceeding the original15790.695. Normalization therefore failed the declared alarm constraints; it was not rejected based on assessment outcomes.', '',
'Pretrain/fine-tune retained its original threshold6.202751159667969: no tested change improved the objective while satisfying the constraints. Expanded inputs selected threshold6.900436115264893, a10% increase from6.273123741149902, with no gap filling and minimum duration1. This retained all9 detected system calibration IDs (and6 local IDs) while reducing nominal alarm duration from15790.695 to12910.695 seconds (18.24%). The small calibration event population does not guarantee later-year preservation.', '',
'## Pooled assessment results','',
'| Method | Rule | Scope | Detected / total | Precision | Event TPR | Time FPR (%) | F0.5 |',
'|---|---|---|---:|---:|---:|---:|---:|']
pooled=df[df.period=='2007-2013']
for _,r in pooled.iterrows():
 lines.append(f'| {names[r.method]} | {r.variant} | {r.scope} | {int(r.TPe)}/{int(r.num_true_events)} | {r.EW_precision:.6f} | {r.EW_recall:.6f} | {100*r.nominal_false_alarm_rate:.6f} | {r["EW_F_0.50"]:.6f} |')
lines+=['','## Event retention and interpretation','']
for method,name in names.items():
 for scope in ['system_all_channels','input_channels_40_47']:
  subset=events[(events.method==method)&(events.scope==scope)]
  lost=subset[subset.lost].event_id.tolist();gained=subset[subset.gained].event_id.tolist()
  a=pooled[(pooled.method==method)&(pooled.scope==scope)&(pooled.variant=='original')].iloc[0]
  b=pooled[(pooled.method==method)&(pooled.scope==scope)&(pooled.variant=='refined')].iloc[0]
  lines.append(f'- {name}, {scope}: F0.5 {a["EW_F_0.50"]:.6f} → {b["EW_F_0.50"]:.6f}; events {int(a.TPe)} → {int(b.TPe)}. Lost IDs: {", ".join(lost) or "none"}. Gained IDs: {", ".join(gained) or "none"}.')
lines+=['',
'The unchanged pretrain/fine-tune result is a valid negative calibration finding, not a new training result. Any assessment recall loss from the expanded refinement must remain visible despite its calibration retention. No additional thresholds or models were tried after inspecting these outcomes. The short-no-roll reference remains0.836070 system F0.5 and0.907140 for40–47, with48/91 and47/67 detected events respectively. Full event-level comparisons against it are in event_comparison.csv.', '',
'Precision is corrected ESA event precision, including its nominal-time penalty. Event TPR is detected target IDs divided by all target IDs. Time FPR is false-alarm duration divided by nominal duration, with other annotated intervals neutral. Smaller40–47 event scope is not an all-system result. Extra-input model still consumes14,21,29 as well as40–47. The study preserves existing offline window overlap and duration processing; no real-time latency claim is made.', '',
'The higher expanded-input threshold split 22 original alarm intervals. Total intervals increased from 115 to 211 and false alarm events from 11 to 70: 47 refined false intervals originated in annotation-touching parent intervals, and 23 in nominal parents. This explains why slightly less false-alarm duration nevertheless produced worse event precision. This post-assessment explanation did not feed back into selection; no gap-filling or further tuning was performed. See fragmentation.json.', '',
'## Annual and pooled metrics','']
for scope in ['system_all_channels','input_channels_40_47']:
 lines+=['### '+scope,'','| Method | Rule | Period | Detected / total | Precision | Event TPR | Time FPR (%) | F0.5 |','|---|---|---|---:|---:|---:|---:|---:|']
 for _,r in df[df.scope==scope].iterrows():
  lines.append(f'| {names[r.method]} | {r.variant} | {r.period} | {int(r.TPe)}/{int(r.num_true_events)} | {r.EW_precision:.6f} | {r.EW_recall:.6f} | {100*r.nominal_false_alarm_rate:.6f} | {r["EW_F_0.50"]:.6f} |')
lines+=['','## Verification and sources','',
'All64 metric rows were independently recomputed from raw annotations. Exact original aggregate-score and binary-prediction parity passed after fresh per-channel inference. Artifact, source, frozen-data and decision hashes passed, as did calibration event-ID constraints and frozen candidate selection. The140-test thesis suite includes inference-interruption replay and corrected evaluator parity. Original results and unsuccessful candidates are preserved.', '',
'[Full precision metrics](tables/false-alarm-reduction-2026-10-23/verified_results.csv), [event IDs](tables/false-alarm-reduction-2026-10-23/event_comparison.csv), [verification](tables/false-alarm-reduction-2026-10-23/verification.json), [frozen protocol and primary methodological sources](false-alarm-reduction-protocol-2026-10-23.md).', '',
'[Essential artifact Release](https://github.com/kamyarfaz/ESA-ADB/releases/tag/false-alarm-reduction-2026-10-23). Download checksum evidence is recorded in tables/false-alarm-reduction-2026-10-23/github-backup.json. This additive backup does not certify a whole-project migration or replace an external-drive snapshot.']
(ROOT/'docs/research/false-alarm-reduction-results-2026-10-23.md').write_text('\n'.join(lines)+'\n')
for method in names:
 for filename in ['calibration_channels.csv','calibration_months.csv','calibration_alarm_segments.csv','calibration_candidates.csv','diagnosis.json','frozen_rule.json']:
  shutil.copy2(RUN/method/filename,OUT/(method+'_'+filename))
shutil.copy2(RUN/'research_review.json',OUT/'research_review.json')
BUILD.mkdir(parents=True,exist_ok=True)
# Keep one row per technique. Numeric fields show the best reported system-F0.5
# variant descriptively; all experimental variants remain in the report above.
updates=[]
for method,label in [('pretrain_finetune','Masked pretrain + short-no-roll'),('expanded_short_no_roll','Short-no-roll + extra inputs')]:
 s=pooled[(pooled.method==method)&(pooled.scope=='system_all_channels')]
 original=s[s.variant=='original'].iloc[0];refined=s[s.variant=='refined'].iloc[0]
 winner=refined if refined['EW_F_0.50']>original['EW_F_0.50'] else original
 local=pooled[(pooled.method==method)&(pooled.scope=='input_channels_40_47')&(pooled.variant==winner.variant)].iloc[0]
 status='Original rule retained; no feasible improvement' if method=='pretrain_finetune' else ('Refined rule. Best reported system F0.5' if winner.variant=='refined' else 'Original rule best; refinement reported separately')
 updates.append(dict(label=label,values=[float(winner.EW_precision),float(winner.EW_recall),float(winner.nominal_false_alarm_rate),float(winner['EW_F_0.50']),float(local['EW_F_0.50']),status]))
(BUILD/'data.json').write_text(json.dumps(dict(updates=updates),indent=2)+'\n')
print(pooled[['method','variant','scope','TPe','EW_precision','EW_recall','nominal_false_alarm_rate','EW_F_0.50']].to_string(index=False))
