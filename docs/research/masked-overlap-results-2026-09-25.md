# Hidden-target overlap pilot: completed results

Source: `results_longrun/development/masked_overlap_2004_seed42_v1`.
Inputs 40–47; 25,000 nominal windows before 2004; five epochs per arm;
January–March calibration; April–December development assessment. Seed 42.
Both arms selected epoch 1. See [predeclared protocol](masked-overlap-pilot.md).

| Measure | Original masked control | Overlap-constrained masked |
|---|---:|---:|
| Calibration system F0.5 | 0.171616 | 0.000000 |
| Assessment system F0.5 | 0.090409 | 0.092948 |
| Assessment input-channel F0.5 | 0.091652 | 0.094262 |
| Detected system event IDs | 4 / 10 | 4 / 10 |
| False-positive events | 48 | 47 |
| Missed system event IDs | 6 | 6 |
| Nominal false-alarm hours | 90.781 | 34.133 |

The candidate meets the minimal predeclared condition for *considering*
continuation: higher development system F0.5 and no increase in false-positive
events. The absolute gain is only 0.002539, with no increase in detected-event
count. Alarm duration drops about 62.4%, but 47 false-positive events remain.
This is not sufficient evidence to promote the configuration or justify a long
run. The earlier matched unmasked baseline on this 2004 protocol scored 0.594608
system F0.5 (5 TP, 3 FP), substantially above either masked arm.

The candidate's calibration F0.5 is zero at all five epochs. The frozen rule
selects epoch 1 under the earliest-tie policy and issues no calibration alarms
(0 TP, 0 FP, 1 FN). Its threshold is 4.383703708648682, gap 0, duration 1.
That rule still generates assessment alarms because assessment scores can exceed
the calibration threshold. Zero calibration F0.5 therefore does not imply zero
assessment detections. Calibration contains only one positive event, limiting
what can be inferred from it.

Normal reconstruction MSE and synthetic penalty improve overall during training
without improving candidate calibration detection. Restricting segment placement
alone did not resolve the mismatch observed in the earlier diagnostic. This does
not rule out all masked models or identify the unique cause.

## Verification

- Both five-epoch histories are complete; selected checkpoints match earliest
  best calibration epochs. Logged loss matches normal MSE plus synthetic penalty.
- Reconstructed predictions from saved scores and frozen rules match exactly.
- Recomputed system and input-channel F0.5, TP/FP/FN and false-alarm duration agree
  with saved results, using the declared complete 30-second assessment grid.
- Original masked control predictions exactly reproduce the earlier 2004 pilot.
- Current package-source and annotation hashes match the run protocol.
- Final result files are present; the runner writes them after its final training
  CSV integrity gate. This review did not reread the full training CSV or rerun
  neural inference.

## Decision

Stop this placement variant for now. Keep the full-period baseline; no automatic
40-epoch extension, additional folds, OR ensemble or 2007–2013 evaluation. This
is an exploratory result on an already-inspected development year, not an
independent test. Preserve the evidence rather than adding more variants without
a new rationale. Compact artifacts are in `tables/masked-overlap-2026-09-25/`.
