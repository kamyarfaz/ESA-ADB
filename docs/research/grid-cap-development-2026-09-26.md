# Development assessment: grid/cap candidate rejected

Every candidate rule frozen on disk before reading any assessment arrays in this
run. Rules came from the completed calibration2x2 study, dense grid + nominal
false-alarm duration<=1%. All original cached predictions and event metrics
reproduced; inputs hashed before/after. No training, original rule changes or
2007–2013 access. April–December2003–2005 periods have been inspected previously;
these are development results, not an untouched final test.

|Year|Original system F0.5|Candidate system F0.5|Original TP/FP/FN|Candidate TP/FP/FN|
|---|---:|---:|---|---|
|2003|0.833170|0.833170|5/1/1|5/1/1|
|2004|0.428131|0.428248|6/9/4|6/9/4|
|2005|0.869055|0.082397|4/0/3|7/95/0|

Local input-channel results equal system in2003/2005.2004 local F0.5 changes
0.447280 to0.447408, with6TP/9FP/1FN in both. Tiny2004 improvement reflects
nominal alarm duration, not new event detection.

Decision: do not promote the candidate or deploy its rules.2005 calibration
improvement0.832880→0.936708 failed to generalize: increased recall comes with
95false-positive events in the later period. Both model and input channels are
unchanged, so this demonstrates failure of the selected decision policy in this
comparison, not proof that every lower threshold or every nominal-duration
constraint must fail. It does not establish the mechanism of the score shift.
Do not choose per-year winners or optimize another threshold on these outcomes.

Original baseline remains reference. Further work should investigate score
stability in training and calibration over time before another training or
threshold experiment. Keep all failed attempts as negative research evidence.
[Complete results and frozen protocol](tables/grid-cap-development-2026-09-26/).
