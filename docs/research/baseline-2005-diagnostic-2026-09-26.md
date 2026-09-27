# 2005 frozen calibration diagnostic

Saved epoch5 baseline, channels40–47, January–March2005. CPU inference completed;
training CSV golden checksum passed before and after; original calibration
F0.5 reproduced exactly at0.8328804736937059. No assessment data accessed,
no training or production decision changed. Nominal training reference10000
windows; calibration8093windows. Full outputs:
`results_longrun/diagnostics/baseline_2005_epoch5_v1`.

Detected id_10 and id_17. Missed id_98 (locally annotated March25–28) and id_99
(outside inputs). Frozen threshold15.792491; id_98 peak12.881515, strongest
channel43, followed by45 (12.568056),41 (12.196573),46 (9.952641),42 (8.733096).
These maxima need not occur at the same instant; this does not prove correlation.

| Threshold multiplier | F0.5 | TP | FP | FN | Predicted anomalous fraction |
|---|---:|---:|---:|---:|---:|
|1.00 frozen|0.832880|2|0|2|1.0000%|
|0.90|0.832806|2|0|2|1.0617%|
|0.80 diagnostic|0.936802|3|0|1|1.1728%|
|0.75 diagnostic|0.936614|3|0|1|1.2099%|

A20% reduction recovers id_98 without adding event-level FP on this calibration
period, but exceeds the existing1% predicted-positive-rate cap. This cap counts
all alarmed samples, including true-anomaly time, not just false alarms. It is a
plausible contributor to this miss; the diagnostic threshold was not necessarily
in the original quantile grid. This is calibration-only exploration, not a new
validated improvement or permission to change a frozen assessment rule.

Next focused hypothesis: review the rationale for the1% total-alarm cap, then
predeclare a development comparison against a false-alarm constraint that does
not penalize time spent inside true events. Use2003–2005 development folds and
report alarm occupancy as well as FP counts/duration. Do not pick the new policy
using2007–2013 scores. A naive globally lowered threshold is contradicted by the
2004 and2006 diagnostics, where it can add false alarms. No new training needed
for this policy investigation. Timing-context candidate stays unpromoted.

[Compact tables](tables/baseline-2005-diagnostic-2026-09-26/).
