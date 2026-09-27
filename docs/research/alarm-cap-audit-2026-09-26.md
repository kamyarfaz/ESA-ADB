# Alarm-cap audit: separate the cap from threshold resolution

Calibration-only analysis of saved selected-checkpoint candidate grids for2003,
2004 and2005. All original selected rules reproduced. No assessment access,
new training, checkpoint reselection or frozen rule modification.

Compared existing total predicted-positive fraction<=1% with nominal false-alarm
duration fraction<=1%, retaining saturation guard and existing F0.5/tie ordering.
The alternative limit is exploratory, not an established operational requirement.

| Year | Existing calibration F0.5 | Alternative-cap F0.5, same grid |
|---|---:|---:|
|2003|0.714241|0.714241|
|2004|0.383907|0.383907|
|2005|0.832880|0.832880|

Every selected rule is identical. Therefore changing the cap alone does NOT
reproduce the earlier2005 diagnostic gain on the original grid.

2005 unmerged threshold grid jumps from6.353486 (3TP/4FP,2%alarm occupancy) to
15.792491 (2TP/0FP,1%occupancy). The diagnostic12.633993 threshold (3TP/0FP,
1.1728%occupancy) lies between them and was absent from that search. It also
violates the original cap. Both grid resolution and cap matter for this example;
neither may be credited alone with the exploratory0.936802 calibration score.

Recommended next experiment: a predeclared2x2 calibration-only diagnostic with
original versus uniformly denser quantile grids, crossed with total-alarm versus
nominal-false-alarm caps. Keep selected checkpoints fixed, retain all postprocess
candidates, and use identical grids under both caps. Report all development folds,
TP/FP/FN, total occupancy and nominal false-alarm duration. Do not report the
calibration winner as independently validated. If worth pursuing, freeze the
policy before separate development assessment and disclose prior inspection.
Do not rerun2007–2013 to select a policy. No extra training is required.

Evidence: [tables](tables/alarm-cap-audit-2026-09-26/).
Reproduction script: `scripts/research/audit_alarm_cap.py` (new output required).
