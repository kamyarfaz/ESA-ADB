# Completed calibration grid/cap comparison

All three frozen calibration rules now reproduced.2003GPU diagnostic exactly
reproduces historicalF0.5 .7142410612656628 and nominalalarm745.224seconds.
CPU/GPU cached raw/scaled inputs and timestamps are identical; checkpoint/rule
hashes match.32scores onGPU equal6.974297523498535, whileCPU scores are the next
float32 value6.974298000335693. Strict greater-than explains the full16minute
alarm extension. Maximum CPU/GPU aggregate score difference3.5762786865234375e-6.
This identifies a device-dependent numerical boundary difference in this run,
not a repair of the separate historical intermittent integrity problems.
No tolerances relaxed, thresholds changed, or failedCPU diagnostics accepted.

Using the verifiedGPU2003 cache, GPU2004 cache and CPU2005 cache (each reproduces
its frozen metrics), the unchanged2x2 plan was rerun in a new directory. Original
partial run preserved. All included inputs hashed before/after comparison.

| Search / constraint |2003 F0.5|2004 F0.5|2005 F0.5|
|---|---:|---:|---:|
|Original / total alarm1%|0.714241|0.383907|0.832880|
|Original / nominal false alarm1%|0.714241|0.383907|0.832880|
|Dense / total alarm1%|0.714241|0.384048|0.832955|
|Dense / nominal false alarm1%|0.714241|0.384048|0.936708|

Only2005 gains a detected event:3TP/0FP/1FN versus2TP/0FP/2FN.2004 retains1TP/2FP;
its smallscore change is nominal alarm duration.2003 remains1TP/0FP/2FN.
Candidate hypothesis is therefore promising for one calibration fold, not a
robustly validated improvement.1% nominal false-alarm cap remains exploratory.

Next proposed step: freeze this candidate policy and its per-fold calibration
rules, then compare against original frozen rules on the existing April–December
2003–2005 development assessments using saved scores. No retraining required.
These periods have been previously inspected and are development data, not an
untouched final test. Report every fold and both scopes without selecting rules
or per-year winners using assessment scores. Do not rerun2007–2013 to select the
policy. This review itself did not access any assessment predictions or scores.

[Evidence](tables/grid-cap-comparison-complete-2026-09-26/).
