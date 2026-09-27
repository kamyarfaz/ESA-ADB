# Partial calibration grid/cap comparison

Plan recorded before computation: original grid versus original plus uniformly
spaced90–100percentiles in0.1percentile increments (retaining original tail
percentiles), crossed with total-alarm1% versus nominal-false-alarm1% constraints.
Same selected checkpoints, merge gaps0/16/64, minimum durations1/8/32, existing
saturation guard and tie ordering. No assessment files read or decisions changed.
This is exploratory following inspection of2005 calibration, not independent
validation. Alternative cap is not an established operational requirement.

| Grid / constraint | 2004 calibration F0.5 | 2005 calibration F0.5 |
|---|---:|---:|
| Original / total alarm |0.383907|0.832880|
| Original / nominal false alarm |0.383907|0.832880|
| Dense / total alarm |0.384048|0.832955|
| Dense / nominal false alarm |0.384048|0.936708|

2004 event counts remain1TP/2FP/0FN.2005 dense+nominal policy reaches3TP/0FP/1FN
versus2TP/0FP/2FN otherwise. Its total alarm fraction is1.1975%; nominal false
alarm fraction0.1126%. The tiny improvements under the total cap reflect reduced
nominal alarm duration, not new detections. Do not substitute the earlier hand-
chosen threshold's0.936802 for this predeclared-grid result0.936708.

## Incomplete2003 verification

Fresh CPU diagnostic of the saved2003epoch25 checkpoint failed strict reproduction:
F0.5=0.7141835329272609 versus saved0.7142410612656628. Both have1TP/0FP/2FN,
but nominal alarm seconds are1705.224 versus745.224 (960seconds difference).
Cause unresolved; no tolerance relaxed and no2003 dense results accepted.
Original-grid cap-only comparison for2003 is available in the previous audit,
but cannot fill the missing dense-grid cells. Outputs preserved at
`results_longrun/diagnostics/baseline_2003_epoch25_v1`; no completion marker.
Initial CSV checksum passed; failure preceded final checksum and acceptance.

2004/2005 completed diagnostic caches reproduced frozen predictions and metrics;
input hashes checked before/after.68thesis tests passed during this work; no
production selection code changed. Comparison status is explicitly partial.

Next: resolve2003 inference reproduction (CPU/GPU/checkpoint comparison) before
claiming a three-fold result. Keep this as a promising calibration-policy
hypothesis; no new training or benchmark assessment rerun warranted yet.
[Evidence](tables/grid-cap-comparison-2026-09-26/).
