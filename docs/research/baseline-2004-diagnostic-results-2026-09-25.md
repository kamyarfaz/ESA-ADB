# Completed frozen 2004 calibration diagnostic

Source: `results_longrun/diagnostics/baseline_2004_epoch35_v2`.
Epoch35; inputs40–47; unchanged model, scaler and rule. Final results reproduce
calibration F0.5=0.38390663095893557, TP1 FP2 FN0. The completed runner passed its
final CSV checksum gate. This review independently reproduced metrics and frozen
predictions from caches, checked annotation hashes, and inspected all three plots.
No training, threshold tuning or assessment access.

| Calibration alarm (UTC) | Label | Duration | Peak score | Main channel pattern |
|---|---|---:|---:|---|
| Jan1 04:16–05:36 | Detected id_12 |80min|3.55125|47 alone crosses threshold|
| Jan24 06:08–09:20 | False positive |192min|3.64783|42/43/44/45/46 cross threshold|
| Feb14 14:24–15:12 | False positive |48min|3.62879|46 alone crosses threshold|

Frozen threshold3.13217. Both false alarms peak above the sole detected event's
peak. Therefore a higher scalar threshold sufficient to remove either false
episode entirely would also lose that real event under this scoring and current
gap0/min-duration1 rule. This is a score-separation issue in this calibration,
not an invitation to search more thresholds. A multi-channel voting rule also
has no obvious justification: the detected event is a single-channel threshold
crossing, while one false alarm affects several channels.

## Prepared-value repetition

During the January24 false alarm, channels41–46 repeat their preceding value in
52.5–61.1% of successive30-second samples. In the four hours before/after the
alarm, the same channels have approximately9.0–23.2% repetition. The plots show
flat sections in the prepared telemetry during this interval. This is an observed
change in repetition, not proof of missing measurements, forward-fill artifacts,
or an unannotated physical anomaly. Native timestamps must be inspected before
making such claims. Channels40/47 also repeat frequently in surrounding periods.

The February14 false alarm is primarily a channel46 reconstruction-error peak;
these multichannel input plots alone do not establish a physical or preprocessing
cause. Channel error attribution is not causal feature importance. Window means
and overlapping aggregation spread each error over time; alarm boundaries do
not establish the exact onset or duration of the underlying signal change.

The sole positive event id_12 is already in progress at the January1 calibration
boundary and remains annotated through January8. It is not a newly starting
independent event inside this period. Detecting it once gives event recall1.0;
that does not mean the model alarms throughout its duration.

## Next bounded investigation

Inspect original sampling timestamps and the preparation logic for channels40–47
around January24, with the adjacent normal periods and other two alarm contexts
as controls. Determine whether repeated prepared values correspond to absent
native observations or genuinely repeated measurements. Do not remove samples,
change labels, add inputs or train a new model before this distinction is known.
No new run has been launched. Preserve the full baseline and existing results.

Compact evidence: `tables/baseline-2004-diagnostic-2026-09-25/`.
Full plots: `alarm_00.png`, `alarm_01.png`, `alarm_02.png` in the source run.
