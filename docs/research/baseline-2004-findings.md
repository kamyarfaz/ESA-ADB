# Frozen 2004 baseline alarm audit — 22 September 2026

## Method

Audited the saved eight-channel pseudo-anomaly Transformer AE, epoch 35. Recomputed
all numeric saved event metrics to absolute tolerance 1e-12, verified the prediction
mask equals scores above the frozen threshold (3.132171630859375; gap 0, duration 1),
and matched all alarm timestamps to the previously verified channel attribution.
No thresholds, weights, masks or benchmark test data were changed or used for tuning.

## Findings

There are nine wholly false alarms, spread over May through December:
channel 41: two; 42: one; 43: two; 44: two; 45: one; 46: one.
They last 960–4800 seconds (16–80 minutes). The shortest distance to any raw
annotation, including neutral categories, is 66,929.571 seconds (18.59 hours).
This rules out immediate annotation-boundary spreading as their explanation;
it does not prove that the underlying telemetry is physically nominal.

False-alarm peaks span 3.2740–4.2330, or 1.045–1.351 times the frozen threshold.
Five event-overlapping alarm peaks span 3.6307–4.7220, while the sixth is 114.6677.
Thus peak magnitude overlaps substantially. A cutoff above every false-alarm peak
would also eliminate four of the six existing event-overlapping alarm intervals.
Alarm intervals are not unique event IDs; no alternative cutoff was selected or
reported as a new detector. Short durations also overlap: two event-overlapping
alarms last 48 minutes, within the false-alarm duration range.

The model detects six of ten unique annotated events. Missed IDs are id_91,
id_92, id_94 and id_97. The latter three are annotated only on channels 14, 21 and
29, outside the eight inputs. id_91 includes channel 47, which is an input.
Unobserved annotated channels limit direct coverage, though correlated signals
could still allow detection. This is not proof of an absolute recall ceiling.

Calibration contains only one selected event, with two false alarms at the chosen
rule. Its F0.5 is 0.383907; assessment F0.5 is 0.428131 (six detections, nine false
alarms, four misses). Sparse calibration is a material limitation, not evidence
that selecting a different threshold from assessment would be valid.

## Decision

Retain pseudo-anomaly training and the eight-channel baseline. This audit does
not support suppressing a single channel, raising the threshold globally, or
using a simple duration cutoff as a demonstrated fix. The prior eleven-channel
joint model improved coverage but increased false alarms substantially.

A next candidate is a separate detector for channels 14/21/29, preserving the
original eight-channel model. Such a design would need a predeclared combined
calibration rule and false-alarm budget; simply OR-ing detectors can increase
false alarms. It is a proposed development experiment, not a proven improvement
or an implemented change. The current audit cannot establish the physical cause
of the nine alarms; raw telemetry context analysis would be needed for that.

## Reproduce

```bash
PYTHONPATH=. python scripts/research/baseline_2004_alarm_audit.py \
  --output results_longrun/development/baseline_2004_audit_NEW
```

Requires the saved baseline 2004 results and the previous channel attribution
CSV. The output directory must be new. Compact audit outputs are versioned under
[tables/baseline-2004-audit-2026-09-22](tables/baseline-2004-audit-2026-09-22).
