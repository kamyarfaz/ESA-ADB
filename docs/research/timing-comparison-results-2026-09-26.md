# Full-period timing comparison results — 26 September 2026

Completed run: `results_longrun/timing_comparison/full_period_seed42_v1`.
Both arms finished five epochs. Four pooled and 28 annual metric rows exist;
the runner writes these after its final telemetry and timing integrity checks.
No fresh independent model inference was performed in this review.

Inputs40–47; training2000–2005, calibration2006, continuous assessment2007–2013.
Same five-epoch budget, seed42 and pseudo-anomaly objective; candidate adds causal
observation age and availability. Models/rules selected on calibration only.

| Arm | Selected epoch | Calibration F0.5 | System F0.5 | Input-channel F0.5 | System TP / FP / FN |
|---|---:|---:|---:|---:|---|
| Control | 1 | 0.833264 | 0.689598 | 0.782049 | 28 / 0 / 63 |
| Observation age | 5 | 0.833229 | 0.595655 | 0.652065 | 28 / 8 / 63 |

Timing context reduced pooled system F0.5 by0.093943 and local F0.5 by0.129984.
It did not increase the number of detected system events; detected event identities
may differ. Both system recalls were28/91=0.307692. Local detections were28/67
for control and27/67 for timing. The control had zero event-level false positives,
but some alarm duration extended into nominal periods, so corrected precision
was slightly below1. This distinction is intentional in the evaluator.

Both calibration decisions detected5/10 events with zero event-level false
positives. Their near-equal calibration scores did not guarantee equal later
performance. Annual outcomes vary; do not select a different model for each
assessment year after looking at these results.

Conclusion: do not promote this observation-age candidate. This is evidence about
this specific five-epoch setup and one seed, not proof timing context can never
help. Historical40epoch baseline system0.805722/local0.862907 remains a separate
reference, not a matched training-budget comparison. Target system0.85 remains
unmet. Assessment data were previously inspected and are not an untouched test.

Next: retain the baseline; investigate its missed-event patterns and checkpoint/
threshold stability using training and calibration data before another training
run. Do not tune thresholds on these assessment outcomes or start another broad
architecture sweep. Underlying intermittent integrity fault is still unresolved.

Compact results and frozen decisions: [tables](tables/timing-comparison-2026-09-26/).
