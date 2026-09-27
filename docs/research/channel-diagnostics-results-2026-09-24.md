# Completed channels 40–47 diagnostic — 24 September 2026

Source: results_longrun/diagnostics/ch40_47_epoch10_v2. Completed result JSON
confirms frozen calibration reproduction; the command writes it only after its
final training CSV checksum check. Recomputed metrics from cached predictions
exactly match saved results. Score array is finite with shape 8 x 1,051,200.

## Findings

- Epoch 10 calibration F0.5 = 0.8821482657069778, TP6 FP0 FN4, recall0.6.
- All six event IDs annotated on the selected inputs were detected:
  id_106, id_107, id_109, id_110, id_112, id_114.
- Missed IDs id_105, id_108, id_111, id_113 have no annotation on channels40–47.
  Their maximum aggregate score on inside-interval samples is approximately
  0.138, 0.071, 0.132, 0.118 respectively, versus threshold7.65037.
  Lack of local annotation is not proof of absent signal or an absolute recall ceiling.
- Point-in-time sampling alone would misclassify the very short id_109, id_110,
  id_114 as having no observable event sample. Correct interval-overlap evaluation
  counts their detections; diagnostic attribution reproduces this behaviour.
- Zero false-positive EVENTS does not mean zero falsely alarmed normal duration:
  there are 10,030.695 seconds of nominal alarm overlap, penalized in precision.
- 1,024 alarm samples. Argmax winners: channel47 480 (46.875%), channel43 256,
  channel41 128, channel46 96, channels42/45 32 each, channels40/44 zero.
  Winner counts are not feature importance: a non-winning channel may still
  exceed threshold or affect the multivariate reconstruction.
- Median nominal training window errors differ across channels: about0.0063
  (40) to0.0680 (46), motivating normalization as a hypothesis. Training-window
  and overlap-averaged calibration distributions are not directly equivalent.

## Implication for the next experiment

With the same six detections and zero FP events, even eliminating all nominal
alarm duration yields only0.882353, versus current0.882148. Large calibration
improvements require additional detected IDs, not merely shorter alarms.
Do not train new models or add OR branches solely to optimize these four IDs.
Do not replace the global metric with a local-channel metric or add new inputs.

The planned single training-normalized-max comparison can still be investigated
using cached errors (no extra GPU inference), but 2006 alone is a weak selection
basis. Define the normalization before inspecting its performance and require
chronological checks with fold-specific models and training-only scale estimates.
Never apply this long-run checkpoint to earlier years and call them held-out folds.
If no robust benefit appears, retain baseline; do not start a broad ensemble sweep.
Assessment F0.5 remains0.805722; this diagnostic produced no new assessment score.

Evidence copied to tables/channel-diagnostics-2026-09-24/. Large NPZ caches and
original event plot remain in the source run. No weights, thresholds or input
channels changed. Completed GPU diagnostic supersedes earlier awaiting-run notes.
