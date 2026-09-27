# Supervisor request: input-channel and system-level detection

Federico requested reporting detection of events on the model input channels and
on the complete system. His message says41–47; the existing model/pilot uses40–47.
Preserve the actual input list in all reports. Do not silently remove channel40,
retrain or claim a seven-channel experiment. Other-channel input exploration remains
paused; system-wide annotation evaluation does not add model inputs.

## Same fixed predictions, two event scopes

| Period | Event scope | F0.5 | TP | FP | FN | Total IDs |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2006 calibration | Input channels40–47 | 0.999737 | 6 | 0 | 0 | 6 |
| 2006 calibration | Entire system | 0.882148 | 6 | 0 | 4 | 10 |
| 2007–2013 assessment | Input channels40–47 | 0.862907 | 53 | 7 | 14 | 67 |
| 2007–2013 assessment | Entire system | 0.805722 | 54 | 7 | 37 | 91 |

For assessment,24IDs have no annotations on input channels; one was detected,
23missed. All53input-scope detected IDs were also detected system-wide. In this
run there are no extra system detections of locally missed IDs due only to another
channel's interval of the same ID. This attribution is descriptive; no assessment
scores selected thresholds, models or channel subsets in this reporting step.

## Exact scope conventions

System: original corrected metric on all supplied raw annotation IDs/intervals.
Input: the same metric on ID unions restricted to rows whose Channel is40–47.
Annotations outside scope remain present as neutral intervals, as do communication
gaps; they are not relabeled nominal false-alarm periods. False-alarm duration is
therefore measured on the same globally nominal time in both reports. This is a
category/annotation selection convention, not a per-channel localization metric.
One global prediction vector is used unchanged in each scope.

Both scopes retain Anomaly and Rare Event categories, identical model weights,
threshold and postprocessing. No separately optimized local threshold. The local
score above0.85 must not replace the original system score or be presented as a
new improvement. System target remains unmet; exposed benchmark limitations remain.

## Evidence and future reporting

Script: scripts/research/dual_scope_report.py. Saved compact predictions-derived
results, per-ID outcomes and protocol: tables/dual-scope-2026-09-24/.
System metrics reproduced prior saved results within1e-12. Prediction files hashed
before/after reads. A synthetic test verifies that an alarm on another channel's
interval can detect a shared ID globally without detecting it locally, and remains
neutral rather than false-positive locally. Test passed. No package source changes
were made for this addition, preserving a possible running pilot's source hashes.

Add both scopes to subsequent masked-pilot reporting after its decisions are frozen.
Do not automatically change that pilot's training, model-selection objective or
input channels. The local/global distinction is required in presentations and
future result tables.
