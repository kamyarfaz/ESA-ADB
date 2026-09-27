# Baseline calibration event review

Frozen epoch-10 baseline, inputs 40–47, trained on 2000–2005; cached 2006
calibration predictions only. No assessment access, new training, or threshold
selection. This extends the existing diagnostic rather than a new experiment.

## Main correction

There are **no missed input-annotated events in this calibration period**:
6/6 detected, local F0.5 0.999737. System detection is 6/10, F0.5 0.882148.
The prior suggestion to investigate missed events *on channels 40–47* was
inaccurate for this period. All four missed IDs are annotated on channels
14, 21 and 29. This does not establish that channels 40–47 contain no indirect
signal, and is not a justification to change the user's input scope.

## Durations and frozen score margins

Threshold is 7.650369. Peak ratios below use scores on 30-second bins overlapping
the annotation interval, including short events between sample timestamps.

| Event | Detected | Annotated on inputs | Event duration | Peak / threshold |
|---|---|---|---:|---:|
| id_105 | No | No | 5.05 hours | 0.0180 |
| id_106 | Yes | Yes | 3.94 days | 1.1186 |
| id_107 | Yes | Yes | 16.29 minutes | 1.5927 |
| id_108 | No | No | 7.325 hours | 0.0093 |
| id_109 | Yes | Yes | 6.912 seconds | 1.1773 |
| id_110 | Yes | Yes | 3.912 seconds | 1.1068 |
| id_111 | No | No | 5.30 hours | 0.0173 |
| id_112 | Yes | Yes | 6.99 days | 1.4723 |
| id_113 | No | No | 4.05 hours | 0.0155 |
| id_114 | Yes | Yes | 9.912 seconds | 1.1170 |

All ten IDs have one merged interval in the saved diagnostic, so first-to-last
duration here is not a sum over disconnected intervals. Full channel lists and
peak channels are saved in `tables/baseline-event-review-2026-09-25/calibration_events.csv`.

The four missed events peak at only 0.9–1.8% of threshold. They are not near misses
under this frozen scoring rule. Threshold reduction is not justified by this
observation; no alternative threshold was tested. Four detected events have peaks
only about 11–18% above threshold, a descriptive sensitivity concern rather than
a robustness estimate.

Three detected events last less than a 30-second sample interval. The evaluator
correctly uses interval overlap. Because the detector averages overlapping
128-minute windows, these detections do not prove second-scale localization or
real-time response. Peak-channel identity is neither feature importance nor a
claim that that channel was anomalous (for example, id_110 peaks on channel47
although its annotations are on41–46).

## Implication

2006 already has perfect event recall for annotated input events. It is therefore
weak evidence for choosing another local-recall improvement. With the same six
system detections and no false-positive events, eliminating all nominal alarm
duration would move system F0.5 only from0.882148 to0.882353. Do not optimize away
this tiny difference or substitute local performance for the system objective.

Retain the baseline and pause new architectures. Any next proposal should address
robustness across the existing chronological development periods, with their own
training-only checkpoints, rather than repeatedly optimizing this nearly saturated
local calibration period. No new run or input expansion is proposed here.

## Verification

Recomputed local and system metrics from cached predictions reproduce the saved
values. Annotation channels and durations were joined by event ID; bin-overlap
peaks were calculated from the existing eight-channel score cache. No raw feature
CSV, benchmark assessment scores or GPU inference was used.
