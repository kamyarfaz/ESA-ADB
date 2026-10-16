# Missed-event and annotation-channel coverage audit

## Main finding

The candidate's 43 missed system event IDs comprise **23 events annotated only
outside inputs 40–47** and **20 events with annotations on those inputs**.
The eight detections lost relative to the baseline all belong to the latter
group. Restricted inputs may contribute to incomplete system coverage, but they
do not explain the candidate's loss of detections compared with the baseline.

| Pooled event outcome, 2007–2013 | Input-annotated | Outside inputs only | Total |
|---|---:|---:|---:|
| Detected by both | 45 | 1 | 46 |
| Baseline only | 8 | 0 | 8 |
| Short-no-roll only | 2 | 0 | 2 |
| Missed by both | 12 | 23 | 35 |
| Total unique event IDs | 67 | 24 | 91 |

The candidate gains two detections and loses eight, producing a net loss of six
(54 → 48). Its higher F0.5 reflects improved precision, not broader event detection.
Both methods miss 23 of the 24 outside-input events. One such event, id_46, overlaps
an alarm from both models; this does not establish that the models identified
its physical cause or observed its annotated channels.

## Annual diagnosis

| Year | Baseline detected | Candidate detected | Baseline-only | Candidate-only | Candidate misses: input-annotated | Candidate misses: outside inputs |
|---|---:|---:|---:|---:|---:|---:|
| 2007 | 13 | 11 | 2 | 0 | 2 | 3 |
| 2008 | 7 | 7 | 0 | 0 | 2 | 4 |
| 2009 | 8 | 7 | 1 | 0 | 3 | 2 |
| 2010 | 5 | 3 | 3 | 1 | 5 | 4 |
| 2011 | 7 | 8 | 0 | 1 | 3 | 4 |
| 2012 | 5 | 4 | 1 | 0 | 5 | 1 |
| 2013 | 9 | 8 | 1 | 0 | 1 | 5 |

Annual counts must not be summed to obtain pooled unique-event counts: some IDs
occur across years. Annual annotation coverage is computed within each year.

In **2010**, baseline-only IDs are id_140, id_149 and id_150; the candidate gains
id_44. All are annotated on current inputs. Thus the decline is not simply an
increase in events outside the input set. Of nine candidate misses, five are
input-annotated and four are outside-input events.

In **2012**, the candidate loses id_169 and gains no event. Five of six candidate
misses have input annotations. A change in channel coverage alone cannot account
for the weak result in this year.

Other baseline-only IDs are id_15 and id_121 (2007), id_40 (2009) and id_14 (2013).
The second candidate-only ID is id_157 (2011).

## Channel evidence and limits

Among the 23 outside-input events missed by both methods, 20 are annotated on
the channel group 14/21/29; three involve other outside-input channel groups.
This is descriptive benchmark evidence, **not a channel-selection rule**.
Earlier development experiments adding 14/21/29 already increased false alarms;
see `expanded-channel-results.md` and `representation-decision-2026-10-14.md`.
Do not repeat that expansion as a new discovery or infer that adding channels
would necessarily recover these events.

Annotations identify recorded affected channels, not all possible correlated
signals. An event annotated on inputs is not necessarily readily detectable at
30-second sampling or with the current score; absence of input annotations does
not prove impossibility. Temporal alarm overlap is the benchmark detection rule,
not proof of causal attribution. No raw-signal or score-separation mechanism was
tested by this audit, so it does not establish why individual detections changed.

## Research implication

Keep this assessment descriptive and leave its thresholds, inputs and model fixed.
Two separate questions need training/development evidence: coverage of additional
subsystems, and retention of useful anomaly evidence within current inputs.
Before another training run, review the existing expanded-input and missed-event
diagnostics, then examine training/development channel coverage and score behavior
to formulate one controlled hypothesis. Do not choose channels or optimize a
threshold against these 2007–2013 outcomes. A literature-supported design and
predeclared comparison remain necessary; this audit does not validate a new method.

## Reproducibility

Script: `scripts/research/audit_full_period_misses.py`.
Tables: `tables/full-period-missed-events-2026-10-16/`:

- `event_outcomes.csv`: every event's period, annotation channels, detection status
  and overlap on input-channel annotation intervals.
- `coverage_summary.csv`: annual and pooled detection transitions by coverage.
- `annotation_channel_counts.csv`: descriptive pooled counts per channel;
  overlapping event IDs mean channel counts are not additive.
- `verification.json`: dependency hashes and audit checks.

Baseline predictions, saved metrics and annotations match their original migration
hashes. Candidate predictions and saved metrics match its completion manifest.
Per-event detections reconcile with corrected evaluator TPe; all annual and pooled
system counts and metrics reconcile with published results. Seven evaluator parity
and regression tests passed. No training, threshold search or input change occurred.
The unrelated historical checksum exception remains excluded from this analysis.
