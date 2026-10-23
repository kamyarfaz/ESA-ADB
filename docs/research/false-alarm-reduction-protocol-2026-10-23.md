# False-alarm reduction on two fixed models — 23 October 2026

## Authorization and purpose

The user authorized calibration research and then rerunning experiments for
masked pretraining plus short-no-roll fine-tuning, followed by short-no-roll with
extra inputs. They requested scheduled supervision and interruption recovery.
This study changes alarm decisions, not neural weights. It reuses the selected
seed42 models trained on2000–2005, calibrates only on2006, and then reassesses
continuous2007–2013. It is not a fresh training replicate. The assessment has
already been inspected: this is a reused-benchmark exploratory study, not
independent validation. No guaranteed preservation of assessment recall is claimed.

Inputs remain40–47 for pretrain/fine-tune, and40–47 plus14,21,29 for the second
method. Evaluate both all-channel system events and events on40–47; use corrected
event F0.5, precision, event TPR, time FPR, detected/missed IDs and false alarm
event counts. The smaller40–47 scope does not establish system target achievement.
MOSAD remains stopped. No new architecture, real anomaly training supervision,
checkpoint selection, retraining, or test-based threshold selection is permitted.

## Research basis and bounded hypotheses

Separating model scoring from alarm decisions is supported by
[scikit-learn threshold documentation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_tuned_decision_threshold.html).
Spacecraft anomaly literature also studies thresholding and false-positive
mitigation: [Hundman et al.](https://arxiv.org/abs/1802.04431).
These sources motivate studying alarm decisions; their datasets and results do
not establish effectiveness here. We do not implement their full algorithms.

First regenerate2006 per-channel window-mean reconstruction scores from each
existing checkpoint. Record alarm segment durations, nominal alarm duration,
false alarm events, peak/dominant channels, monthly nominal alarm samples,
nominal channel quantiles, and exact detected calibration event IDs. Distinguish
sample diagnostics from the corrected evaluator's duration metrics. Review both
diagnoses before opening assessment features or selecting a final rule.

Two score families are fixed in advance: existing raw maximum across channels,
and maximum after dividing each channel by its own globally nominal2006 99th
percentile (floor1e-8). The latter tests whether unequal nominal scales cause
one input to dominate. No channels are removed or weighted using test outcomes.

For each family, candidate thresholds are score percentiles
99,99.5,99.75,99.9,99.95,99.975,99.99,99.995,100. Raw scores also include the
original threshold multiplied by1,1.1,1.25,1.5,2. Minimum durations1,8,32 samples
(30 seconds each); no gap filling. Include the original rule unchanged as a
fallback. This is a bounded extension of earlier calibration, not a claim that
ordinary threshold/duration calibration had never been tried.

A feasible candidate must retain the **exact set** of event IDs detected by
the original rule in each2006 scope, keep total alarm fraction<=1%, and not
increase false alarm event count or nominal false-alarm duration versus that
method's original calibration rule. Among feasible candidates select greatest
corrected system F0.5, then least nominal false-alarm duration, then unchanged
rule on an exact tie, then fixed candidate order. Calibration has few events;
constraints do not guarantee later-year recall. Record all candidates, including
rejected choices. If no changed rule qualifies, retain the original.

Freeze both methods' chosen rules, normalization, candidate-table hashes and
research-review hash in selection_complete.json **before** assessment. Do not
alter the grid after inspecting any new2007–2013 result. Recompute per-channel
assessment scores and require exact raw maximum-score and original-prediction
parity with the completed source run. Report both original and refined outcomes,
including failures, rather than choosing a winner on assessment and suppressing
other results. Annual slices use continuous predictions; pooled scores are
recomputed rather than averaged. Existing window-overlap and duration filtering
are offline operations; no new real-time/detection-latency claim is made.

## Integrity, execution and recovery

Source run: results_longrun/combinations_full_20261021_seed42_v1.
New output: results_longrun/false_alarm_reduction_20261023_seed42_v1.
Runner: esa_thesis/false_alarm_reduction.py.
Service: esa-false-alarm-reduction.service.
Log: /home/k_faz/project-backups/run-logs/false-alarm-reduction-20261023.log.
Run --phase diagnose first; after recorded research review, run --phase finish.
Both methods run sequentially in either phase. GPU batch64, no mixed precision.
Inference saves completed window errors every100 batches and at completion;
restart skips completed inference prefixes and verifies stage receipts.
Use a process lock; never launch duplicate workers. Memory limits constrain our
job but do not reserve shared RAM. Do not stop other users' processes.

Before each phase verify all frozen destination data hashes, source-run receipts,
original package hashes and runtime fingerprint. Pin new package/protocol hashes
for resume. The unrelated historical migration exception remains excluded;
verification here is not a whole-project transfer certification. Preserve all
original artifacts and use this new directory. If interruption occurs, inspect
the cause/resources and verify no worker before restarting the same phase.

After completion independently replay both rules and all annual/pooled metrics,
compare gained/lost event IDs against both methods' controls and the short-no-roll
reference, update the academic report and historical-preserving workbook, save
compact code/results to Git and essential artifacts to a checksum-verified Release.
Disable the service and delete the completed30-minute follow-up after delivery.
