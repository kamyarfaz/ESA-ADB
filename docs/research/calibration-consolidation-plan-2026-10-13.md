# Calibration and generalization consolidation

Declared 13 October 2026, before the new diagnostic results below are generated.

## Question and scope

Does the current calibration objective depend on too few distinct temporal
events to justify selecting a checkpoint and threshold? Existing audits show
only two detected calibration IDs per internal fold, but IDs may overlap in
time. The next work measures diversity and selection sensitivity without
training or selecting a new detector on benchmark test outcomes.

The user authorized autonomous setup and research continuation on 13 October.
The old MLP array with a checksum mismatch remains preserved and excluded from
new analyses. This work uses independently checked annotations and completed
internal calibration artifacts. It neither resolves that historical discrepancy
nor certifies a lossless full transfer. No GitHub actions or deletion are planned.

## Analysis A: annotation diversity

Use raw annotations strictly before 2007. Describe all calendar years 2000–2006,
the existing 2001/2002 half-year internal folds, the existing 2003–2005
quarter-year calibration folds, and the established 2006 calibration year.
Do not inspect 2007–2013 data or choose new split dates from the results.

For each period and system/input-channel scope, report ID counts, categories,
classes, channels, interval-union durations, and connected components of IDs
whose actual annotation intervals overlap. A component is a temporal dependence
warning, not proof of a shared physical cause or statistical independence.
Do not replace the corrected ESA metric's ID counting with component counting.
Compare calibration class/channel coverage with subsequent development periods;
these have already been inspected and are explicitly developmental.

## Analysis B: calibration event influence

Use the six completed joint-control runs (2001/2002, seeds 42/43/44), all five
saved epochs, original threshold candidates, postprocessing and 1% total-alarm
cap. Verify input hashes and reproduce saved grid metrics and selected rules.
Do not load subsequent validation scores or predictions during this analysis.

For each calibration temporal component, make its IDs neutral while preserving
its annotated intervals for nominal-time accounting. Recompute corrected
event metrics on the remaining target IDs; keep the candidate grid, alarm cap
and tie-breaking unchanged. Select the best rule per epoch, then the earliest
best epoch as the original policy does. Report how checkpoint/rule and retained
event detection change. Also record whether the withheld component is detected
by that hypothetical rule, without adopting it.

This is a label-influence diagnostic, **not cross-validation**: nominal background,
calibration score distributions and candidate thresholds are shared. It provides
no independent performance estimate. If removing a component leaves no target
events, mark the case uninformative instead of presenting zero as evidence.

Stop after this fixed audit. If evidence is dominated by one component or
selection becomes uninformative, reject further threshold-only tuning on these
folds as a well-supported improvement strategy. This is a diagnostic stopping
rule, not a statistical test or a guarantee of future success. No new model is
promoted by this analysis, regardless of its diagnostic score.

## Subsequent protocol

Keep the three frozen reference files unchanged. All-channel event evaluation
is primary; input channels remain 40–47 until a separate training-only justified
input comparison is specified. Research review must distinguish score separation,
nominal distribution drift, event coverage and selection variance. A proposed
method must address one measured cause, have a matched control, and declare its
budget and stopping rule before execution. Previously viewed periods cannot
serve as independent confirmation. New training is not part of this audit.
