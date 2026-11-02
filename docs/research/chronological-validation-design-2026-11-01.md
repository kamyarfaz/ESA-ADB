# Chronological validation before broader tuning

Prepared alongside the authorized target/boundary study. No additional training
campaign is launched by this document. The current2000–2005/2006/2007–2013 comparison
is retained for comparability;2007–2013 has been inspected repeatedly and is not
independent confirmation. Freezing new rules does not undo earlier exposure.

## Checks in the current study

Report already frozen2006 predictions by calendar halves and quarters, including
counts/IDs, false-alarm duration and both scopes. These expose empty/small event
populations and temporal concentration. They are descriptive calibration checks:
checkpoint selection and thresholds have already used the whole year. Do not
optimize settings per block or describe these results as cross-validation.

## Later robustness design

Use expanding chronology, with a full calibration year separate from the next
validation year. Illustrative fixed folds for the existing historical record:

| Fold | Training | Calibration | Validation |
|---|---|---|---|
| A |2000–2002|2003|2004|
| B |2000–2003|2004|2005|
| C |2000–2004|2005|2006|

These periods also informed historical development: they test stability, not new
untouched confirmation. Current2000–2005 checkpoints cannot be reused. Each fold
needs training-only scaling, nominal windows and weights, including fold-specific
masked pretraining. Do not transfer later weights or thresholds into earlier folds.

Fix targets40–47, expanded context14/21/29, one matched control and one declared
candidate before running folds. Use at least three declared seeds, pairing seeds,
windows and budgets. Report every seed/fold, not just averages. Do not count
deterministic postprocessing duplicates as independent neural replicates.

Windows must remain inside their own split. Interior overlap-score support spans
480samples/240minutes. Use a conservative512-sample/256-minute separation at split
boundaries and record excluded timestamps/events. Do not bridge alarm merging or
hysteresis across embargoes/splits. A causal operational study must separately
remove future access; current retrospective reconstruction cannot be relabeled.
Chronological splitting and explicit gaps are standard tools, but do not fix prior
human exposure, label-informed preparation or regime shift. See
[TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).

Before execution, fix budget, candidate inventory and promotion criteria. A proposed
detection-preserving gate is improved mean localF0.5 in each fold, no decrease in
mean eventTPR and no increase in mean timeFPR relative to its matched control.
Report all results when it fails. Check retained IDs per seed/fold, since equal
counts can conceal exchanged IDs. Do not substitute a favorable secondary metric
after inspecting outcomes. This criterion is a future proposal, not the current
study's frozen selection rule.

## Independent operational evidence

New telemetry or a justified held-out task is needed for genuinely new confirmation.
A new seed or a split of a previously examined year is insufficient. Pre-register
selection before new outcomes. For deployment claims, make a separately versioned
input preparation that does not restore numerical values using anomaly annotations.
Preserve the existing prepared benchmark and report that as a distinct evaluation.
The inherited preparation and target/context distinction are documented in the
[ESA benchmark paper](https://arxiv.org/pdf/2406.17826).

This design is the stopping point for broader tuning in the current bounded
inference study. No robust target or independent-generalization claim follows
from single-seed measurements on the reused benchmark.
