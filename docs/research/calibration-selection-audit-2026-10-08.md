# Calibration selection audit

## Finding

The 1% total-alarm cap restricts recall in some 2001 calibration runs, but it
cannot explain the repeated failures across both years. Removing it does not
improve the best available calibration F0.5 in any of the ten 2002 cases.
No detector rules were changed and no validation scores were loaded in this audit.

## Scope

Twenty saved selected-checkpoint calibration grids: joint AE (six), isolated AE
(six), score-change transform (six), direct telemetry reference (two). Reproduced
each selected rule, then inspected the best grid row without the cap and the
best higher-recall row within the cap. Counterfactual rows are diagnostics, not
new frozen rules. The grid is finite; absence of a candidate does not prove that
no other threshold or postprocessing setting could work. Checkpoints were not
reselected, so this is not a comparison of entirely retrained/reselected policies.

## Results

| Diagnostic | 2001 | 2002 |
|---|---:|---:|
| Cases inspected | 10 | 10 |
| Higher calibration F0.5 available without the alarm cap | 6 | 0 |
| Higher-recall grid rule available within the cap | 0 | 1 |

Examples:

- Joint AE, 2001 seed42: selected TP2/FP0, F0.5=0.6667. An uncapped
  calibration candidate has TP4/FP0, F0.5=0.8692, but alarms for about5% of
  calibration samples. Zero false events does not mean zero nominal alarm time.
- Direct telemetry, 2001: selected TP2/FP0, F0.5=0.6667. An uncapped row has
  TP3/FP0, F0.5=0.7895, but alarms for about4.74% of calibration samples.
- Isolated AE, 2002 seed42: a higher-recall row does fit under the cap, but it
  trades TP2/FP0/F0.5=0.7691 for TP3/FP8/F0.5=0.3056. The precision-weighted
  objective rejects it for a concrete reason.
- Every2002 case: removing the cap leaves the best calibration F0.5 unchanged.

The cap is on **all predicted alarm samples**, including true-event time. It is
not a1% nominal false-alarm-duration cap. Any future policy change must distinguish
these concepts and specify operational meaning before running validation.

## Decision

Keep existing rules and the full-period baseline. Do not rerun outer assessments
with relaxed thresholds. The evidence does not justify treating the alarm cap as
a universal bug or replacing F0.5 with recall to manufacture an improvement.

The useful next step is to consolidate the failed internal comparisons and the
calibration/validation event differences into a revised research protocol. In
particular, specify which anomaly types and time regimes calibration represents,
and how to assess transfer across different ones. The current two small,
already-inspected internal folds should not become an endless tuning target.
A new evaluation protocol must be declared before another model search; its
results would remain developmental where those periods were previously viewed.

## Reproducibility

All20 selections match saved thresholds, postprocessing and F0.5 within1e-12.
Source grid/decision hashes were recorded and unchanged after inspection.
No neural training, new thresholds, validation evaluation or GitHub push.

[Full trade-offs](tables/calibration-selection-audit-2026-10-08/calibration_tradeoffs.csv)
· [Verification](tables/calibration-selection-audit-2026-10-08/verification.json)
