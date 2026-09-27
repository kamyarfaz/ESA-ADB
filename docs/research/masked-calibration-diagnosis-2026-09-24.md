# Masked pilot: 2004 calibration diagnosis

## Scope

Read-only analysis of saved training histories, frozen rules and selected-epoch
calibration grids in
`results_longrun/development/masked_pilot_2004_2005_seed42_v2/2004/`.
Inputs remain channels 40–47. Training precedes 2004; calibration is January–March
2004. No new training, raw-data loading or assessment-based tuning was performed.

## Findings

There is only **one positive event ID** in this calibration period. Both arms
selected epoch 1, correctly following the best-calibration-score rule.

| Epoch | Baseline training loss | Baseline calibration F0.5 | Masked training loss | Masked calibration F0.5 |
|---|---:|---:|---:|---:|
| 1 | 7.631369 | 1.000000 | 8.683048 | 0.171616 |
| 2 | 4.958168 | 0.199131 | 6.649272 | 0.000000 |
| 3 | 4.391293 | 0.120779 | 5.711798 | 0.000000 |
| 4 | 3.855564 | 0.000000 | 5.043074 | 0.000000 |
| 5 | 3.582060 | 0.000000 | 4.804711 | 0.000000 |

Lower training loss did not translate to better event detection for either arm.
These total losses alone cannot distinguish overfitting, the interaction of
reconstruction with the pseudo-anomaly objective, or changing score separation.

At the selected epoch, the baseline detects the event with zero false-positive
events and zero nominal false-alarm duration. The masked rule detects it with
six false-positive events and 34,560 seconds (9.6 hours) of nominal false alarms.
Across the saved masked calibration grid, every candidate detecting the event
has at least six false-positive events. Raising the threshold enough to reduce
that count loses the event. This is evidence of poor separation on this period
within the tested scoring and postprocessing grid, not proof that every possible
threshold or masked model must fail.

The baseline's perfect calibration score describes one event, so it is weak
evidence of generalization. The masked model's weakness was already observable
before assessment; its later failure is not a reason to optimize on assessment.

## Decision and next diagnostic

Retain the existing full-period baseline. Do not promote this masked configuration,
extend it to 40 epochs, or choose a different detector for each assessment year.

Before another training experiment, the useful next diagnostic is to separate
nominal reconstruction error from the pseudo-anomaly penalty and inspect their
score distributions on a fixed training sample. For the masked arm, also measure
whether synthetic corruption overlaps the hidden patches that receive loss.
These are hypotheses to check, not established causes. Use training/calibration
only and keep assessment results out of parameter selection. No new run was
launched as part of this review.
