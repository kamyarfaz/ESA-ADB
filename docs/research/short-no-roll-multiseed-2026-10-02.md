# Removing short temporal rolls: three-seed results

## Fixed comparison

Both arms use channels40–47, five epochs, 25,000 nominal windows, batch64 and
seeds42/43/44. Each fold trains on preceding years, calibrates January–March,
and assesses the same real April–December telemetry and event annotations.
Only the short-corruption branch changes: offset/noise replace offset/noise/roll.
The legacy branch still includes rolls. These are previously inspected development
periods, not an untouched test or the full-period2007–2013 experiment.

## System F0.5, mean and sample standard deviation across seeds

| Assessment | With short rolls | Without short rolls | Mean difference |
| --- | ---: | ---: | ---: |
| Apr–Dec2003 | 0.836292 ± 0.123532 | 0.918525 ± 0.074272 | +0.082233 |
| Apr–Dec2004 | 0.241707 ± 0.123166 | 0.458731 ± 0.248442 | +0.217024 |
| Apr–Dec2005 | 0.931322 ± 0.065321 | 0.953285 ± 0.024076 | +0.021963 |

Six paired increases, two decreases and one tie. The2003 seed43 increase is
only0.000065 with unchanged event counts. Do not present all increases as large.
These means are not pooled event metrics. Local candidate means are
0.918525/0.499582/0.939362 for2003/04/05.

## Remaining failure

The2004 candidate scores by seed42/43/44 are0.526134/0.666518/0.183543.
False alarm events are3/1/32 versus control9/45/27. Detections are4/4/6
versus control5/6/6. Mean false alarms fall27 to12, but mean detections fall
5.67 to4.67. Candidate calibration F0.5 is zero for seeds42/43 and0.093391
for seed44. Seed44 selects epoch3. This does not establish robust calibration
or achievement of the0.85 target.

## Verification

All18 saved predictions exactly reproduce threshold/postprocessing from the
frozen rules. All36 local/system metric rows reproduce within1e-9. All histories
contain epochs1–5 and checkpoint choices follow the earliest-best calibration
rule. Protocols are identical except seed. All nine control score arrays and
predictions exactly match the earlier corresponding short-coordinated runs.
No model retraining, prediction alteration or threshold reselection occurred.

Sources: results_longrun/development/short_no_roll_seed{42,43,44}_v1.
Compact records: tables/short-no-roll-multiseed-2026-10-02/.
Excel supplement: outputs/short-no-roll-20261002/Short_anomaly_comparison_2026-10-02.xlsx.
The30 September full-method workbook remains an earlier snapshot.

## Decision and next step

Retain the existing full-period baseline. Removing short rolls is promising
relative to the short-coordinated control, but not sufficiently stable for
automatic promotion or a new full-period training run.

Next diagnose2004 using only training and calibration observations: compare
seed44 against42/43 at matched epochs, inspect normal reconstruction-error tails
and false calibration alarms by channel, and separate checkpoint variability
from threshold transfer. Keep real assessment periods fixed. Do not tune rules
against the assessment predictions or choose a favorable seed. No new method
or broad training sweep is authorized by this result review.
