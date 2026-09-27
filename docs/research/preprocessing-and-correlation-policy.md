# Preprocessing and correlation review before further input experiments

## Scope

The 2000–2013 descriptive inventory remains the whole-mission overview. Input
selection for the longer baseline must instead use its training interval,
2000–2005. Calibration 2006 and assessment 2007–2013 are not read by the new
correlation script. Current model code/data and the ongoing run remain unchanged.

## Existing preparation: preserve history, specify future changes separately

| Step | Current behaviour | Decision for the review |
| --- | --- | --- |
| Raw input | Anonymized telemetry and command events | Retain all source channels; metadata alone does not establish physical meaning. |
| Channels 4–11 | Forward differences of raw counters | Trace timing and boundary look-ahead; do not interpret prepared constancy as empty raw data. |
| 30-second sampling | Forward hold | Inspect raw cadence and missed short changes; retain rare-change evidence. |
| Initial/missing values | Forward/backward filling | Quantify fill effects before designing a replacement; finite CSV values do not prove complete raw coverage. |
| Short annotated events | Label-dependent sample restoration | Preserve existing benchmark results; any label-independent operational preparation must be a separate version. |
| Nominal training | All paired label columns zero | Use the same explicit mask for correlations; never fall back to all rows. |
| Scaling/clipping | Nominal training median/IQR, clipping ±10 | Fit only on allowed training data; assess rare-channel/zero-IQR behaviour before changing policy. |
| Telecommands | 11 priority-selected event types in prepared CSV | Treat event timing separately from value constancy; all 698 source types remain in the inventory. |

No new cleaning transformations have been applied. The table identifies what needs
measurement and decisions, not a claim that new preprocessing is complete.

## Corrected correlation analysis

`scripts/research/training_correlation_review.py` streams prepared training data
with a strict 30-second grid and valid numeric/label checks. It computes:

- Pearson matrices from all globally nominal samples, pooled and by year.
- Pearson matrices of adjacent 30-second changes, requiring both observations
  nominal and in the same year. No differences bridge excluded periods.
- An exploratory Spearman matrix from fixed hourly nominal samples, explicitly
  distinguished from the full-resolution Pearson calculation.
- Every telemetry pair, including undefined relationships, not just high values.
- Diagnostic groups whose pairs have the same correlation sign and meet absolute
  Pearson cutoffs of 0.90 or 0.95 in all six years. Complete linkage prevents
  chains of weakly related channels from passing as mutually strong groups.
  Singletons are retained. These are review thresholds, not tuned model settings.

The chunked centered-covariance calculation is checked against NumPy Pearson on
synthetic correlated/anticorrelated signals. Constant inputs remain undefined
rather than receiving invented zero or perfect correlation values.

High correlation is not interchangeability: a relationship breaking may provide
anomaly evidence. Level correlations can reflect common drift; change correlations
and temporal stability help distinguish that from synchronized changes. Neither
correlation measure proves causality or predictive usefulness.

## Reuse before training

Compare these groups with the nine completed historical correlation/representative
variants documented in `historical-channel-review-2026-09-23.md`. The original
matrix's nominal-only provenance is not established because its legacy loader
missed the paired labels. Do not silently overwrite that matrix or reinterpret
its historical experiments as having used the corrected selection procedure.

Freeze explicit input sets and preprocessing versions before small matched
comparisons. Include coherent correlated groups as candidates alongside the
baseline, broad input set and representative/complementary inputs. Shortlist
using training/calibration evidence; do not rerun all old variants or choose the
next model by historical test scores.

## Completed training-period review — 23 September 2026

Processed 6,312,960 samples in [2000-01-01, 2006-01-01), including
5,615,041 globally nominal samples. All 76 telemetry channels and 2,850 pairs
are represented. The nominal mask checks all 87 paired labels.
Evidence: `tables/training-correlation-2000-2005-2026-09-23/`; figure:
`outputs/training-correlation-2000-2005-20260923/yearly_correlation.png`.

At the strict 0.95 cutoff in every year, complete-linkage groups include:

- 15, 22, 23, 31, 39
- 16, 24, 32, 33, 40
- Pairs 62/63, 20/27, 30/38 and 14/21

At 0.90, the previously tested specialist 14/21/29 remains a coherent group.
Its poor historical assessment performance demonstrates that correlation alone
does not establish detection usefulness. Channel 44 pairs with 41 at this cutoff,
even though the old minimum-size-three grouping omitted it. Singletons remain
available; none are declared useless.

Pooled relationships can hide temporal changes: 40/47 has pooled Pearson 0.8833
but only 0.2518 in 2001. The fixed baseline 40–47 is therefore not one uniformly
strongly correlated group. Existing broad clusters should not be assumed stable
across years. The separate adjacent-change matrices are available for follow-up
on candidate groups; no input set has been selected by these diagnostic cuts.

Prepared nominal constancy affects some counter channels 4–11, depending on year.
This is not evidence that their raw signals contain no information; see the counter
preprocessing trace. Any counter representation changes need separate measurement.

Validation: synthetic chunked Pearson agrees with NumPy to 1e-12; production
matrices pass symmetry and bounds checks; both groupings retain each of 76 channels
exactly once; pair and nominal-row totals match. The source size/mtime did not
change during the run. The heatmap was visually inspected. This audit neither
trains a model nor changes model package code or prepared datasets.
