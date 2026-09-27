# Existing channel-selection experiments: reuse before retraining

## User direction

The supervisor's sequence is data exploration, preprocessing review, channel
correlations, then justified input experiments. Correlated groups themselves must
be considered, not only all channels or one representative from each cluster.
The user explicitly asked whether these experiments had already been performed.
This review confirms saved results, not merely planned RUN_SPECS. Do not start
another sweep or modify source files while the current long-baseline run may be active.

## What already exists

`results_longrun/channel_correlation/` contains a correlation matrix, raw/clustered
heatmaps, dendrogram and a representative-channel table. Nine correlation/cluster
variants have saved corrected recalibration results in
`results_longrun/audits/f05_corrected_baseline_2026-09-16/results.csv`, sourced
from the historical `20260519_155620` reconstruction sweep.

| Existing variant | Validation F0.5 | Corrected historical test F0.5 |
| --- | ---: | ---: |
| 6 representatives: 3,5,13,29,48,57 | 0.587364 | 0.196336 |
| 12 representatives/complements: 1,3,5,6,13,29,38,39,42,48,57,60 | 0.879903 | 0.675297 |
| Correlated group 41–52 | 0.879903 | 0.765242 |
| Correlated group 17–35 | 0.081188 | 0.052523 |
| Correlated group 12–35 | 0.587882 | 0.163255 |
| Cross-group 20–45 | 0.879903 | 0.539565 |
| Correlated group 12–40 | 0.587773 | 0.096138 |
| Group 12–40 plus inversely correlated channel 61 | 0.587937 | 0.293667 |
| 6 representatives with pseudo-anomaly objective | 0.882108 | 0.664937 |

These are corrected event-wise metrics on historical saved scores, not new training
under the current 2000–2005 / 2006 protocol and not the three yearly development
folds. The old benchmark test covers 2007–2013 plus its January 1, 2014 boundary
sample. Previous test exposure and original checkpoint provenance remain caveats.
Do not rank these test results to select the next model and then call that same
test independent. No raw masks were recomputed by this inventory-only review.

## Selection-method issues to resolve

- `analysis/correlation.py` searches for a single `is_anomaly`, `label`, `anomaly`
  or `y` column. The actual dataset has paired `is_anomaly_channel_*` columns.
  Its fallback uses all rows. Consequently the saved matrix's claimed nominal-only
  provenance needs verification; comments and plot titles are not evidence.
- `analysis/representatives.py` defaults to `min_size=3`. The saved table excludes
  singleton/two-channel clusters, including channel 44, merely because their
  clusters are small. Distinct signals are not automatically uninformative.
- Historical configuration comments explicitly reference earlier test performance
  for some choices. Preserve these runs as exploratory history, not clean
  prospective feature-selection evidence.
- The newer training-only Pearson/Spearman audit fixes paired-label filtering,
  but covers 2000–2002 and an hourly sample. It is not yet the 2000–2005 nominal
  correlation analysis needed for the current longer protocol.

## Revised sequence after the current run

1. Reuse the old channel sets, saved weights/results and plots as the starting
   inventory. Identify exact overlap before requesting any repeated training.
2. Finish the preprocessing specification and compute correctly filtered training
   correlations over the permitted interval. Check stability by year/month,
   inverse relationships and short changes; separate full-mission descriptive
   plots from feature selection. Keep original data intact.
3. Define candidates representing the reference baseline, broad inputs, a coherent
   correlated group, and cluster representatives with complementary signals.
   Do not assume a correlation threshold proves redundancy or force isolated
   channels out. Shortlist with training/calibration evidence before running.
4. Run only missing, small matched comparisons with the same window budget and
   preprocessing; advance promising alternatives to longer training. Reuse valid
   existing scores where protocols match. Do not relaunch the nine old variants
   simply because they have now been rediscovered.

No training or package-source change was made during this historical review.
