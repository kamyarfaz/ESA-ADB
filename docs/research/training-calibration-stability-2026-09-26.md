# Training/calibration score stability

Completed26September2026 using saved verified baseline diagnostics. No new
inference/training, assessment data access, model changes or rule changes.
Frozen calibration metrics reproduced for each checkpoint; source hashes
unchanged before/after. All windows used for distributions are wholly nominal
under all prepared labels. Compare window MSE with window MSE, not aggregated
calibration scores against unaggregated training errors.

## Findings

Within each checkpoint, typical max-channel reconstruction errors remain fairly
stable while upper-tail errors vary more strongly. For the2005checkpoint:

|Nominal reference|Median window error|99th percentile|99.9th percentile|
|---|---:|---:|---:|
|Training sample,2000–2004|0.07935|0.72503|9.59542|
|January2005|0.08560|4.34129|12.13216|
|February2005|0.08497|5.92039|10.23442|
|March2005|0.08110|1.44437|10.19085|

Thus the larger nominal errors become more common relative to this training
sample, with only modest changes in typical error. The training99.9th percentile
already shows large errors; the pattern is not evidence of an entirely new
error magnitude. The2005training-year99th percentiles also vary substantially
(0.13681–1.06507), cautioning against one pooled summary.

For2004 pooled training q99=1.26611 versus January3.09858, February2.57742,
March2.07664; medians0.05707 versus0.05723–0.05842. Channel44 inJanuary has
1.61% of clean calibration windows above its sampled training q99.9, and
channel40 has1.19%. These are descriptive flags, not significance tests.
For2003 median0.05568 versus0.05442–0.05588; q99=3.29679 versus4.46606–4.99175.

## Limits and decision

Training reference contains2000 evenly spaced eligible windows per year;
calibration uses every eligible stride32 window fully within each month.
Overlapping windows are dependent; selection/sampling differs. Models and
scalers differ across folds: compare periods within a checkpoint, not absolute
scores between checkpoints. Training is in-sample; this is not an unbiased
validation comparison. Label errors or unfamiliar normal operations remain
possible explanations, not established causes. Window errors are not the final
overlap-aggregated decision score, so these quantiles must not be substituted
for frozen detection thresholds. This audit does not explain the95development
false-positive events because their periods were not inspected here.

Keep baseline and thresholds unchanged. Next focused diagnostic: inspect a
fixed, small set of the highest-error wholly nominal training/calibration
windows alongside typical windows, with raw/scaled traces and per-channel
errors. Check whether errors align with known nominal transitions or prepared
data behavior before choosing an augmentation or training change. No broad
model sweep or another threshold change justified by these tables alone.

[Plot](tables/training-calibration-stability-2026-09-26/nominal_score_stability.png)
[Tables and provenance](tables/training-calibration-stability-2026-09-26/)
Script: `scripts/research/training_calibration_stability.py`.
