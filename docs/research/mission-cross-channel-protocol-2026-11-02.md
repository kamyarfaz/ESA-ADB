# Mission correspondence and training-only channel investigation

User authorization, 2 November 2026: preserve necessary work on GitHub, investigate
whether Mission1 and Mission2 share channels and patterns, and identify sensible
channels related to Mission1 40–47 before proceeding to training and testing.
This stage runs no neural training and changes no existing benchmark or checkpoint.

## Preservation and integrity

Source main was synchronized at 4d7069a. The five target-boundary Release assets'
live digests match the previously downloaded verification receipt. Forty-eight
remaining supporting files were additionally archived, uploaded and downloaded;
all content hashes match. See tables/mission-cross-channel-2026-11-02/preservation-receipt.json.
Both raw mission archives must match the official Zenodo version1.0 MD5s, and all
extracted telemetry archives and available metadata CSVs must match their archive
members before pickle loading. Save a new scoped integrity receipt; this does not
certify the whole historical migration copy. Preserve older artifacts and results.

## What can be matched

The authors explicitly preserve subsystem and physical-unit identifiers across
missions. Channel numbers and group numbers do not establish cross-mission identity.
Timelines are independently shifted and scaled by undisclosed factors. Consequently,
do not calculate cross-mission Pearson correlation at matching date labels or claim
matching physical periods. Group-normalized amplitudes also do not establish a
shared physical scale. Sources: [dataset v1](https://zenodo.org/records/12528696),
[author paper, Supplementary 2.3–2.4](https://arxiv.org/pdf/2406.17826).

Inventory all76 Mission1 and100 Mission2 channels. Profile Mission1 channels40–47
and all Mission2 target channels matching their (subsystem, physical unit) pairs:
subsystem5/unit4 and subsystem6/unit3. This metadata rule is fixed before reading
signal values. No event-count or assessment-score selection of candidates.
Reuse the existing all76-channel Mission1 training correlation audit as additional
context, explicitly identifying its prepared-data provenance.

## Chronology and raw sampling

- Mission1 descriptive signal data: [2000-01-01,2006-01-01), its original training.
- Mission2 descriptive signal data: [2000-01-01,2001-01-01) only.
- Reserve Mission2 [2001-01-01,2001-10-01) for later calibration and all data from
  2001-10-01 for assessment, aligned with the author's official assessment start.
  These reserved values and event outcomes must not influence this investigation.
- Pickle files contain complete histories, so deserialization necessarily reads
  the container. Slice immediately by time; do not inspect later signal values or
  produce later signal summaries. Full-file hashes are integrity checks only.
- Diagnostic regular grid: five minutes in each mission's anonymized clock,
  previous observation only, maximum age ten minutes. No backward fill, no value
  interpolation or label-dependent numerical restoration. This is descriptive
  downsampling, not the future model input pipeline; fast transients can be missed.
- Exclude each channel's annotated intervals in the descriptive training profiles,
  both at the raw source time and at the held sample time. Exclude all annotation
  categories, including rare events. No filling across excluded observations.
  Windows never extend past the training endpoint.

## Fixed descriptive analyses

Calculate pairwise-complete Pearson and Spearman level correlation, and Pearson
correlation of adjacent five-minute changes. Never difference across missing or
excluded samples or quarter boundaries. Report pooled and calendar-quarter values,
pair counts, missing coverage and constant-channel cases. Undefined correlations
remain missing. A strong stable level pair has the same sign and absolute Pearson
at least0.90 in every training quarter. Separately flag pairs whose change
correlation has the same sign and absolute value at least0.50 in every quarter.
These descriptive thresholds are not a model-advancement gate.

Cross-mission comparisons use separate signal summaries, never aligned dates.
For each channel and quarter calculate standardized quantiles (1,10,25,75,90,99%),
autocorrelations at1,6,12,72,288 grid steps, and change summaries (median and95th
percentile absolute changes divided by IQR; fraction of zero changes). Median
quarter features describe a channel. Within each matching metadata pool, report
distances using three equally weighted families: mean absolute difference of
quantiles, autocorrelations, and log1p change magnitudes plus zero-change fraction.
Rank distance only as an exploratory similarity heuristic, not a calibrated
probability, physical equivalence, or cross-mission correlation coefficient.
Do not optimize a time-warp factor or feature weights after viewing results.

Save every candidate and pair, including poor matches. Produce annotated heatmaps
and a report distinguishing same-unit/subsystem analogues, stable within-mission
correlation, and unproven direct model transfer. The actual training setup will
require a separate explicit protocol addressing channel roles, input order,
training-only scaling, window duration, split boundaries and event categories.
No Mission1 weights may be described as Mission2-trained, and no claim of improved
F0.5 follows from correlation alone. No new test assessment in this stage.
