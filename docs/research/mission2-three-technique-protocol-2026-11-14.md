# Mission 2: three-technique generalization study — 14 November 2026

Authorized by the user: run all three techniques and send progress every three
hours. This supersedes the earlier outcome-only hold. No Mission 1 retraining,
direct weight transfer, MOSAD or candidate sweep is authorized by this protocol.

## Fixed question and inputs

Compare fresh short-no-roll, masked pretraining followed by short-no-roll, and
short-no-roll with three additional Mission 2 inputs. Use seeds 42,43,44 for each:
nine sequential runs. Base inputs in order: 58,59,75,76,77,83,84,85, chosen by the
published training-only correspondence study. Selected-channel evaluation always
uses this same eight-channel scope, even for the expanded model. Whole-system
events are secondary. Three repeats are limited variability evidence.

The third method adapts expanded-input short-no-roll; Mission 1's 14/21/29 are
not copied. Select exactly three additional numeric target channels from the
same subsystem/unit pairs as the base, excluding base channels. Use only the
verified Mission 2 2000 five-minute descriptive grid. For each candidate and base
anchor compute pooled and four quarterly Spearman correlations. Eligible pairs
have at least 500 observations per quarter, finite coefficients and consistent
nonzero sign. Rank each candidate's best anchor by highest minimum absolute
quarterly coefficient, then absolute pooled coefficient, then lowest anchor ID.
Rank candidates by that quality, then lowest candidate ID; retain top three.
Save all candidates and independently compare coefficients with stored matrices.
Stop if fewer than three eligible candidates; never substitute using assessment.
Redundancy is allowed and disclosed: this is a correlated-input extension, not
proof of incremental information. Additional inputs also supply alarm evidence.

## Data, chronology and preprocessing

Use raw Mission 2 v1, verifying every extracted channel/metadata hash against the
previous official-archive-verified integrity receipt before deserialization.
Training [2000-01-01,2001-01-01); calibration [2001-01-01,2001-10-01);
assessment [2001-10-01, common end of the eight base-channel records]. The end
uses timestamps only, not signal values or labels. Do not use 2007–2013 for M2.

New numerical preprocessing is label-independent: an 18-second grid within each
split, causal last observation from that split only, no backfill/interpolation,
no restoration of values from anomaly labels. Observations older than 600 seconds
and nonfinite observations become missing. Fit per-channel median/IQR using only
globally unannotated training rows with all study inputs present, require >=100
such rows; reject constant/invalid scale rather than silently choose a substitute.
After scaling, missing values map to training median (zero), clipping at +/-10.
Save availability and observation age. Missing rows cannot enter nominal training
windows; assessment remains continuous and is not censored for missing inputs.
This preparation differs from inherited annotation-restored Mission 1 data.

All training annotation intervals (any category, any channel) exclude overlapping
grid bins and windows; no test annotation enters inputs or model selection. All
methods share the same conservative all-study-input nominal window mask. Windows
are contained within each split; no training/calibration/assessment overlap and
no carry of a prior-split observation. No additional embargo: split containment
prevents shared windows, but temporal dependence is still possible. Raw files
contain whole records; slice before numerical statistics. Full-file hashing is
integrity verification, not assessment-based selection.

## Training and selection

Reuse existing Transformer AE architecture and short-no-roll augmentation.
40 epochs per run; masked method uses 20 nominal masked-pretraining epochs then
20 unmasked short-no-roll epochs, transferring shared weights and resetting the
optimizer. Window256, training stride16, scoring stride32, batch64, AdamW LR1e-4,
weight decay1e-4, gradient clip1, cosine ending1e-6. All eligible nominal windows,
no subsampling. The 18-second cadence gives a 76.8-minute nominal window length;
this differs from Mission 1, and is not a same-physical-time transfer experiment.
Each run starts fresh. Checkpoint optimizer/scheduler/RNG and partial epoch state
every100 batches and at epoch boundaries. Permute windows by seed+epoch.

Evaluate checkpoints every5 epochs (only25/30/35/40 for masked). Choose solely by
corrected selected-eight-channel Anomaly+Rare Event calibration F0.5. Retain
existing11 quantiles, gaps0/16/64, minimum durations1/8/32, <=1% total sample
alarm fraction, established rule tie order and earliest checkpoint tie. No new
postprocessing search. Fail explicitly if selection is undefined. Freeze all
nine model/rule selections together before any assessment inference.

Score window-mean reconstruction errors, float32 overlap averaging, maximum over
all model inputs. Append a final split-contained scoring window if needed to
cover the tail. This is retrospective reconstruction, not causal detection.

## Assessment and delivery

Apply each frozen rule continuously over assessment before slicing reports.
Report annual partial/full calendar periods plus pooled, both selected-eight and
system scopes: F0.5, precision, event TPR, nominal-time FPR, TP/FP/FN and all
detected/missed IDs. Primary is Anomaly+Rare Event. Secondary anomaly-only uses
identical predictions and treats Rare Event intervals as neutral for false
alarms, per the corrected evaluator's nonselected-category convention; it is
not a claim of recognizing/rejecting rare events as normal. Out-of-scope
annotations are also neutral. No reselection for any secondary breakdown.

Preserve all seeds and failures; compare methods on identical periods/scopes.
Mission 2 has a different event composition, so a higher F0.5 does not establish
superiority over Mission 1. No best-seed reporting or assessment tuning.
Before claiming results, independently replay grids/selections, checkpoint
inference spots, complete predictions, metrics and IDs; verify source, data and
runtime fingerprints. Publish verified W&B evaluations, academic report and
approved workbook preserving history; compact results to Git and essential
artifacts to a Release with fresh-download checksums. Freeze implementation
and runtime before training. New service: esa-mission2-study.service; output:
results_longrun/mission2_three_technique_20261114_v1. Resource failure requires
inspection and unchanged checkpoint recovery, never altered frozen settings.
No automatic restart loop or duplicate worker. Disable completed service and
delete three-hour follow-up only after verified delivery.
