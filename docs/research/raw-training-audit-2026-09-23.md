# Training-data audit — 23 September 2026

## Research decision

Retain the eight-channel Transformer baseline and pause model experiments.
The calibration-selected combination produced assessment F0.5 of 0.499681,
0.428131 and 0.869055 in 2003, 2004 and 2005. In 2003 it added five false
alarms without detecting another event. In the other folds calibration disabled
the specialist. Do not override those decisions using assessment outcomes.
The completed results are preserved in tables/raw-training-2026-09-23/combined_results.csv.

## Audit scope

`scripts/research/raw_training_audit.py` reads trusted local Mission 1 telemetry
pickles and measures only [2000-01-01, 2003-01-01). Raw statistics include annotated
periods; the joined earlier prepared-data profile uses nominal periods only.
It checks all 76 telemetry channels sequentially, without modifying source data.
The 11 prepared telecommands are included in the review table but their raw
encoding audit remains pending. Source size/mtime checks detect concurrent changes;
these are not a cryptographic dataset-integrity guarantee.

Reproduce in an environment with pandas and numpy, using a new output directory:

```bash
python scripts/research/raw_training_audit.py --output research_workspace/raw_training_audit_new
```

## Preprocessing findings

From `notebooks/data-prep/Mission1_semisupervised_prep_from_raw.py`:

- Channels 4–11 become forward differences before the temporal split. Constant
  prepared values therefore do not prove constant raw counters. The last
  pre-boundary difference can incorporate the next raw observation: this needs
  explicit boundary review before a strict causal or leakage-free claim.
- Values are held forward onto a regular grid. Large raw intervals are descriptive;
  asynchronous sampling means a long interval alone does not prove packet loss.
- Some short annotated anomalies are restored into the resampled series using
  annotation labels. This is label-dependent preprocessing inherited from the
  preparation script, not a deployable online transformation without labels.
- Final forward/backward filling hides original missingness. Backward filling
  can use a later observation. A finite prepared CSV does not establish raw completeness.

These are source-code observations, not a quantified reconstruction of every
prepared cell. Preserve historical results and preparation; investigate a separately
versioned transformation before changing future experiments.

## Channel decisions

`channel_review.csv` is a review queue, not a final selected subset. Nominal-training
constant inputs are removal candidates only after checking rare events and
transformations. Coarse samples can hide useful rare changes. Correlation alone
is not a removal rule: breaking a normal relationship can be an anomaly signal.
Metadata is anonymized; physical meanings cannot be inferred confidently.

Next: inspect training-period full-resolution traces and annotation coverage for
constant/rare channels, quantify fill and derivative effects, then justify at most
two small pilot subsets. No new training or tuning was started by this audit.
Presentation workbook remains the older snapshot until presentation preparation
is requested. New specialist/combined binary artifacts still need a release backup.

## Measured raw telemetry checks

All 76 channels completed; the joined review contains 87 unique inputs.
Duplicate timestamps: 0; backwards steps: 0; non-finite values: 0.
Median positive sampling intervals range from 6 to 900 seconds.
76 channels have at least one interval over one hour; the largest is 159672 seconds. These intervals require context, not automatic imputation or deletion.
