# Controlled reconstruction scoring comparison

## Purpose

Compare two anomaly scores using the same saved Transformer autoencoder,
channel group, scaler, input windows, and forward passes:

- `window_mean`: average reconstruction error over each window, then average
  overlapping windows and take the maximum across channels.
- `per_timestep`: retain reconstruction error at each timestamp, average
  overlapping estimates of that timestamp, then take the maximum across channels.

A brief anomaly can be diluted by window averaging. This experiment tests whether
preserving its timestamp helps; improvement is not guaranteed.

Each method selects its threshold, merge gap, and minimum alarm duration using
only the last three calendar months of training data. The preferred method is
recorded in `frozen_rules.json` before test data are opened. Both test results are
reported, but the larger test score must not determine the preferred method.
The primary metric is corrected ESA event-wise F0.5, using raw event IDs and the
nominal-time false-alarm correction.

## Dataset repair completed (2026-09-16)

The repaired training CSV is installed. Two malformed digits were restored from
raw telemetry using the benchmark zero-order-hold rule:

| Timestamp | Channel | Restored value | Byte offset |
| --- | --- | --- | ---: |
| 2004-04-19 23:58:00 | channel_32 | 0.6858631 | 4500371516 |
| 2004-04-19 23:59:30 | channel_51 | 0.7693153 | 4500374716 |

Both source ZIP CRCs passed; no nearby annotation restoration or channel derivative
applies to these samples. A complete comparison verified exactly those two byte
changes. Both train and test passed full structural validation: 7,364,161 rows,
175 columns, finite features, valid paired labels, and a continuous 30-second grid.

- Original training SHA256: `9d3cbdffdcbc68d192cf981ac551a666ccbd9cc96fa8b27dc3cd41c33e8633e1`
- Repaired training SHA256: `fe6812dd5d324fe2eb347a703894551f0a03f40253742a3cd35f42d02fdb80ad`
- Test SHA256: `9990c46dba0101074a19b7e58382bee787ea9a2051230a5b1716d207d8315f6c`

The original is preserved in `data/repairs/2026-09-16/84_months.train.damaged.csv`.
That directory also contains reports and a failed initial copy with three unexpected
byte differences. The cause of those initial-copy differences is unresolved; the
installed copy passed a subsequent full comparison and checksum verification.
Structural validation does not prove every other finite value matches raw telemetry.
These local datasets and repair artifacts are excluded from Git.

## Run

Use the thesis environment from the repository root. The following checkpoint
was selected by validation score in the diagnostic baseline, not by test score.
It is a local experiment artifact, not included in GitHub.

```bash
conda activate timeeval
python -m esa_thesis compare-scoring \
  --source-run results_longrun/mission1_reconstruction_ae_sweep_optimized/20260519_155620/pa_center_ch08_40_47 \
  --output results_longrun/scoring_comparison/paired_v1 \
  --device cuda \
  --batch-size 64
```

Add `--dry-run` to check the checkpoint/output paths without loading the model or
reading the datasets. A dry run does not verify CUDA or dataset integrity.
Use a new output directory for each attempt; existing directories are rejected.
If memory is insufficient, use a smaller batch size and a new output directory.
`--device cpu` is supported but is slower. This command performs inference and
calibration; it does not retrain the model.

The output contains source/configuration provenance, checkpoint SHA256,
validation candidate tables, frozen rules, scores and masks for both methods,
and `results.csv`. Dataset size/mtime signatures are provenance, not integrity
checks; the separate validator produces full dataset SHA256 values.

## Limitations

These are old weights selected using the historical metric, and the historical
test set has already been inspected. This is a development experiment, not an
independent final benchmark. Validation contains only five event IDs.
Both scoring methods are retrospective; the input window includes future context.
The existing missing-value filling policy is retained. Trailing samples not
covered by a full window keep score zero in both methods and their count is
reported. Earlier chronological folds are needed before selecting a final model.

Synthetic tests check spike localization, agreement with the original window
scoring, overlap/edge behavior, and the complete validation-to-test workflow.
No real-data GPU result is claimed by these tests.

## Dataset integrity

Before running on another copy of the dataset, validate it:

```bash
python scripts/research/validate_mission1_csv.py \
  data/preprocessed/multivariate/ESA-Mission1-semi-supervised/84_months.train.csv \
  --split train --report data/repairs/train_validation.json
python scripts/research/validate_mission1_csv.py \
  data/preprocessed/multivariate/ESA-Mission1-semi-supervised/84_months.test.csv \
  --split test --report data/repairs/test_validation.json
```

The validator reads every field, checks finite features and valid paired labels,
and verifies the complete 30-second grid and expected extent. This checks the
prepared-file structure; it does not prove every value matches the raw dataset.
Local repair details are recorded separately in `data/repairs/2026-09-16/`.
