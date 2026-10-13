# Thesis evaluation structure and research direction

13 October 2026

**Later setup update, 13 October:** the three separate reference CSVs and binary
labels have now been packaged and verified in `data/frozen/mission1-reference-v1/`.
The post-recovery transfer scan has 16 remaining differences, including one
historical array; it is not a zero-failure transfer. The original audit below is
retained as chronology. See `docs/research/frozen-reference-2026-10-13.md` and
`docs/research/transfer-recovery-2026-10-13.md` for current setup evidence.
Research was subsequently stopped at the user's request. No new training or
nominal-quantile comparison has been executed.

The required evaluation target is all-channel events. Model input channels will be selected only after research; the current eight-channel model is not an all-channel input model. No new training or model selection was performed for this review.

## Existing data structure

The established reference uses fixed chronological boundaries, enforced in `esa_thesis/long_baseline.py` and `esa_thesis/development.py`:

| Role | Interval (end exclusive) | Samples | Target event IDs | Existing physical source |
|---|---|---:|---:|---|
| Training | 2000-01-01 to 2006-01-01 | 6,312,960 | 95 | First part of `84_months.train.csv` |
| Calibration | 2006-01-01 to 2007-01-01 | 1,051,200 | 10 | Remaining part of `84_months.train.csv` |
| Testing | 2007-01-01 to 2014-01-01 | 7,364,160 | 91 | `84_months.test.csv`, excluding the final boundary row |

Both sources are under `data/preprocessed/multivariate/ESA-Mission1-semi-supervised/`. Both full-file SHA256 checks passed on this server against the established golden checksums. Each source contains one extra endpoint row; the documented half-open intervals exclude it. Together the three logical subsets are disjoint and cover 14,728,320 consecutive 30-second samples.

**Three separately frozen train/calibration/testing data files do not currently exist for this reference.** The existing code extracts the first two subsets from one file. The full migration still has 25 recorded failures unrelated to this successful two-file check; it must not be described as fully verified.

The requested final structure should use three versioned data files, a separate label file, and a manifest containing hashes, column order, timestamps, label semantics and preprocessing provenance. Before each experiment, reject any hash, boundary, schema or overlap mismatch. Fit preprocessing only on training data. Freeze checkpoint, threshold and alarm rules after calibration, before testing. The audit does not change the current source files, split dates or pipeline.

The 2007–2013 period has already been inspected. Packaging it into immutable files cannot turn it into an untouched test set. Whether final confirmation uses genuinely unexamined data or explicitly reported reuse must be resolved before calling a result independent confirmation.

## Binary labels delivered

`ESA_Mission1_binary_labels.csv.gz` contains all 14,728,320 historical reference timestamps, with four columns:

| Column | Meaning |
|---|---|
| timestamp | Original prepared 30-second timestamp, interpreted as UTC |
| split | train, calibration or testing under the existing reference boundaries |
| is_anomaly | 1 if any of the 76 channel labels is Anomaly or Rare Event; otherwise 0 |
| is_ignored | 1 for communication-gap rows with no simultaneous target anomaly; otherwise 0 |

Interpret `is_anomaly=0` as normal only when `is_ignored=0`. These are ground-truth labels, not model predictions. Telecommands are not telemetry channels and are excluded from this all-channel target. The original prepared label codes are 0=nominal, 1=anomaly, 2=rare event, 3=gap. The export preserves their existing resampling alignment rather than inventing a new annotation-to-grid policy. It does not replace the existing training-window filter.

| Split | Positive rows | Ignored rows | Normal rows |
|---|---:|---:|---:|
| Training | 693,158 | 4,761 | 5,615,041 |
| Calibration | 33,656 | 0 | 1,017,544 |
| Testing | 817,290 | 0 | 6,546,870 |

Compression keeps this file small; it can be read directly with `pandas.read_csv(path, compression='gzip')`. Use chunked reading when memory is limited. The complete file exceeds Excel's worksheet row limit.

Keep `data/ESA-Mission1/labels.csv` and `anomaly_types.csv` alongside these binary labels: event IDs and exact intervals remain necessary for the corrected event-wise evaluator. Pointwise binary F0.5 must not replace `esa-ew-id-duration-v1`.

## Concise thesis results

`ESA_thesis_results.xlsx` contains one sheet, 65 method-period rows and 12 essential columns: technique, comparison group, input channels, training years, calibration period, assessment period, run count, corrected precision, recall, F0.5, F0.5 sample standard deviation and false-event count. Multi-run rows use arithmetic means; they are not pooled event scores. The full-period baseline is a pooled assessment. Source references and experiment budgets are attached as cell notes. No seed was selected for its test score.

The table retains unsuccessful techniques, excludes individual training histories and diagnostic logs, and avoids repeated annual slices of the pooled reference. The exactly reused short-anomaly control is represented once. Secondary loss-weight/isolation checkpoint policies remain in the comprehensive historical workbook; the compact table uses their prespecified primary policy. Historical scoring has a different training split and includes a boundary sample, explicitly identified in its row. Different protocols are not a single fair ranking.

## Research decision before another experiment

Existing evidence supports prioritizing observability and calibration transfer before another architecture change:

- In the retained 2007–2013 reference, 24 event IDs had no annotations on channels 40–47; 23 of these were missed. This is descriptive evidence of a coverage limitation, not permission to choose channels from test outcomes.
- Internal event audits show calibration detects a narrow subset of events, while some later anomalies score below normal observations. Threshold changes alone cannot resolve poor score separation.
- The earlier all-input inventory already reviewed all 76 channels on training-period data. Reuse this work before repeating investigations. Input expansion or reduction must be justified using permitted training/development data and a controlled comparison.

For each proposed technique, document: (1) the observed failure it addresses, (2) relevant primary literature and its applicability, (3) the expected mechanism, (4) a falsifiable hypothesis, (5) one matched baseline comparison with fixed data/budget/seeds, and (6) advancement and stopping criteria. Prefer the smallest diagnostic that can reject the hypothesis before expensive training. Evaluate a justified candidate on the final fixed test only after development decisions are frozen. No new technique is selected or claimed to improve performance by this review.

Initial literature check: the [ESA-ADB paper](https://arxiv.org/abs/2406.17826) and [official implementation](https://github.com/kplabs-pl/ESA-ADB) are the primary benchmark references. This is protocol consolidation, not a completed literature review of a new candidate method.

Repository evidence: `esa_thesis/long_baseline.py`, `esa_thesis/development.py`, `esa_thesis/evaluation/esa.py`, `tests/thesis/test_esa_evaluation.py`, `notebooks/data-prep/utils.py`, `notebooks/data-prep/Mission1_semisupervised_prep_from_raw.py`, `docs/research/dual-scope-evaluation.md`, `docs/research/all-input-review-2026-09-23.md`, `docs/research/inner-event-audit-2026-10-08.md` and `docs/research/calibration-selection-audit-2026-10-08.md`.

Machine-readable source hashes, label counts, row provenance and verification results are saved in the sibling `build/` folder. Earlier workbooks, reports, datasets and weights remain unchanged. Regenerate the transfer manifest immediately before any subsequent copy, as required by `MIGRATION.md`.
