# Where counter changes disappear

## Measured explanation, 2000–2002 training only

Channels 8–11 are not empty. Forward differences produce short pulses; most
pulses fall between the 30-second grid points. Every remaining nonzero prepared
sample is excluded by the global nominal mask (all 87 annotation columns must
be zero). Thus the nominal prepared profile becomes constant.

| Channel | Raw changes | Changes missed by 30s grid | Nonzero prepared rows | Nonzero nominal rows |
| --- | ---: | ---: | ---: | ---: |
| 4 | 172 | 133 | 126 | 2 |
| 5 | 172 | 131 | 123 | 2 |
| 6 | 393 | 333 | 142 | 2 |
| 7 | 192 | 153 | 126 | 2 |
| 8 | 292 | 264 | 28 | 0 |
| 9 | 292 | 264 | 28 | 0 |
| 10 | 75 | 67 | 8 | 0 |
| 11 | 169 | 155 | 14 | 0 |

A raw change and a prepared row are different units: one held difference may
occupy several grid rows. Counts must not be interpreted as event recall.
For channels 8–11 every surviving nonzero sample coincides with a nonzero label
on channel 44; other label overlaps also occur. Nonzero annotations can include
gaps or other categories, not only anomaly events. This association does not
establish physical meaning or prove predictive usefulness.

## Verification

`scripts/research/trace_counter_preprocessing.py` reconstructs forward-difference
values followed by forward holding on actual prepared timestamps. It scans all
3,156,480 training rows, comparing 3,156,476 rows per channel where both raw
difference endpoints are inside training. Boundary rows are excluded from the
reconstruction comparison, not silently assigned a terminal zero.

All eight channels had zero mismatches at rtol=1e-6, atol=1e-12; largest absolute
error was below 2.8e-11 (float32 values versus CSV decimal representation).
The nonzero prepared-row counts include the entire training interval. Count
partitions, finite values, annotation codes, total row count and prepared-source
size/mtime stability were checked. The script reconstructs numeric forward hold,
not annotation restoration; agreement establishes adequacy for these channels
and period, not all preprocessing or the whole dataset.

Reproduce with pandas/numpy:

```bash
python scripts/research/trace_counter_preprocessing.py --output NEW_DIRECTORY
```

Evidence: `tables/counter-trace-2026-09-23/trace.csv`, `exclusion_labels.csv`,
`protocol.json`. Exclusion-label counts overlap and must not be summed as unique
excluded samples.

## Decision

Do not remove channels 8–11 as intrinsically uninformative. Their nominal
constancy is explained by resampling and filtering. Keep them in the review pool;
no model input set, historical CSV, threshold, or training run was changed.
Channels 4–7 have only two nonzero globally nominal prepared samples each, which
also explains why the hourly nominal sample missed their variation.

Before any new pilot, finish training-only telecommand/redundancy review and
specify a separate preprocessing policy that preserves relevant short changes
without using anomaly labels to select values. Historical baseline evaluation
remains unchanged; no claim of improved F0.5 follows from this audit.
