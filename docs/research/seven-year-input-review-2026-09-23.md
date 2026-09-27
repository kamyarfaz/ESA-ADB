# Descriptive input review: all seven years, 2000–2006

## Scope

The user's requested whole-period overview covers [2000-01-01, 2007-01-01):
seven calendar years, all 76 raw telemetry channels and 698 telecommand types.
It supplements rather than replaces the earlier first-fold training-only review.
No source files, model input lists or historical results are changed.

The script reads source values and metadata only: no annotation files, calibration
scores or model predictions. Annual rows report activity, variation and data-quality
checks. Telemetry also has monthly minimum/maximum/mean envelopes and annual
sampling-interval summaries. Every channel appears in the plot atlas.

## Interpretation

This is descriptive exploration, not an input-selection rule. A variable channel
is not automatically useful; a rarely changing channel is not automatically useless.
Telecommand occurrences carry timing information even with constant payload values.
A zero occurrence count describes that year only. Monthly envelopes retain extremes
but hide their exact timing and do not distinguish normal from abnormal behavior.

Yearly sampling gaps measure intervals between observations inside the year;
intervals crossing year boundaries are not counted. Long intervals are not proof
of missing packets for asynchronously recorded channels. Physical units remain
anonymized. No per-channel anomaly-detection F0.5 is inferred from these statistics.

## Evaluation boundary

2006 source values have now been examined descriptively, so **2006 is not completely
unseen**. Its anomaly labels and model scores were not inspected by this script.
It remains reserved from model tuning. Do not select features, transformations or
hyperparameters from these later-year summaries and then claim an independent
2006 validation result. If the overview guides such decisions, document that use
and revise the final evaluation protocol before making an independence claim.

For chronological experiments, input selection and fitted preprocessing must still
use each experiment's permitted training data. The existing assessment folds are
already development data. No new training is started by this review.

## Reproduction and verification

```bash
python scripts/research/seven_year_input_review.py --output NEW_DIRECTORY
```

Requires pandas, numpy and matplotlib. The script checks metadata against source
filenames, source size/mtime stability, 5,418 unique input/year records, 6,384
telemetry/month records and equality of monthly and yearly telemetry counts.
It records non-finite values, duplicate timestamps and backwards steps rather
than silently repairing them. Source size/mtime checks are not cryptographic
integrity verification. Existing corruption-recovery records remain relevant.

Tables: `tables/seven-year-review-2026-09-23/`. Atlas:
`../../outputs/seven-year-review-20260923/`.

## Verified results

All 76 telemetry channels vary in each of the seven years. No non-finite values,
duplicate timestamps or backwards steps were found in this scan. These checks
do not establish gap-free sensing or model usefulness.

| Year | Variable telemetry channels | Active telecommand types | Telecommand occurrences |
| --- | ---: | ---: | ---: |
| 2000 | 76 | 399 | 186,980 |
| 2001 | 76 | 286 | 106,578 |
| 2002 | 76 | 266 | 74,558 |
| 2003 | 76 | 228 | 76,458 |
| 2004 | 76 | 308 | 104,103 |
| 2005 | 76 | 322 | 176,823 |
| 2006 | 76 | 226 | 99,305 |

Across all seven years, 517 command types occur and 181 do not. The 2000–2002 observation totals agree exactly with the preceding audit for every one of the 774 inputs. No inputs have been excluded.
