# Completed longer baseline — 24 September 2026

Run: `results_longrun/long_baseline/ch40_47_train2000_2005_seed42`.
Transformer AE with pseudo-anomalies, channels 40–47, seed 42. Training:
2000–2005, all 349,461 eligible nominal windows; calibration: 2006;
continuous frozen assessment: 2007–2013. All 40 epochs completed.
Calibration selected epoch 10 (F0.5 0.882148); epoch 40 calibration was 0.357143.
Lower training loss did not imply better event detection.

Pooled corrected ESA event F0.5: **0.805722**; precision 0.884871;
recall 0.593407. Detected 54 of 91 events, missed 37, with 7 false-positive
events. No uncovered assessment samples. Metric: esa-ew-id-duration-v1.

| Assessment year | F0.5 |
| --- | ---: |
| 2007 | 0.812198 |
| 2008 | 0.713877 |
| 2009 | 0.908721 |
| 2010 | 0.781121 |
| 2011 | 0.813754 |
| 2012 | 0.657728 |
| 2013 | 0.899896 |

Annual values use the same frozen model/rule, not yearly retuning. Pooled score
is not an arithmetic mean; cross-year events can appear in multiple annual slices.
The pooled 0.85 target remains unmet. Historical paired-scoring result 0.761655
is numerically lower by 0.044067, but training duration/window budget, calibration
interval and checkpoint selection differ; this does not isolate a causal benefit
of more training data. Benchmark test results/covariates were previously inspected;
this is not an untouched independent test.

Verified saved history contains epochs 1–40, best calibration epoch agrees with
frozen selection, and saved assessment reports zero uncovered samples. Compact
original evidence copied to `tables/long-baseline-2026-09-24/`.

Next: complete training-only preprocessing measurements and use corrected yearly
correlations plus historical experiments to freeze a small input comparison.
Do not tune against the newly inspected assessment years or launch another broad
sweep. No new training launched as part of this result review.
