# Similarity and MOSAD full-period comparison — interim results

18 October 2026. Similarity is complete and independently verified. MOSAD is
training with its original frozen settings; its results are pending.

All eight inputs: channels40–47. Training2000–2005; calibration2006;
assessment2007–2013. Primary scope: system events across all channels.
Corrected event evaluator; one run, seed42; same immutable data files.

| Method | Corrected event F0.5 | Event precision | Event TPR | Nominal time FPR |
|---|---:|---:|---:|---:|
| Short-no-roll baseline | 0.836070 | 0.979306 | 0.527473 | 0.000292 |
| Normal-reference similarity | 0.710676 | 0.999444 | 0.329670 | 0.000556 |
| MOSAD | Pending | Pending | Pending | Pending |

Similarity does not improve the baseline: it detects30/91 system events versus
48/91. Corrected event precision increases, but nominal false-alarm duration
also increases (109,125.612s versus57,340.011s). Event precision and duration-based
FPR measure different properties; a reduction in unmatched alarm events is not
a reduction in total false-alarm time. This method therefore does not meet the
combined objective of recovering missed events and reducing false alarms.

These scores alone do not establish the mechanism of failure. Joint distance can
be insensitive to localized channel deviations; that is a hypothesis, not a
validated explanation. No settings or MOSAD training choices were changed using
this assessment. No ensemble or retrospective threshold search was introduced.

## Similarity annual results

| Period | Corrected event F0.5 | Event precision | Event TPR | Nominal time FPR |
|---|---:|---:|---:|---:|
| 2007 | 0.694295 | 0.999612 | 0.312500 | 0.000388 |
| 2008 | 0.599869 | 0.999546 | 0.230769 | 0.000454 |
| 2009 | 0.714111 | 0.999572 | 0.333333 | 0.000428 |
| 2010 | 0.780633 | 0.998737 | 0.416667 | 0.001263 |
| 2011 | 0.555441 | 0.999537 | 0.200000 | 0.000463 |
| 2012 | 0.681682 | 0.999635 | 0.300000 | 0.000365 |
| 2013 | 0.833048 | 0.999487 | 0.500000 | 0.000513 |
| 2007-2013 | 0.710676 | 0.999444 | 0.329670 | 0.000556 |

Pooled scores are computed across the continuous seven-year prediction, not the
mean of annual scores. Verification covered all7,364,160 timestamps, artifact
hashes, finite scores, binary predictions, exact frozen-rule replay and recomputed
annual/pooled metric tables for both system and input-channel scopes.
See [verified data](tables/similarity-mosad-full-20261018/similarity_verified_results.csv)
and [verification receipt](tables/similarity-mosad-full-20261018/similarity_verification.json).

MOSAD uses46 real training-event examples plus pseudo anomalies; similarity and
the baseline have no real-positive gradient supervision. This difference and
unequal computational budgets must be disclosed. The benchmark has already been
inspected; this single-seed comparison is not independent validation.

MOSAD is running under esa-similarity-mosad.service with recovery checkpoints.
The first real-data training batch completed with finite loss5.79326. The
provisional40-epoch runtime estimate is about20days, based on an8s synthetic
batch smoke test; actual throughput may differ. The follow-up monitor will
verify final results, update the workbook, and back up essential artifacts.
