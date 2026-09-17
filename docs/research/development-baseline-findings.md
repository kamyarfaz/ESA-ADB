# Development baseline diagnosis — 2026-09-17

## Verified results

Recomputed event-wise metrics from the saved assessment masks and raw annotations.
F0.5 and TP/FP/FN counts matched the saved results within 1e-12. No thresholds,
weights, or predictions were changed. The benchmark test set was not used.

| Fold | F0.5 | Precision | Recall | Detected | FP events | Missed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2003 | 0.833170 | 0.833129 | 0.833333 | 5/6 | 1 | 1 |
| 2004 | 0.428131 | 0.399520 | 0.600000 | 6/10 | 9 | 4 |
| 2005 | 0.869055 | 0.999157 | 0.571429 | 4/7 | 0 | 3 |

These are development assessments of three separately trained models. A good fold
is not evidence that the benchmark target has been met. Small event counts remain
a major limitation, especially the single calibration event in 2004.

## Coverage of missed events

The baseline receives channels 40–47. In 2004, missed IDs `id_92`, `id_94`, and
`id_97` are annotated exclusively on channels 14, 21, and 29. The fourth missed
ID, `id_91`, includes input channel 47. The missed 2003 ID `id_13` includes channels
40 and 47; all three missed 2005 IDs (`id_100`, `id_104`, `id_16`) include input
channels. In total, five of eight missed events overlap input channels and three
do not. Channel coverage therefore matters, but cannot explain all misses.

Annotations outside the input set do not prove an event is impossible to detect
through correlated channels. Likewise, event-wise detection does not establish
correct channel localization.

## What drove the 2004 false alarms?

Reconstructed the peak of every wholly false alarm from its exact overlapping
256-sample windows, using the saved checkpoint and scaler on the prepared telemetry.
Each peak matched the saved score with relative tolerance 2e-4 and absolute tolerance
1e-5 (CPU inference versus the original GPU inference).

| False-alarm start (UTC dataset time) | Duration (minutes) | Peak score | Dominant residual channel |
| --- | ---: | ---: | --- |
| 2004-05-06 06:40 | 16 | 3.30575 | 43 |
| 2004-05-09 02:40 | 80 | 4.23300 | 44 |
| 2004-07-16 08:32 | 48 | 3.62293 | 41 |
| 2004-08-27 09:20 | 48 | 3.90263 | 43 |
| 2004-08-27 23:44 | 48 | 3.94786 | 41 |
| 2004-10-23 05:36 | 48 | 3.54417 | 44 |
| 2004-10-31 17:20 | 32 | 3.35806 | 46 |
| 2004-11-23 13:20 | 32 | 3.56395 | 42 |
| 2004-12-17 11:28 | 16 | 3.27403 | 45 |

The frozen threshold is 3.13217. Peaks are about 1.05–1.35 times that threshold.
Six different channels dominate these peaks; removing one channel would not address
all nine alarms. Raw feature ranges in their contributing windows span approximately
0.017–0.032 for the dominant channel. These are anonymized values, not physical
units. This establishes residual attribution, not a physical explanation: command
transitions, operating regimes, and telemetry quality have not been established as
causes. Duration multiples of 16 minutes are consistent with the scoring stride of
32 samples on the 30-second grid, not necessarily physical event durations.

## Calibration versus assessment scores

At the same frozen 2004 threshold and postprocessing rule:

| Quantity | Calibration | Assessment |
| --- | ---: | ---: |
| Fraction of all samples flagged | 0.2442% | 2.7838% |
| Fraction of nominal time flagged | 0.1996% | 0.1199% |
| Wholly false alarm events | 2 | 9 |

The overall flagged fraction increased, but nominal flagged duration decreased.
The nominal-tail comparison across the fixed calibration thresholds is mixed;
these results do not support a blanket claim of increased nominal score noise.
Different annotation coverage and event durations affect overall flagged fraction.

The large F0.5 loss is driven primarily by the event-count precision term:
6/(6+9) = 0.4, with a small duration correction giving 0.39952. Nine short false
alarms can be costly despite a low nominal-time alarm rate. Raising the threshold
using these assessment outcomes would constitute development tuning and must not
be presented as a new independent evaluation. No such tuning was performed.

## Next experiment specification

Keep this baseline fixed. Compare a compact causal patch Transformer forecaster
on the same channels and chronological folds before changing channel coverage:

- Context 256, patch size 16, hidden dimension 128, two encoder layers, horizon 16.
- Inputs strictly precede targets; score each prediction only when its observation
  arrives. Fit normalization on nominal training observations only.
- Nominal-only context/target training, no pseudo-anomaly objective in the initial
  forecasting comparison. Consequently this is a comparison of detector designs,
  not an isolated estimate of architecture alone.
- Include persistence and an MLP forecaster under the same horizon, target coverage,
  residual aggregation, and calibration protocol. Disclose information/latency
  differences from the retrospective AE.
- Keep calibration-based selection and report every assessment fold. Predeclare
  the candidate before training; do not choose horizon/channels from benchmark test.
- Evaluate adding channels 14, 21, and 29 as a separate development ablation for
  both detector designs. Its motivation comes from development errors and must be
  disclosed; it does not solve the five misses with input-channel annotations.

The current AE calibration grid should not be enlarged merely to rescue 2004.
Sparse calibration remains a limitation for all candidates. Multi-seed confirmation
and broader calibration evidence are needed before a final benchmark claim.
The [forecasting runner](forecast-development.md) now implements this comparison.
Its threshold-only alarm policy is an explicit change from AE postprocessing to
preserve causality; it must be disclosed when comparing the detector designs.

## Local artifacts and reproduction

Verified tables are in `results_longrun/development/diagnostics_verified_2026-09-17/`:
`summary.csv`, `events.csv`, `false_alarms.csv`, per-fold `*_score_tails.csv`, and
`2004_false_alarm_channels.csv`. They remain excluded from Git.

```bash
PYTHONPATH=. python scripts/research/analyze_development.py --output NEW_OUTPUT_DIRECTORY
PYTHONPATH=. python scripts/research/attribute_development_alarms.py
```

The attribution script is specific to the named 2004 baseline and reads the
`diagnostics_verified_2026-09-17` directory. It writes the channel-attribution CSV;
the first command refuses to replace an existing output directory. Calibration
score arrays were not saved by training: the score-tail comparison uses the saved
checkpoint's calibration candidate table, rather than recomputing calibration.
