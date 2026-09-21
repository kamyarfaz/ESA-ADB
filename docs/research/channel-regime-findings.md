# Channel operating-range diagnosis — 21 September 2026

## Scope

Read channels 14, 21, and 29 through 2005 from the prepared Mission 1 training
CSV. Used the frozen expanded 2005 checkpoint's training median/IQR scaler.
Nominal samples have zero labels across every supplied annotation column.
No detector, threshold, or assessment mask changed; no benchmark test data read.

## Findings

All 35 wholly false assessment alarm intervals start between 7 and 22 October
2005. Channel 14 dominates eight peaks; channel 21 dominates 27. Channel 14 stays above its entire
2000–2004 raw training maximum throughout its eight padded alarm contexts. Its
scaled context values span 1.5887–1.7963. Channel 21 contexts span 1.6512–2.0743,
around and sometimes above its training maximum of approximately 1.7358. These
are sustained higher ranges relative to calibration.

| Channel | Nominal training median | Nominal calibration median | Nominal assessment median |
| --- | ---: | ---: | ---: |
| 14 | 0.22931 | 0.26703 | 0.31983 |
| 21 | 0.23686 | 0.27306 | 0.32491 |
| 29 | 0.24892 | 0.27759 | 0.34182 |

These are raw prepared telemetry values. Monthly summaries show earlier rises
and falls, followed by a higher late-2005 level. Channel 29 also shifts but does
not dominate any false alarm peak: range shift alone does not explain every
channel's reconstruction behavior.

**Clipping is ruled out for these three inputs:** no sample in training,
calibration, or assessment exceeds the fixed ±10 scaled clipping bounds.

Eight channel-14 false-alarm contexts and eleven channel-21 contexts contain no
30-second step above that channel's nominal training 99.9th percentile. The other
16 channel-21 contexts do. Large abrupt jumps therefore are not necessary for
these false alarms. This percentile is a descriptive reference, not a tuned
alarm rule. Contexts overlap and are not independent observations.

The 24 channel-21 alarm intervals overlapping selected event annotations occupy
similar scaled ranges (1.6512–2.0743); 16 also contain above-reference steps.
These are alarm intervals, not 24 unique events. Neither high level nor this
simple jump indicator separates the observed true-overlap and false groups.

## Interpretation and next experiment

The evidence is consistent with reconstruction failure under sustained levels
near or beyond the upper historical training range. It does not establish a physical cause,
and retrospective annotation-defined nominal samples are not online state labels.
Keep the eight-channel AE as the working baseline.

A bounded next development experiment should compare the existing Transformer
forecasting control with a version that predicts changes relative to the last
observed value. Restore that level before scoring in the existing units. This
would test whether explicitly carrying the current level forward reduces false
alarms without removing level-change anomalies from the prediction target.
Keep channel set, folds, training budget, evaluation, and calibration-only decision
selection fixed for the comparison. Compare event recall and false alarms as well
as F0.5. This is a proposed hypothesis test, not an implemented or proven fix.
Assessment folds already inspected here remain development data; they cannot
be presented as independent confirmation of a diagnosis-driven improvement.
No new training run was started.

## Reproduction

```bash
PYTHONPATH=. python scripts/research/channel_regime_diagnostics.py \
  --output results_longrun/development/channel_regimes_NEW
```

Requires the saved expanded 2005 checkpoint, prepared training CSV, and prior
`channel_diagnostics_2026-09-21/alarm_channel_errors.csv` attribution output.
The output directory must be new. Outputs include full period statistics,
monthly nominal quantiles, alarm contexts, checkpoint hash, and a range plot.
Alarm contexts extend 255 samples on either side to cover retrospective AE
window spreading; they include future observations and are not causal features.

[Period statistics](tables/channel-regime-periods-2026-09-21.csv) are versioned.
Full outputs are preserved as `channel-regimes-20260921.tar.gz` on the
[September 21 migration release](https://github.com/kamyarfaz/ESA-ADB/releases/tag/migration-update-2026-09-21).
