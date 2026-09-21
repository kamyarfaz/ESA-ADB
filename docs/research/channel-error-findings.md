# Channel reconstruction-error diagnosis — 21 September 2026

## Method and scope

Compared the frozen eight-channel and eleven-channel AE checkpoints on all three
development folds. For each fold, sampled 128 nominal training points with seed 42,
using the same points for both models. Every overlapping window contributing to
each reference point is nominal across all supplied label channels. These reference
scores use the existing per-window time-mean residual and overlapping-window average,
then retain individual channels before the maximum operation.

Also reconstructed every saved assessment alarm peak, using its exact contributing
windows and saved scaler. All reconstructed maxima matched saved scores within
relative tolerance 2e-4 and absolute tolerance 1e-5 (CPU versus original GPU
inference). Per-fold official-style event metrics were rechecked within 1e-12.
No models, thresholds, or prediction masks were changed. No benchmark test data
were accessed. These samples diagnose error scales; they are not a fitted new
normalizer or a reliable estimate of extreme-tail probabilities.

## Main findings

| Fold / model | Wholly false alarm events | Channels dominating their peaks |
| --- | ---: | --- |
| 2003 baseline | 1 | 43: 1 |
| 2003 expanded | 0 | None |
| 2004 baseline | 9 | 41: 2; 42: 1; 43: 2; 44: 2; 45: 1; 46: 1 |
| 2004 expanded | 33 | 41: 9; 42: 6; 43: 7; 44: 2; 45: 3; 46: 6 |
| 2005 baseline | 0 | None |
| 2005 expanded | 35 | 14: 8; 21: 27 |

The 2004 false-alarm increase is expressed in the original input channels, not
directly in maximum errors of the new channels. The expanded model is separately
trained and has a different selected epoch (40 versus 35); attributing all change
to one cause is not justified. Its threshold is also slightly higher, 3.5018
versus 3.1322. This is not simply a lower global threshold.

In 2005, channels 14 and 21 dominate every wholly false alarm peak. Their error
scales during sampled nominal training are comparatively low:

| Expanded 2005 channel | Median nominal residual | 95th percentile residual | False alarm peaks dominated |
| --- | ---: | ---: | ---: |
| 14 | 0.0334 | 0.1710 | 8 |
| 21 | 0.0320 | 0.1863 | 27 |
| 29 | 0.0452 | 0.1845 | 0 |
| 46 | 0.1108 | 0.8295 | 0 |

Thus normalizing each residual by its nominal median/upper quantile would increase
the *relative* influence of channels 14 and 21 versus 46. This does not prove a
recalibrated normalized detector would fail, but it contradicts the simple story
that high stationary nominal-error scales alone cause these added-channel alarms.
Do not adopt normalization without a separate development comparison.

For expanded 2005, channel 21 also dominates 24 alarms overlapping selected event
annotations. These are alarm intervals, not 24 distinct detected events. Their
peak scores range from 15.64 to 21.14, while channel-21 wholly false alarms range
from 15.50 to 21.38. Peak magnitude alone does not separate these observed groups.
The selected global threshold is 15.4626. This overlap motivates examining temporal
patterns and operating conditions rather than choosing a cutoff from assessment.

## Input scaling versus residual scaling

All checkpoints already use training-only per-channel median/IQR scaling and
clipping. The shared channels have the same input scalers between baseline and
expanded runs for a fold. Channel 47 has a small input IQR (about 5e-5–7e-5), but
it does not dominate any wholly false alarm peak in these runs. A small raw IQR
therefore is not, by itself, evidence for the observed failure.

Nominal residuals remain heterogeneous after input scaling. For example, in the
expanded 2005 checkpoint the sampled channel medians range from about 0.0228 to
0.1108. Baseline/expanded residual distributions also differ on shared channels,
showing that adding inputs changes learned reconstruction behavior. The 128-point
reference is a modest diagnostic sample; expand it before fitting any tail-based
normalizer. Long-range telemetry dependence further limits independent-sample
interpretations.

## Decision and next question

Keep the eight-channel AE as the working baseline. The investigation does not
justify blindly dividing every channel score by its typical training error.
The next focused diagnostic should compare channels 14/21 during nominal training,
calibration, false alarms, and detected events, including time evolution and
clipping. That can distinguish operating-range changes from brief deviations and
inform a predeclared treatment. Do not suppress channels or choose thresholds by
assessment results and describe those same results as independent validation.
No new full training run has been launched.

## Reproduction and evidence

```bash
PYTHONPATH=. python scripts/research/channel_error_diagnostics.py \
  --output results_longrun/development/channel_diagnostics_NEW
```

Requires all six saved AE checkpoints, masks, scores, and the prepared training CSV.
The output must be new. Full local outputs are at
`results_longrun/development/channel_diagnostics_2026-09-21/`.
The compact [channel summary](tables/channel-error-summary-2026-09-21.csv) is
versioned with this report. The full alarm attribution table and checkpoint-hash
manifest are preserved in the diagnostic release attachment. This is a read-only
analysis of existing experiments, not an implementation of a new detector.
