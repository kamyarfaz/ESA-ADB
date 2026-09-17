# Causal forecast alarm confirmation — 2003 development fold

## Protocol

Frozen forecasting checkpoints were reused without retraining. Five predeclared
confirmation policies were compared: 1-of-1 (baseline), 2-of-2, 3-of-3, 2-of-3,
and 3-of-5 threshold crossings in a trailing window including the current sample.
No rule backdates an alarm. A k-of-w rule can remain positive after the last
crossing while sufficient evidence remains in its trailing window.

Thresholds use the original fixed calibration quantiles. Each model's choice
maximizes corrected calibration F0.5 under the 1% predicted-rate cap. Ties favor
less nominal alarm duration, shorter windows, fewer required hits, then higher
thresholds. All three choices were saved before any assessment scores were loaded.
Existing predictions and calibration F0.5 were reproduced as integrity checks.

This is development tuning of both threshold and confirmation, not an isolated
causal estimate of adding confirmation at the original threshold. The source
checkpoints were selected with threshold-only calibration; they were not reselected
for the new policies. The benchmark test set was not used.

## Findings

| Model | Original F0.5 | Confirmed F0.5 | Original detected / FP | Confirmed detected / FP | Selected policy |
| --- | ---: | ---: | --- | --- | --- |
| Persistence | 0.04285 | 0.22727 | 3 / 83 | 1 / 3 | 2-of-2 |
| MLP | 0.08277 | 0.23255 | 5 / 69 | 4 / 16 | 3-of-5 |
| Transformer | 0.08670 | 0.11904 | 6 / 79 | 3 / 27 | 3-of-5 |

There are six assessment event IDs. Of the original wholly false alarms, 80/83
for persistence, 68/69 for MLP, and 78/79 for the Transformer lasted one sample
(30 seconds). Confirmation addresses these short spikes but also loses detections.
Calibration F0.5 after selection was 0.71429, 0.33333, and 0.45454 respectively;
the assessment results show that these calibration gains did not transfer reliably.

The three Transformer detections after confirmation have first-overlap delays of
1,964.526 seconds (`id_13`), 2,104.788 seconds (`id_19`), and 69,034.806 seconds
(`id_89`). IDs `id_12`, `id_88`, and `id_90` are missed. These delays are measured
from the earliest event-ID onset clipped to the assessment interval. Multipart
IDs and long annotations can produce large delays; they are not merely the extra
30–120 seconds needed to collect confirmation evidence. Missing events have no
delay value and must not be excluded silently from a latency claim.

No candidate approaches the existing 2003 retrospective AE result of 0.83317.
That comparison still has different information availability and alarm policies;
it does not establish that forecasting architectures in general are inferior.

## Decision

Retain these negative results. Do not launch the remaining forecasting folds merely
to look for a better period, or relax the metric. Short causal confirmation alone
has not justified adopting this forecasting configuration. Keep the AE baseline
as the working detector while investigating training-only per-channel residual
scaling and the already identified channel-coverage gaps as separate, predeclared
development experiments. No additional training was launched by this diagnosis.

## Artifacts and reproduction

Local output: `results_longrun/development/causal_alarm_2003_v1/` contains the
protocol, frozen rules, calibration grids, results, all alarm intervals, per-ID
first-overlap delays, and prediction masks. It is ignored by Git.

```bash
python -m esa_thesis.evaluation.causal_alarms \
  --source results_longrun/development/forecast_2003_seed42 \
  --fold 2003 \
  --output results_longrun/development/causal_alarm_2003_NEW
```

Existing output directories are rejected. Tests cover no backdating, prefix
invariance, warm-up exclusion, detected-event delay, and missing-event handling.
