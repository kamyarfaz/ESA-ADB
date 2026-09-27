# Existing checkpoint comparison on six nominal windows

Compared baseline pseudo-anomaly and reconstruction-only2005 models on the exact
six windows selected in the previous diagnostic (highest and median nominal
max-channel error in each Jan–Mar month). No new training or assessment access.

## Compatibility

Both selected checkpoints are epoch5 after a40epoch run, seed42, batch64,
float32, channels40–47, same2005split and checkpoint selection policy. Stored
median/IQR scalers are exactly equal. Architecture/data source hashes match;
development.py hashes differ. Recorded architecture, LR, clipping and250000window
cap agree. Training objectives and random-number consumption differ; actual
training sample/order identity has not been verified. This is an exploratory
comparison of existing checkpoints, not a fully controlled causal ablation.
Input file hashes are stable before/after; strict state-dict loading succeeds.

## Observed maximum channel MSE per window

|Month|Selected window|Pseudo-anomaly|Reconstruction-only (rounded)|
|---|---|---:|---:|
|January|Highest error|12.700634|0.000003|
|January|Typical|0.085600|0.000002|
|February|Highest error|10.441171|0.000005|
|February|Typical|0.084973|0.000002|
|March|Highest error|11.469875|0.000002|
|March|Typical|0.081097|0.000003|

Plots show reconstruction-only outputs closely tracking the observed signal;
large negative, oscillatory outputs occur in the pseudo model on these examples.
This supports investigating the pseudo-separation objective but does not prove
it caused the behavior. Windows were selected using pseudo-model errors, which
biases this comparison toward its failures. Six windows do not establish general
model superiority. The architecture gives decoder queries access to input tokens;
very small reconstruction errors alone do not establish useful anomaly detection.

The current pseudo objective minimizes normal MSE plus a hinge encouraging
corrupted-window MSE to exceed a multiple of detached normal batch MSE. Poor
reconstruction of synthetic anomalies is deliberate; poor reconstruction of
nominal windows is not. Whether the objective transfers this behavior to normal
inputs remains a hypothesis requiring a controlled diagnostic/ablation.

Next bounded step: compare both existing models on fixed annotated calibration
event windows as well as nominal references, using their own frozen scaling and
threshold conventions. Check separation of normal/anomalous scores before any
new training or claim that removing pseudo anomalies improvesF0.5. No model or
threshold replaced. Do not tune against2007–2013.

[Plot](tables/nominal-reconstruction-comparison-2026-09-26/dominant_channels.png)
[Full per-channel errors and provenance](tables/nominal-reconstruction-comparison-2026-09-26/).
