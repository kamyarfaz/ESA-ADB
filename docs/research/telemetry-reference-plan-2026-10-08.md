# Direct telemetry reference: fixed protocol

Before execution: channels40–47, same prepared data and training-only robust
median/IQR scaler with ±10 clipping as the original AE. No neural model fitting.
For each channel, absolute deviation from the previous exponentially weighted
mean, fixed one-day half-life (2880 samples). Initialize at first training input;
update every sample including annotated ones, without label-dependent state logic.
Normalize deviations by each channel's 99th percentile on nominal training
samples (global labels==0), floored at1e-6. Score=max normalized channel deviation.
Scale fitting uses only training. No validation-fit normalization or channel choice.

For2001 train2000; for2002 train2000–01. Calibrate Jan–Jun and validate Jul–Dec
of each year. Carry reference state through both boundaries. Retain all samples.
Use existing quantiles, merge gaps0/16/64, min durations1/8/32 and corrected
system-event F0.5 selection with1% calibration alarm cap. Persist rules before
loading validation. Compare to all three saved original joint-AE seeds; reference
is deterministic, run once per fold (not three pseudo-replicates).

Advance only if reference systemF0.5 exceeds joint-AE mean in bothfolds with
no higher false-event count than jointmean in either. Report local/system metrics,
false-alarm duration, missed IDs and calibration scores. No ensemble, outerrun or
promotion. Already inspected development years, not untouched tests.

Causal score computation does not guarantee causal end-to-end preprocessing;
prepared CSV provenance and offline gap/duration filtering remain limitations.
Daily adaptation can absorb sustained anomalies. Nominal training residual
normalization can amplify quiet-channel shifts. One day and q99 fixed for this
pilot, no tuning from validation or implied optimum.
