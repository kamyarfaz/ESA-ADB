# id_98: local timing, channel behavior and augmentation coverage

Calibration-only review of exact annotations and cached2005 telemetry. No new
training, assessment access or threshold changes. Input hashes stable before/after.
Type metadata labels id_98 Rare Event, Multivariate, Global, Subsequence; these
are dataset taxonomy fields, not inferred spacecraft physical meaning.

## Correct local duration

The March25–28 system event includes longer intervals on other channels. On
inputs41–46, the union consists of only9minutes onMarch25:

|Start UTC|Duration|Channels annotated|
|---|---:|---|
|03:59:55.134|5min|42,45|
|04:36:55.134|30sec|41–46|
|04:56:55.134|3min30sec|41–46;46 continues30sec longer|

Grid points inside these unions:10,1,7 respectively. The 30-second event cannot
support a reliable event-only correlation estimate. Previous shorthand calling
this a multi-day anomaly described the system extent, not the input-channel
extent. Peak-to-threshold ratios recomputed using exact local intervals remain
pseudo0.815673 and reconstruction-only0.00005544.

## Traces and correlations

Plots show brief coordinated excursions on top of oscillatory channel traces;
physical cause is unknown. On the fixed surrounding context, correlations of
pairs41/44,42/45,43/46 are0.938/0.928/0.915. Same-clock previous-day nominal
reference gives0.938/0.920/0.866. Broad pair correlations remain high; this is not
strong evidence for a sustained loss of those relationships. Context correlations
include normal samples, can dilute brief effects, and are descriptive only.
Raw/scaled traces were inherited from prepared data; no claim about native timing
fidelity beyond the30-second grid or annotated physical cause.

## Mismatch with current synthetic examples

For256samples at30seconds, generator chooses12–84samples (6–42minutes), affecting
1–4 of8channels. Its three types are3–8local-standard-deviation offsets,
within-segment temporal rolls and2–5standard-deviation noise bursts. Thus it does
not directly generate the shorter durations or all-six-channel support seen in
two local bursts. Synthetic channel subsets can overlap real ones; this is a
coverage mismatch, not proof the generator cannot teach detection.

A128minute reconstruction window averages errors over256samples perchannel.
A30second local event occupies only1/256samples if aligned inside a window;
this illustrates potential dilution, not a measured causal explanation. The
previous per-timestep scoring experiment was worse; do not automatically replace
window-mean scoring or repeat that claim as a new finding.

Decision: keepbaseline. Before a new training run, inventory local event durations
and simultaneously annotated input channels across training/development annotations
to test whether short coordinated events recur beyondid_98. Then, if supported,
predeclare one bounded augmentation comparison including short durations and
coordinated channel subsets; preserve nominal reconstruction objective and test
false alarms as well as recall. Do not optimize around this one calibration event.

[Traces and tables](tables/id98-local-review-2026-09-26/).
