# Six nominal calibration windows: reconstruction inspection

Saved2005epoch5 checkpoint; January–March calibration only. For each month,
selected the highest max-channel windowMSE and the window nearest the median
(earliest tie), requiring all256 samples nominal under every prepared label.
Selection recorded before reconstruction. Six freshCPU reconstructions reproduce
cached per-channel MSE within rtol1e-5/atol1e-5; source hashes unchanged. This
numerical check is for diagnostic reconstruction only: no event-metric tolerance
or frozen threshold changed. No assessment access or new training.

|Highest-error window start|Dominant channel|Window MSE|Observed scaled range|Reconstructed scaled range|
|---|---|---:|---|---|
|2005-01-03 17:04|41|12.7006|−1.0000 to1.0002|−6.0695 to0.1084|
|2005-02-14 06:56|43|10.4412|−1.5000 to0.9001|−5.9887 to0.2286|
|2005-03-12 03:12|44|11.4699|−1.8334 to1.2500|−6.1600 to0.4153|

Ordinary reference max-channel MSEs are0.08560/0.08497/0.08110. The selected
high-error traces show large downward offsets and oscillatory reconstructed
outputs absent from observed inputs; several other reconstructed channels also
show offsets. The dominant channel differs by example. Observed changes in
these selected high-error channels are not obviously larger than ordinary
examples (see per-channel largest-step statistics). The narrow conclusion is
that reconstruction is distorted on these labelled-nominal examples, not simply
that unusually large telemetry amplitudes produced large squared errors.

This is a six-example calibration case study, not prevalence, causal proof,
physical anomaly interpretation or explanation of95later false-positive events.
Model objective/pseudo-anomaly separation, representation and ordinary-operation
coverage are hypotheses. Do not conclude pseudo-anomaly training is responsible
from these plots alone. Ordinary examples use separate plot scales; compare
numeric ranges as well as shape. Raw prepared ranges are in window_channels.csv;
plots show training-scaled values, not physical units.

Next bounded diagnostic: inspect whether the same reconstruction pattern occurs
in the existing reconstruction-only checkpoint on these same windows, provided
features, split and scaling provenance are compatible. If budgets/checkpoint
selection differ, treat that as an exploratory comparison, not causal ablation.
Review pseudo-anomaly objective alongside that evidence before commissioning
new training. Keepbaseline and thresholds unchanged.

[January plot](tables/nominal-windows-2005-2026-09-26/month_01.png)
[February plot](tables/nominal-windows-2005-2026-09-26/month_02.png)
[March plot](tables/nominal-windows-2005-2026-09-26/month_03.png)
[Selection, statistics, verification](tables/nominal-windows-2005-2026-09-26/).
