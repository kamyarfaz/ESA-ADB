# Plan to improve trustworthy F0.5 on ESA Mission 1

Research review: 2026-09-15. These are prioritized hypotheses and experiments,
not demonstrated gains. No model was trained or production metric changed in
this review. See the companion `f05-evaluation-audit.md` for measured diagnostics.

## Objective and protocol

Working target: **official corrected event-wise F0.5 ≥ 0.85** for a declared
Mission 1 test interval and target-channel scope. This is different from ordinary
point-wise F0.5, uncorrected event F0.5, channel-aware F0.5, and a Kaggle leaderboard
score. Full mission and a lightweight subset must have separate result tables.
The supplied target does not establish which of these was originally intended.

For benchmark comparability, retain the prescribed train/test boundary and use
the final three calendar months of training for validation. Tune architecture and
channel choices on earlier chronological development folds, then calibrate the
final decision rule on the reserved validation interval. Do not repeatedly choose
models using the already inspected test set. That set is now an exploratory test
set for this project; stronger confirmation requires an additional untouched
period/dataset or a clearly disclosed predeclared final protocol.

Keep annotation IDs, timestamps, categories, and target channels. Score anomalies
and rare events together for the main combined-event track, with category results
reported separately. Do not silently treat communication-gap labels as anomalies.
Retain non-selected annotations as neutral regions when required by the official
category-filtering implementation. Never merge or filter ground truth to improve
scores. Changes to prediction postprocessing are validation hyperparameters.

The benchmark specifies a chronological half/half split and three-month validation
period; it prioritizes false alarms and also evaluates channel identification and
timing. [ESA-ADB paper](https://arxiv.org/pdf/2406.17826), sections on evaluation.

## What the saved-mask audit changes about our priorities

The highest full-mission diagnostic score among the 31 primary masks is 0.7378,
with corrected precision 0.9413 and recall 36/91 = 0.3956. At the same precision,
F0.5 0.85 would require recall about 0.6123 (at least 56/91 events). That is an
illustrative requirement, not a forecast: changing detections also changes precision.
For this model, recovering missed events while preserving precision matters most.

Across all 31 configured groups, only **44 of the 58 target channels** appear as
input features. No run directly includes channels 58, 59, 62–66, or 70–76.
This does not prove their events are undetectable through correlated channels, but
it is a concrete coverage gap for a full-mission objective. Inspect missed-event
channels on development folds, then compare small complementary channel models or
a shared Transformer covering all targets. Calibrate the fused detector globally;
blind OR fusion can erase the precision advantage.

The six-channel pseudo-anomaly run instead has corrected precision around 0.7930,
recall 49/91, and a 19.33% predicted test rate. It needs substantially better
false-alarm control. A single generic instruction to increase recall or raise all
thresholds would therefore be inappropriate for the entire project.

## Prioritized experiments

| Priority | Experiment | Why it fits this code | Success criterion |
| --- | --- | --- | --- |
| 0 | Centralize official event-ID/time-domain evaluation | The additive precision correction currently affects checkpoint, threshold, and result selection | Synthetic edge cases and raw-annotation fixtures agree with pinned ESAScores; each result states protocol and metric version |
| 1 | Recalibrate existing cached scores using validation only | Cheapest way to establish what the current models can do under the right objective | Improved corrected validation F0.5 across chronological folds; frozen threshold applied to test once |
| 2 | Compare window-average versus per-timestep AE residuals using the same weights | Current scoring spreads one residual over 256 samples, weakening localization | Fewer nominal alarm seconds without losing event recall; also report detection delay |
| 3 | Replace full-series min-max normalization and raw channel maxima | Test extremes currently change ensemble scaling; channels have different residual distributions | Training/validation-fitted calibration is stable across time and reduces false-alarm duration |
| 4 | Evaluate a compact causal patch Transformer forecaster | Meets the Transformer requirement and avoids an AE directly seeing its reconstruction target | Beats current AE and MLP under the same split, calibration, and compute budget |
| 5 | Masked reconstruction or denoising objective | Forces recovery from context instead of simply copying an observed anomaly | Adds complementary detections at a fixed false-alarm budget; mask policy matches inference |
| 6 | Regime-aware features and cautious ensemble fusion | Nominal command transitions and drift may explain elevated test alarm rates | Improvement persists on separate temporal folds; rare events remain governed by the declared label policy |

### 1. Recalibration before retraining

Use the validation score arrays to select thresholds and short-duration/gap rules
with corrected precision. Compare global residual scaling with per-channel robust
scaling or empirical nominal-tail ranks fitted on training/calibration data.
These ranks are calibration features, not guaranteed p-values: telemetry is
correlated and nonstationary. Calibrate the *combined* detector's alarm rate;
a per-channel budget does not automatically control the mission-wide false alarms.

Track false alarms per day and nominal alarm duration, rather than relying only
on the fraction of all points flagged. A fixed 1% predicted-rate cap is not a
universal operational budget, and the current 1% validation rule can yield much
higher test alarm rates. Test EWMA/hysteresis and small duration filters as explicit
ablations, recording the delay they introduce. Avoid broad gap filling that makes
a detector appear to cover events while raising nominal-time alarms.

Replacing the metric after training is insufficient to fully repair the experiment:
existing best checkpoints were chosen using the wrong metric too. Cached scores
can establish an inexpensive diagnostic baseline; later retraining must select
checkpoints using the corrected protocol. Do not pick a new epoch using test scores.

### 2. Fix the score-localization experiment

At the 30-second sampling grid, 256 samples span 128 minutes and a 16-point patch
spans 8 minutes. Current `score_series` averages squared errors across all 256
positions before spreading each average back over the window. A single large
residual can therefore become a long low-amplitude alarm footprint. Compare that
with averaging *per-timestep* errors across overlapping windows.

Per-timestep reconstruction is still retrospective if it uses later observations
within the same window. For operational claims, separately evaluate a window-end
score or delayed output with explicit timestamps, and remove future-dependent
backfilling. Do not compare a retrospective detector with a causal one without
reporting their different information and delay budgets.

### 3. Transformer experiment specification

Start with a PatchTST-style forecaster, not a large pretrained language model:

- Context 256 as the first controlled comparison; 512 as a later context ablation.
- Patches 16; hidden size 64 or 128; two encoder layers initially.
- Predict 1, 16, or 32 future samples in a small predeclared sweep.
- Training uses nominal contexts/targets and training-only scaling.
- Score each forecast after the corresponding real observation arrives.
- First compare a fixed channel group used by the existing model; evaluate expanded
  target coverage separately instead of changing model and scope simultaneously.
- Compare shared channel-independent encoding with the current cross-channel design.
- Keep the current MLP and a persistence forecaster as controls. An improvement over
  a weak baseline alone is not sufficient evidence for a Transformer contribution.

PatchTST introduces patch tokens and shared channel-independent forecasting weights;
its published forecasting results are motivation, not evidence of ESA F0.5 ≥ 0.85.
[Paper](https://arxiv.org/abs/2211.14730),
[author implementation](https://github.com/yuqinie98/PatchTST).

A subsequent masked-prediction experiment is supported by MAD's self-supervised
mask-and-estimate formulation. A future-aware masking scheme must be identified as
offline, or redesigned to hide the current target using only available history.
[MAD paper](https://arxiv.org/abs/2205.02100).

Anomaly Transformer and DCdetector offer association-discrepancy and contrastive
alternatives. They are useful later comparators, not the first engineering step:
their results on other datasets/metrics do not establish ESA benchmark performance.
[Anomaly Transformer](https://arxiv.org/abs/2110.02642),
[DCdetector](https://arxiv.org/abs/2306.10347).

### 4. Nominal regimes, channel groups, and ensembles

Inspect false alarms against channel traces and telecommand timing on development
folds. Test causal command indicators, time since relevant commands, missingness
flags, and channel operating-state features. Avoid unconditional suppression after
commands: the benchmark also labels rare nominal events as positive in the combined
track. Fit channel grouping on training data; repair the correlation utility's
normal-label selection before treating its output as nominal-only correlations.

A maximum/OR ensemble can improve recall but can increase false alarms. Compare
validation-calibrated weighted fusion, agreement rules, and complementary channel
coverage at a matched false-alarm budget. Do not search combinations on test event
coverage and then report that same test score as independent evidence.

A recent ESA-ADB preprint compares supervised CNN/graph models with statistical
methods, but its stated event F0.5 uses uncorrected event precision. Its numbers
are therefore not directly comparable with our target corrected metric. Use it for
ideas about scalable models, not a numerical target to copy.
[Nguyen et al., 2026, evaluation section](https://arxiv.org/html/2607.07335v1#S5.SS1).

## What 0.85 requires

For corrected precision P and event recall R:

`F0.5 = 1.25 P R / (0.25 P + R)`.

Examples computed from this equation:

| Corrected precision | Event recall | F0.5 |
| --- | --- | --- |
| 0.95 | 0.60 | 0.8507 |
| 0.90 | 0.70 | 0.8514 |
| 0.80 | 1.00 | 0.8333 |

Even perfect recall cannot reach the target if corrected precision is below
approximately 0.8193. Conversely, precision 1 requires recall at least 0.53125.
These are trade-off examples, not permission to ignore scientifically important
anomalies or operational recall requirements.

## Reporting and stopping rules

Use the same event scope and calibration budget for every comparison. For finalists,
run at least three random seeds; report each score and its spread, event precision/
recall, false alarms/day, nominal alarm duration, per-category recall, and detection
delay. Measure model size, peak GPU memory, and inference latency on the intended
hardware after warm-up. Prefer event/time-block uncertainty analysis to independent
sample bootstrapping, since adjacent telemetry points are correlated.

Advance from an experiment only if it improves development performance under the
same evaluation rules. A report of 0.85 is acceptable only after the exact metric,
dataset version, channel scope, splits, selection procedure, and operational timing
assumptions are recorded. There is presently no evidence guaranteeing that this
threshold is attainable under the full mission protocol with the available models.
