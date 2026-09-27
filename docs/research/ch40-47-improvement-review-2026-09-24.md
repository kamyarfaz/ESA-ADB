# Channels 40–47: data findings and F0.5 improvement priorities

## Scope and evidence

The user explicitly paused other-channel investigations. Keep **all eight inputs
40–47** fixed and improve their detector before considering other channels.
Reference pooled corrected ESA F0.5 is **0.805722**, not yet the 0.85 target.

New audit: native and prepared data for training 2000–2005 and calibration 2006.
No new assessment-data analysis or threshold tuning on 2007–2013 was performed.
The existing 14-year descriptive inventory remains available, but new method
choices use training/calibration evidence. The benchmark assessment has already
been exposed historically; future improvements on it are development evidence.

Script: `scripts/research/ch40_47_review.py`. Evidence:
`tables/ch40-47-review-2026-09-24/`; plot:
`outputs/ch40-47-review-20260924/monthly_profiles.png`.
Verified 7,364,160 prepared rows, 672 monthly/channel summaries, 56 raw channel/year
summaries, and seven symmetric 8x8 correlation matrices. Raw selected data has no
nonfinite values or duplicate timestamps. Plot visually inspected. Source CSV
size/mtime stable. Anonymous metadata prevents determining actual sensor meanings.
This is a substantial statistical review, not a claim of complete physical understanding.

## What these eight channels look like

- Channel 40: subsystem_6, physical_unit_3, group 6. Typical native cadence 90s;
  about 66–68% of consecutive native values unchanged across individual years.
- Channels 41–46: subsystem_5, physical_unit_4, group 8. Typical cadence 30s;
  each has approximately 8–20% unchanged consecutive native values.
- Channel 47: subsystem_6, physical_unit_3, group 9. Typical cadence 30s;
  about 64–65% unchanged consecutive values.
- All are target channels in supplied metadata. Subsystem/unit labels are anonymous:
  do not invent temperature, voltage or subsystem identities.
- Monthly nominal medians of 40 and 47 show slow multi-year swings. Channels 41–46
  have narrower bands and a level change around 2001. These are descriptive patterns,
  not established physical cycles or faults. Plot axes are independently scaled.
- Training correlations: 41/44 = 0.909 pooled (minimum yearly absolute 0.903),
  42/45 = 0.891 (minimum 0.885), 43/46 = 0.887 (minimum 0.881).
  These relationships motivate checking residual relationships while keeping all inputs.
  Channel 40 versus 47 is 0.883 pooled but 0.252 in 2001. Neither is strongly
  linearly correlated with most of 41–46. A single homogeneous group assumption is weak.
- Native gaps reach 21,330s for 40 and 21,270s for 41–47 (about 5.9h).
  Filled prepared values do not establish observation availability. Gap-aware
  diagnostics are needed before interpreting every long flat region as normal.
- No nominal 2000–2005 values on these inputs clip at ±10. Some excluded training
  values on 41–46 do clip (about 3.5–4.1% per channel of globally nonnominal rows).
  This does not establish that clipping causes misses or justify changing it blindly.

## Anomaly durations and evaluation scope

Training/calibration annotations on 41–46 include point events and short intervals;
median annotated interval durations are 450–660s (7.5–11 minutes). Some 2006
intervals are only 3.912–9.912s. Channel 40's median annotated interval is about
4.9 days; channel 47's about 2 days. These statistics concern annotation rows,
not independent event counts, and include Anomaly and Rare Event categories.
The inherited preparation can restore short annotated events to the 30s grid.
Consequently, model results do not yet represent a strictly label-independent
operational preprocessing pipeline.

The evaluator merges mission-wide annotations by event ID. In 2006 it counts
10 events; only six have annotations on these eight channels. The model detected
six and missed four. Matching counts do **not** prove the detected IDs are precisely
those six; per-event calibration attribution is still required. Events without
local annotations may have correlated signals, so this is not a mathematical
recall ceiling. Keep the mission-wide metric for comparability. Any local-channel
metric must be separately labeled, never substituted to claim reaching 0.85.

## What the current model/scoring does

The model is already a temporal/channel Transformer autoencoder. Sequence length
256 at 30s means 128 minutes of context; patches of 16 mean 8 minutes.
Training stride is 8 minutes; scoring stride 16 minutes. Squared reconstruction
errors are averaged across the entire window for each channel, spread across its
covered timestamps and averaged across overlapping windows. Finally, the maximum
of the eight channel errors is the detection score. This is retrospective scoring.

Input IQR normalization does not guarantee equal reconstruction-error distributions.
The maximum could be dominated by a naturally harder-to-reconstruct channel;
that is a hypothesis requiring saved per-channel errors, not yet a measured cause.
Window averaging could dilute short events, but **the earlier paired comparison
already favored window-mean over per-timestep scoring**. Do not repeat that change
as if it were new or guaranteed better.

The decoder receives input-derived tokens. A flexible reconstruction model can
potentially copy anomalous inputs well; this is a reconstruction hypothesis, not
proof of copying in these checkpoints. Pseudo-anomalies use localized offset,
noise and temporal rolls, lasting roughly 6–42 minutes on 1–4 channels. Their
amplitude depends on local variability, which could make some corruptions weak
on slowly varying signals. Duration/amplitude realism needs measurement.

The completed run selected epoch 10: calibration F0.5 0.882148. At epoch 40,
training loss was lower but calibration F0.5 was 0.357143. More epochs alone
are therefore not supported by this run. Checkpoint selection correctly preserved
epoch 10. Among saved threshold candidates at epoch 10, lowering the threshold
from 7.65037 recovered **no extra event** (still six) while adding false alarms.
Simply lowering the existing threshold is not a promising next experiment.

## Research-based shortlist

1. **First: channel-wise error calibration with fixed model weights.** Save each
   channel's window-mean reconstruction error, fit error scales on a fixed,
   temporally distributed nominal training sample, and compare the original max
   with a prespecified normalized max. Retain the same temporal window aggregation
   so only one factor changes. Garg et al. independently compare models and scoring
   and find that score design can strongly affect results [1]. Their metric and
   datasets differ; this supports a test, not an ESA improvement claim.
2. **Conditional follow-up: causal adaptive error normalization.** If nominal
   error distributions actually drift, compare one specified trailing-history
   normalization against the static version. Initialize from training and exclude
   the current observation from its reference history. Prolonged anomalies can
   contaminate an adaptive baseline; update safeguards must be explicit. Spacecraft
   dynamic thresholding work provides relevant motivation [2]. Do not enable
   adaptation solely because raw medians drift; residual drift must be measured.
3. **Only if scoring is inadequate: masked reconstruction.** Hide selected patches
   and reconstruct them from context to reduce direct copying. TFMAE is a concrete
   Transformer masking reference [3]. A small adaptation to this model is preferable
   to immediately importing its entire pipeline. Training and inference masks must
   match. This is a distinct model experiment, not part of the scoring comparison.

Relationship residuals on 41/44, 42/45 and 43/46 and more realistic pseudo-anomaly
injection remain hypotheses to inspect, not extra experiments to launch in parallel.
No reason currently supports a broad model sweep, additional channels or Chronos.
Published scores using different datasets/metrics are not comparable with ESA F0.5.
Keep the corrected event-ID/duration metric and nominal false-alarm penalty [4].

## Concrete next step

Use the existing epoch-10 checkpoint for **one diagnostic inference pass**, retaining
per-channel errors on a fixed nominal training sample and all 2006 calibration data.
Report per-channel error quantiles, alarm-contribution counts, event IDs detected/
missed, and score/time plots for all calibration events. Check whether shorter
annotations and missing observations explain failures. Preserve sample timestamps,
checkpoint hash and training/calibration boundaries. Do not load assessment scores.

Then compare just original max versus training-normalized max, with a prespecified
normalization formula and the existing threshold/postprocessing grid. Calibration
has only ten events and has already selected checkpoints: report sensitivity and
avoid treating it as independent validation. Any further model development needs
chronological checks using earlier folds with normalization fitted only before
each fold. Do not select schemes or weights from 2007–2013 results.

This turn completed data exploration and method research, not the diagnostic
inference or a new model run. The saved cache contains only aggregated assessment
scores, so channel-wise calibration diagnosis requires new inference. No F0.5
improvement has been claimed or produced yet.

## Primary sources consulted on 24 September 2026

[1] Garg et al., *An Evaluation of Anomaly Detection and Diagnosis in Multivariate
Time Series*: https://arxiv.org/abs/2109.11428

[2] Hundman et al., *Detecting Spacecraft Anomalies Using LSTMs and Nonparametric
Dynamic Thresholding*: https://arxiv.org/abs/1802.04431

[3] Fang et al., *Temporal-Frequency Masked Autoencoders for Time Series Anomaly
Detection*, ICDE 2024, authors' implementation: https://github.com/LMissher/TFMAE

[4] Kotowski et al., *European Space Agency Benchmark for Anomaly Detection in
Satellite Telemetry*: https://arxiv.org/abs/2406.17826
