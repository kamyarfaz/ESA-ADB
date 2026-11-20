# Whole-system and frozen cross-mission transfer execution protocol — v2

The user explicitly approved the15-minute revision on20 November2026. This
separate version preserves v1 source, settings, preparation and failure logs.
Only Mission1 channels62/63 receive a900-second maximum observation age, applied
identically in training/calibration/assessment; all other inputs remain600s.
The evidence is training-only cadence feasibility, not a performance comparison.
Held readings may hide between-observation changes; physical units remain anonymous.
All channel lists, mapping, methods, seeds, splits, training/selection settings and
assessment gates below remain as specified in v1. No new research ranking is used.
Output: results_longrun/cross_mission_transfer_20261120_v2.
Code: esa_thesis/cross_mission_study_v2.py. The unchanged v1 research artifacts are
read and fingerprinted, and all prepared arrays are rebuilt in the new directory.
The prior failed service is disabled to prevent accidental startup; only the new
v2 service may train. Global locks are shared with v1 and historical study engines.
The new source/runtime/data freeze is independent of the preserved v1 freeze.

This is the execution specification following the training-only research in
cross-mission-transfer-protocol-2026-11-18.md. It becomes immutable when the new
service writes preparation_frozen/protocol.json. The user authorized research,
training and assessment in both missions and confirmed separate time splits.

## Research conclusion and input choice

Mission2's historical eight/eleven-input panels contain annotations for14/198
training events (1/21 anomalies). All22 target channels sharing Mission1's two
anonymous subsystem/unit pairs contain19/198 events. Therefore, restricting
fresh whole-system training to these families would omit most annotated input
coverage. These are training membership counts, not a ceiling on indirect
detection. Physical meanings remain undisclosed. Behavioral input mapping is
unstable under leave-season-out analysis and is an experimental transfer test.

Use all13 target-channel metadata groups in **each** mission for fresh training.
Within each group choose a nominal-training medoid: maximize the mean of absolute
quarter-median Pearson level and first-change correlations to the other members.
Undefined correlations contribute0; singleton groups have score1. Ties prefer
higher overall nominal training coverage, then lower numeric channel ID.
The base panel has one representative per group (13 channels). The expanded
panel retains those13 and adds one related channel from each nonsingleton group:
maximize absolute quarter-median first-change correlation to the representative,
then level correlation, coverage, and lower numeric ID. Thus expanded panels have
25 Mission1 and26 Mission2 inputs. All target channels are numeric and nonconstant
in the existing training inventory; no categorical values or counters are silently
cast into analog measurements. Channel ranking uses nominal training signals
and group metadata only, not event counts, calibration or assessment outcomes.
The concrete ordered lists and all ranking values are saved in input-selection.json.
Groups are mission-specific; representative input positions are not homologous
across these freshly trained models.

This changes the research question from a narrow sensor-family benchmark to broad
whole-system monitoring. It is not a matched replication of the historical8/11
or10/13-input studies. Expanded inputs also expand reconstruction-score sources.
Some target channels remain unobserved directly; all system IDs still count.

## Fresh training:18 sequential runs

Mission2 first, then Mission1. Each has short-no-roll, masked pretraining followed
by short-no-roll, and expanded-input short-no-roll; seeds42,43,44 for each method.
All weights and scalers are fresh. No best seed is selected. Forty epochs;
masked method uses20+20 with the existing optimizer reset. Same Transformer
autoencoder, short-no-roll corruption, AdamW1e-4, weight decay1e-4, gradient
clip1, cosine schedule to1e-6, and no AMP. Batch32 for all new jobs, fixed before
training; this differs from the historical batch64 and controls the larger
channel-attention memory cost. Window256, training stride16, score stride32;
score the final tail window to cover every assessment row. Two CPU threads.

Splits (end exclusive):
- Mission1: train2000-01-01–2006-01-01, calibrate2006, assess2007–2013.
- Mission2: train2000, calibrate2001-01-01–2001-10-01,
  assess2001-10-01–2003-07-01.

Mission1 uses30-second and Mission2 uses18-second grids. These preserve established
mission-specific resolutions; anonymization prevents claiming equal physical
window durations. Inputs are raw, causal, split-contained previous observations,
maximum age900s for Mission1 channels62/63 and600s for all other inputs, without label-dependent restoration or backfill. Missing values
remain NaN until training-derived median imputation in the established scaler.
This differs materially from older Mission1 restored prepared data.
Exclude all annotated training intervals, held annotated source observations,
and unavailable rows in the union expanded panel. All three methods in a mission
share exactly this nominal training mask/windows. Robust median/IQR uses these
nominal training rows; existing IQR<1e-8 fallback1 is recorded if encountered.
No calibration/assessment labels alter input values or discard evaluation rows.

Every5 non-masked epochs, sweep the existing11 score quantiles plus the no-alarm
maximum, merge gaps0/16/64 and minimum durations1/8/32. Select corrected
**whole-system Anomaly+Rare Event F0.5**, with calibration predicted-point fraction
at most1%. Within-epoch ties use the existing nominal-alarm-duration/threshold/
gap/duration order; between-epoch ties retain the earliest epoch. Historical
engine field `calibration_target_f05` denotes the explicitly configured system
scope in this new adapter. Freeze all18 selections jointly before preparing any
new assessment arrays. No adaptation based on intermediate assessment results.

## Existing Mission1 models on Mission2

Use the nine **context-arm** models from context_expansion_20261104_v1: all three
techniques and all seeds, preserving weights, original source scalers and2006
decision rules. No best historical checkpoint is chosen anew. Order Mission2
inputs by each checkpoint's source feature order through the sole all-training
mapping saved by the channel audit. No duplicates, sign inversions or time warp.
Use30-second Mission2 input grids to preserve the source model's numerical
window and postprocessing cadence. All Mission2 assessment events still count.

Report both predefined conditions for every source model:
1. **Frozen source scaling:** unchanged Mission1 scaler and rule; Mission2
   training values are used only to select the channel mapping.
2. **Target-training normalization:** same weights/rule/mapping, but per-input
   robust scaler fitted on nominal Mission2 training2000 at30s. This is adaptation
   using target training data, not strict zero-shot transfer. No target calibration
   or assessment enters scaling or rule selection.

The mapping costs/associations are uncalibrated and unstable. Raw groupwise
normalization, unknown time scaling, new causal preparation and covariance
mismatch are reported limitations. A negative transfer result is valid evidence
and must not trigger another map/seed/threshold sweep on the assessment.

## Verification, resources and reporting

One service esa-cross-mission-study-v2.service; global research locks prevent duplicate
training. RAM high10GiB/max12GiB, no additional swap; require at least6GiB available
RAM,8GiB free GPU memory and80GiB disk before each new job. No automatic restart.
Failures remain in logs; recovery requires cause/resource/fingerprint checks and
the same checkpoint/settings. Synthetic tensor memory preflight is not an
experiment and must pass before launch. No Mission3, MOSAD or other ablation.

Freeze the preparation plan, source files, protocols, research/input selection,
old transfer model/rule fingerprints and runtime. Recheck183 official-v1-derived
raw fingerprints. Independently check raw preparation values, age/availability,
annotation boundaries and nominal windows before training. Preserve all candidate
checkpoints/calibration grids for final independent replay.

Fresh results: system and fixed13-base-input scopes; transfer results: system and
mapped-original-eight scopes. In each, primary Anomaly+Rare Event and secondary
anomaly-only (rare annotations neutral for false alarms), annual/pooled corrected
event F0.5, event TPR, nominal-time FPR and raw detected/missed IDs. Expected720
metric rows:432 fresh and288 transfer. Report all seeds, means/ranges, and paired
method comparisons within this campaign; historical comparisons are descriptive.
Independent replay and checkpoint/source spots are required before final publication.
Report annotation coverage separately from actual detections. Reused benchmarks,
only three seeds, extra compute, new source preparation and score-source expansion
preclude an independent-generalization or robust-target claim.

Publish validated W&B evaluation runs, an academic report, additive approved
workbook sheets, compact Git results and essential artifacts in a verified GitHub
Release. The full external copy is preserved. Three-hour chat updates explicitly
include healthy progress. Completion cleanup only after verified delivery.
