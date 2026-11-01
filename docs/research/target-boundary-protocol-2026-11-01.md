# Target-output scoring and boundary refinement — 1 November 2026

User authorized the recommendations following the correctness audit, including
checking post-processing. This is a new, bounded, retrospective study. No neural
training, new seeds, MOSAD, data regeneration or assessment-guided extensions.
All historical artifacts and package source remain unchanged.

## Inputs and questions

Use the three W&B-selected checkpoints from 25 October: original short-no-roll,
masked-pretraining/short-no-roll fine-tuning, expanded-input short-no-roll LR3e-5
epoch10. Training2000–2005, calibration2006, assessment2007–2013; seed42.
Original short-no-roll remains an unchanged reference, with no new inference.
Expanded inputs remain40–47,14,21,29; the masked-pretrained model uses40–47.
Both high-recall models are ordinary unmasked AEs during this inference stage.

Three candidate arms, sharing just two existing checkpoints:

1. expanded_target: expanded inputs, coarse/fine scores pooled over40–47 only.
2. masked_all: existing eight-input masked-pretrained model and original threshold.
3. expanded_all: expanded inputs and existing all11-output score/threshold.

The last arm isolates boundary effects from output-channel pooling. Same predictions
are evaluated against system and40–47 annotation scopes. Context-only pooling is a
declared local-target change; it may reduce system detections.

## Score extraction, frozen before calibration diagnostics

Preserve the existing scaler, float32 squared reconstruction errors, 256-sample
windows, stride32, batch64 and30-second timestamps. Save window means by channel
and overlap-average them exactly as the existing scorer. Require all-output score
parity (rtol1e-5, atol3e-6) and **exact original-threshold prediction parity** against
the saved calibration and, after selection freeze, assessment arrays. Fail closed
on disagreement. Additionally overlap-average the squared errors at each actual
timestamp, rather than broadcasting each window mean, to produce finer evidence.
The fine evidence is used only inside proposed alarm segments. No replacement of
the complete detector by pointwise scoring, no normalization or adaptation.

Bounded inference chunks preserve errors/overlap sums, counts and SHA256 receipts;
resume only with matching source, checkpoint, settings, runtime and input hashes.
Labels are not loaded into neural inference. Preserve contextual future access;
this remains retrospective, not real-time.

## Calibration and bounded candidates

All settings below are declared before inspecting the new per-channel residuals.
Use the corrected annotation-ID/duration evaluator. Primary objective is local
40–47 F0.5, not pointwise F0.5. Maintain <=1% total alarm fraction. Retain every
local ID detected by the respective original model on2006. Selection order:
highest local F0.5; lowest nominal alarm seconds; unchanged fallback; fixed index.
Calibration retention cannot guarantee retention in2007–2013.

Expanded target-only threshold grid: existing percentiles90,95,97.5,98,99,99.5,
99.75,99.9,99.95,99.99,100 plus the original numeric threshold; gap0/minimum1.
Select the best eligible target-only rule. If none retains the original local IDs,
record failure and use the original all-output fallback; do not silently relax the
constraint. Always report the target-only candidate grid and whether target-only
pooling advances versus the unchanged original under the same primary objective.
The target-only arm remains a declared challenger even if its eligible score is
worse than original; no choice is made using assessment.

Boundary refinement: six candidates plus explicit unchanged fallback. Fine-score
cutoffs are95th,99th and99.9th percentiles of globally nominal2006 samples, excluding
all annotated intervals (including neutral annotations). For each coarse connected
component, find the first and last sample with fine score strictly above cutoff;
retain their entire enclosing span, optionally padding0 or16 samples (0 or8min),
clipped to the original component. If there is no crossing, retain that component
unchanged. This trims only the two outer boundaries, never splits or deletes a
component. No annotation boundaries enter the transform. Select per arm using the
constraints above. Preserve all unsuccessful candidates.

Then hold each selected boundary rule fixed and test four existing families,
individually (not chained with each other), on that arm's coarse score:

- Temporal: gap0,1,2,4,8,16,32,64; minimum1,2,4,8,16,32 samples (48 settings).
- Strength-duration: minimum1,32,48,64,96,128,256,512 samples with peak/threshold
 1.1,1.25,1.5,2,3,5; remove only if both weak and short (48 settings + fallback).
- Hysteresis: low factors0.5,0.75,1; high1,1.1,1.25,1.5,2; keep low components
 containing a strictly high crossing (15 settings + fallback).
- Duration: minimum1,32,48,64,96,128,256,512 (8 settings + fallback).

For each candidate, apply the family to the coarse score **then** the already
selected boundary rule using fine evidence. Each family includes an explicit
unchanged-boundary fallback (temporal uses its gap0/min1 entry); identical masks
are counted and cached, not presented as independent evidence. These grids test
whether earlier filters help the newly specified scoring/localization. Do not
expand ranges after seeing outcomes. Each arm has7 boundary +122 family entries;
387 entries overall, plus the target-threshold grid. Original references are kept.

## Freeze, assessment and verification

Save all three arm specifications and all selections in one hashed freeze receipt
**before** loading any new assessment input/residuals. Threshold diagnostics and
selection use2006 exclusively. Assess each arm's base, selected boundary and four
selected family predictions, plus three unchanged original controls:21 variants,
336 annual/pooled metric rows. Pool continuously across2007–2013, not by averaging
annual scores. Report corrected F0.5, precision, event TPR, time FPR, detected/missed
IDs and gains/losses relative to original controls. Report all outcomes, not only
the largest assessment score. No per-year thresholds or2013-specific fix.

Independently replay inference samples, channel pooling, candidate transformations,
calibration metrics/selections, predictions and every metric/ID row. Check receipts,
frozen dataset/source/checkpoint/runtime and parity tests. Report score-replay
tolerance and scope; sampled inference is not full independent neural replication.
Evaluation publication requires successful independent verification.

## Generalization and reporting

Report fixed selections by calibration halves and quarters to expose calibration
event scarcity/heterogeneity. These are descriptive stability checks only: the
checkpoints and rules already used2006, so these blocks are not independent folds.
Prepare a chronological validation design with fold-specific training/scaling,
separate calibration/validation and adequate context gaps before any broader
tuning. Do not launch a new training campaign under this bounded inference study.
Multiple seeds and truly unseen data remain necessary for stronger claims.

Disclose inherited label-informed benchmark preparation, single seed, reused
benchmark, offline lookahead, extra inputs and additional inference/selection
compute. Do not claim robust F0.5>=0.85 achievement. Update the report, historical
workbook, W&B verified-evaluation records and AI_HANDOFF. Commit compact code/results
and checksum-verify essential new artifacts in a GitHub Release.

Service: esa-target-boundary.service. Output:
results_longrun/target_boundary_20261101_seed42_v1. Log:
/home/k_faz/project-backups/run-logs/target-boundary-20261101.log.
Never duplicate workers. On interruption inspect cause/resources and verify no
runner before resuming the same service with exact fingerprints. Disable service
and remove the monitoring follow-up after verified delivery.
