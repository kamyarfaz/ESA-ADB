# Correctness and research-design audit — 1 November 2026

## Verdict and scope

The checked results are reproducible under their recorded protocols. No demonstrated
arithmetic, event-evaluation, checkpoint-selection, or saved-prediction error was
found that invalidates the recent three-model comparison. This is not a certification
of every historical experiment or proof that the research design was optimal.
The main weaknesses concern score resolution, calibration representativeness,
limited effective search diversity, and the scope of operational claims.

The user requested an audit and assessment of remaining opportunities. This audit
started no training, no new candidate selection and no new assessment experiment.
Frozen source/data/settings/checkpoints/results remain unchanged. New files are
only audit diagnostics and documentation. Dates are Europe/Rome.

## Checks actually performed

Fresh on 1 November:

- Full thesis test suite: **153 tests passed** (including corrected metric parity
  against pinned ESAScores and postprocessing boundaries).
- Frozen reference dataset files, source/annotation/reference fingerprints and
  predecessor artifact receipt hashes replayed. This verifies the active frozen
  data, not the full historical migration/external-drive copy.
- W&B: all nine ten-epoch histories, 18 calibration grids, selected model/scaler
  equality, within/across-rate selection and all **96** metric rows replayed.
  Live runtime/settings match that study's recorded protocol.
- Temporal study: all **144** candidates, selection, exact predictions, event IDs,
  segment diagnostics and **96** metric rows replayed.
- Alarm-evidence study: all **222** candidates and nine selections, exact predictions,
  event IDs, segment diagnostics and **192** metric rows replayed. Its verifier
  uses a separately written transform implementation.
- Original assessment predictions match prior controls; annual/pooled scores are
  recomputed with the corrected annotation-ID/duration evaluator.
- All three complete calibration-score arrays checked for score quantization;
  distinct candidate prediction masks counted without assessing new rules.

Verification outputs were redirected to separate temporary audit directories,
then copied here; prior verification receipts/results were not overwritten.
Scripts: `verify_wandb_finetuning_20261025.py`,
`verify_temporal_postprocessing_20261028.py`, `verify_alarm_evidence_20261028.py`.
Design diagnostics: `scripts/research/audit_design_20261101.py`.
Evidence: [audit summary](tables/research-audit-2026-11-01/audit-summary.json).

A separate **31 October** CPU spot inference check is retained with its original
provenance: 35 distinct windows per model at seven calibration locations; saved
GPU versus CPU maximum absolute differences <=1.79e-6. It passed the declared
numerical tolerance. This is not a full neural inference replication and was not
repeated on 1 November. Its compact evidence is
[score mechanics](tables/research-audit-2026-11-01/score-mechanics-20261031.json).

Tests and replay have limits: the temporal runner and verifier share `postprocess`;
all metric replays share EventEvaluator, supplemented by pinned-reference parity
tests. Receipt hashes certify stored bytes, not flawless experiment design.
Source flow and mtimes support the recorded freeze order but cannot prove that
humans never previously inspected assessment; the benchmark is explicitly reused.

## Finding 1 — the first duration grid was structurally ineffective

`config.py:32,41` sets window length256 and scoring stride32. The input grid is
30 seconds (`full_combinations.py:43`). `masked_pilot.py:74` averages reconstruction
error over each entire window. `diagnose_channels.py:72–78` broadcasts that scalar
back over the window and averages overlaps. Thus changing a thresholded prediction
requires crossing a 32-sample score block: **16 minutes**.

All 1,051,200 calibration samples per model confirm exact 32-sample plateaus and
changes only at indices divisible by 32. This was not just an unfortunate duration
sample. In these series every positive segment was at least 16 minutes, so cutoffs
removing segments strictly shorter than <=16 minutes could not remove anything.
The first grid should have been checked against this scoring structure beforehand.

The neural window is 128 minutes. An interior score blends eight windows whose
input union spans 480 samples (240 minutes), with up to 127.5 minutes future context.
This is retrospective detection, not a causal real-time result.

## Finding 2 — candidate counts exaggerate the effective diversity if read literally

| Model | Temporal settings | Distinct temporal masks | Alarm-evidence settings | Distinct alarm-evidence masks |
|---|---:|---:|---:|---:|
| Short-no-roll | 48 | 1 | 74 | 14 |
| Masked pretraining + fine-tuning | 48 | 1 | 74 | 11 |
| Expanded short-no-roll | 48 | 2 | 74 | 14 |

The nominal 222 alarm-evidence candidates produced 39 distinct masks when counted
within each model and summed. This is a real bounded test, not 222 independent
experiments. Similar scalar thresholds/durations yield duplicate predictions.
Future grids should audit effective diversity and physical timescales before launch.

## Finding 3 — whole-segment removal poorly targets calibration false-alarm time

| Model | Nominal alarm seconds | Seconds from wholly false components | Fraction in components also touching annotations |
|---|---:|---:|---:|
| Short-no-roll | 10,521.642 | 0 | 100% |
| Masked pretraining + fine-tuning | 10,379.778 | 960 | 90.75% |
| Expanded selected model | 10,400.067 | 0 | 100% |

Most nominal-time penalty is inside components that also touch annotations,
including neutral/nonlocal annotations. These components are not wholly false
alarms. Deleting an entire component can discard a detection; refining its
boundaries is a different hypothesis. Do not use true annotation boundaries to
crop predictions—that would use answers to construct detections.

The masked model has a 16-minute wholly false component with peak 1.051630 times
threshold and a 16-minute genuine id_107 component at 1.051257. A simple peak or
length cutoff cannot remove that false component while keeping that particular
true component. This explains one failure; it does not prove all richer evidence
is inseparable.

## Finding 4 — retention and F0.5 are different objectives

The calibration-retention constraint was intentional and was implemented correctly.
It preserves the user's high-recall objective, but it is stricter than maximizing
F0.5 alone. For the masked model, saved candidates reach **0.908914702** calibration
localF0.5 vs **0.882105852**, by reducing detected target IDs from **6 to 4**. All
three families contain such an option; it is infeasible under the declared rule.
This is not a missed detection-preserving improvement and it has not been selected
or assessed now. It is wrong to say no candidate anywhere had higher calibration
F0.5; the correct statement is no eligible candidate improved the selected objective.
For the other two models, relaxing the constraint does not improve the maximum
F0.5 within these saved grids.

The original and expanded local calibration F0.5 values are already 0.999724 and
0.999727 on only six target IDs. There is almost no calibration headroom. Searching
many rules on that small set is weak evidence about seven later years.

## Finding 5 — the expanded model's burden is concentrated in a later regime

Retrospective summary of the already published selected result: **98.8948%** of
the sum of annual false-positive seconds occurs in 2013. That year's nominal-time
FPR is **42.398914%**, versus pooled **5.909894%**. This points to a persistent
late-period failure, not principally brief isolated spikes. Cause is not established
by this summary: distribution shift, auxiliary-channel effects and other mechanisms
need separate evidence. Do not fit a special 2013 threshold, pick channels from 2013,
or claim a new independent test after responding to these assessment outcomes.

## Finding 6 — extra inputs are also extra alarm sources

`full_combinations.py:29–30` defines the 11 inputs. The final score in
`masked_pilot.py:80` is the maximum residual across **all 11 channels**. The 40–47
scope in `masked_pilot.py:149–153` changes the evaluated annotations, not that
pooling rule. Thus 14/21/29 provide context and can directly trigger alarms.
This agrees with the recorded implementation; it is not an undisclosed code bug.
The ESA paper also distinguishes target outputs from other contextual inputs
(requirement R6, printed page 30). Our proposed target40–47 pooling is a separately
declared local-target experiment, not a retroactive change to system evaluation.

A distinct future hypothesis is to retain 11 input channels but pool residuals only
from 40–47. No such completed comparison was located in reviewed source/reports.
The prior per-channel q99 normalization still pooled all channels and used an older
expanded checkpoint. Target-only pooling may lose useful detections, especially
system events, so it must be assessed honestly against both unchanged scopes.

## Finding 7 — prepared-benchmark performance is not label-independent deployment

Inherited preparation reads annotations for both train/test branches and restores
the last annotated raw value and label to the next grid point when resampling
would lose a short event (`Mission1_semisupervised_prep_from_raw.py:130–142`). The
frozen reference preserves those prepared rows. This changes numeric inputs,
not only labels. Test inference itself discards labels; no runtime prediction
masking using test labels was found.

This follows the official ESA benchmark preprocessing convention, documented in
its paper, printed page 27, step 3: [ESA benchmark paper](https://arxiv.org/pdf/2406.17826).
It is inherited benchmark preparation, not a new leak introduced by these runs.
Fair comparisons on the same declared prepared benchmark remain measurements of
that benchmark. Claims of label-independent operational detection require a
separately versioned evaluation with annotation-independent preparation.

Existing cell-level reconstruction verified 2000–2002 only (five numeric restorations
across two IDs). This audit did not reconstruct all test cells or quantify the
impact of restoration on current F0.5. No score-inflation amount is claimed.
Training labels also exclude nonnominal windows, as declared; "no real anomaly
supervision" should mean no positive-anomaly classification objective, not no use
of real annotations anywhere in the pipeline.

## Finding 8 — a precision-only ceiling for original short-no-roll

With fixed detected IDs and precision bounded by1, corrected F0.5 is at most
`1.25 * TP / (0.25 * total_events + TP)`. At48/91 system detections, original
short-no-roll is therefore bounded by **0.848056537**, even with perfect precision.
It cannot exceed0.85 system F0.5 through alarm removal alone while detecting no
new IDs. At47/67 local detections, its corresponding upper bound is **0.921568627**
versus current0.907139530. These are ideal mathematical bounds, not achievable
results or predictions. Other models have more recall and hence more potential
precision-only headroom; this does not prove that false and true alarms can be
separated. See [analytical bounds](tables/research-audit-2026-11-01/precision-ceilings.json).

## Most defensible next work — proposals only

1. **Latest-model calibration diagnostics for auxiliary context versus alarm source.**
   Retain inputs/weights and save per-channel calibration residuals. Compare a small,
   predetermined all-output versus target-output pooling rule. Preserve the baseline
   and both evaluation scopes. Do not use 2013 to choose the rule.
2. **Detection followed by boundary localization.** Keep the existing detector as
   an event proposer; investigate whether finer residual evidence can localize
   its alarm boundaries without discarding event cores. Rules must come from
   training/calibration, never true assessment boundaries. Short true events and
   false/true weak peaks remain risks; improvement is unproven.
3. **Stronger validation before another broad search.** Use chronological development
   folds, with fold-specific training/scaling and a gap sufficient for overlapping
   windows, plus multiple seeds for robustness. A model already trained on 2000–2005
   cannot provide independent validation on a 2003 fold. Final 2000–2005/2006/2007–2013
   comparison can remain for comparability, but reused 2007–2013 is development evidence.
   Time-ordered validation and gaps have standard support:
   [TimeSeriesSplit documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).

Do not repeat failed ideas as new discoveries. Wholesale per-timestep replacement
already lost a [historical paired comparison](scoring-comparison.md) (calibration 0.882109→0.588188,
assessment 0.761655→0.671790, different historical checkpoint/period).
Channel normalization, adaptive scoring and score-change/reference variants also
have existing negative or inconsistent evidence. Boundary localization around a
fixed detector is distinguishable from replacing the detector's entire scoring.
The literature supports evaluating scoring separately from models, not a guarantee
of success here: [Garg et al.](https://arxiv.org/abs/2109.11428).

## Current result status

The existing scores remain unchanged and reproducible: original short-no-roll
system/localF0.5 **0.836070/0.907140**; masked **0.795675/0.844315**; expanded
**0.844573/0.830900**. The W&B expanded-model improvement remains a measured
single-seed result on the reused benchmark. This audit produces no new F0.5 result,
no model superiority guarantee, and no robust thesis-target claim.
