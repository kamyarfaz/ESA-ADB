# AI handoff — ESA satellite anomaly-detection thesis

Updated **14 October 2026**. This is the current entry point for a new assistant.
Read this before proposing work. The chat is not required to continue the project.

## 1. Current task and decision

**Latest request: run short-no-roll on the full 2007–2013 period and report every
year plus pooled F0.5.** This explicit request supersedes the earlier no-extension
stopping decision for this candidate. Plan: `docs/research/full-short-no-roll-plan-2026-10-14.md`.
Use frozen training2000–2005/calibration2006/testing2007–2013; channels40–47,
40 epochs, all nominal windows, seed42. Runner `esa_thesis.full_short_no_roll`.
Output `results_longrun/short_no_roll_full_20261014_seed42_v1/`. The run is active in tool session8451 (initial Python PID1529365); log:
`/home/k_faz/project-backups/run-logs/full-short-no-roll-20261014.log`.
Heartbeat `complete-full-period-short-roll-assessment` checks every30minutes and
must deliver verified results then pause itself. Verify live process state rather
than trusting this PID. Do not claim results until complete.json and replay checks pass.
122 tests passed in12.840s before launch. Source/data/settings must stay unchanged
for strict resume; documentation updates are safe.
No yearly tuning or independent-test claim. The original baseline is unchanged.

**Current instruction, 14 October: autonomous research is authorized again,
including justified training and commands. This supersedes the older stop below.**
Follow `docs/research/research-resumption-2026-10-14.md`. The first authorized
comparison executes the predeclared nominal-quantile diagnostic using verified
saved scores only. The damaged historical MLP array is excluded. No final-test
selection or indiscriminate training sweep is authorized by the research protocol.
The dated stopped-state paragraphs below record history, not current instructions.

Completed the fixed q=0.999, epoch-five threshold comparison on all six original
joint runs. All original dependencies matched migration hashes; control masks and
metrics reproduced. All candidate rules were sealed before validation arrays were
loaded. Mean system F0.5: 2001 control 0 → candidate 0.384075; 2002 control 0 →
candidate 0.006370. Candidate 2002 occupancy is 87.20%; false-alarm duration rises.
**Advancement failed in both folds.** Do not tune neighboring quantiles or promote
this candidate. See `docs/research/nominal-quantile-results-2026-10-14.md` and
`docs/research/representation-decision-2026-10-14.md`. Frozen reference reverified;
120 tests passed in 11.846s. No new neural training or 2007–2013 scoring occurred.


GitHub synchronization completed **14 October 2026**. Source update `751fc1a`
is pushed; the full current project-file snapshot is published at
`project-snapshot-2026-10-13` (50 Release assets, 174.06 GiB before compression).
All 10,526 archived entries passed verification and all uploaded asset SHA256
digests matched local files. See `docs/research/github-sync-2026-10-13.md` and
`docs/research/github-snapshot-verification-2026-10-14.json` for evidence and
exclusions. The full thesis suite passed 117 tests before publication.
The original historical array mismatch remains unresolved. Research stays stopped.

**Current instruction: research is stopped at the user's explicit request on
13 October. Do not execute the nominal-quantile plan or launch training.**
The subsequent authorized task is repository synchronization and comprehensive
GitHub backup only. GitHub is authenticated as `kamyarfaz`; credentials are outside
this project. Before publication, local/remote main both pointed to `0fd67c8`.

Completed before the stop: calibration diversity and label-influence diagnostics,
a targeted literature review, and a training-only reconstruction of preprocessing
on channels 40–47. The influence audit reproduced 2,898 candidate rows and six
original decisions. Use `calibration-influence-2026-10-13-v2` (it adds the eligible
candidate flag); v1 remains preserved. Removing the detected 2001 temporal
component leaves all eligible candidates at zero; withholding id_84 in 2002
raises the selected threshold 2.60–3.04 times. These are diagnostics, not new
validation results or adopted rules. The preprocessing reconstruction matches
3,156,480 timestamps per input; annotation restoration changes five channel-values
at two timestamps. Neither analysis trains a model. The nominal-quantile plan is
**written but unexecuted**. The repository check passed **114 tests** in 11.525s.

The historical MLP checksum failure is still unresolved. A search of all 59
same-size local files found no matching original. See
`docs/research/migration-exception-2026-10-13.md` for the bounded diagnostic
continuation used before the stop. It never certified a lossless transfer.

GitHub synchronization records and restore instructions are in
`docs/research/github-sync-2026-10-13.md`. Keep raw datasets/weights out of source
Git; the full project-file snapshot belongs in split Release assets. Backing up
the current mismatching array preserves evidence, not a verified original.

The dated paragraphs below preserve earlier setup history; the latest resumption
instruction and protocol take precedence over earlier stop/continuation notes.

13 October continued setup: packaged and verified the existing reference as
`data/frozen/mission1-reference-v1/{train.csv,calibration.csv,testing.csv}` plus
`labels.csv.gz` and `manifest.json`. Files are read-only, source row bytes are
preserved, and disjoint 30-second coverage and checksums passed. Manifest SHA256:
`a49a8c42c23a5d72d91a48596f485d762a805cc6101ad037c01a36c8037bf099`.
Use `python scripts/research/freeze_reference_data.py verify` before consuming.
The historical runners still use the original sources; no silent redirect or new
experiment occurred. See `docs/research/frozen-reference-2026-10-13.md`.
Five new integrity tests and the existing suite pass: **103 tests**, 13.330 seconds.

Transfer recovery: restored `.gitattributes` and the October 12 backup report from
HEAD; restored eight missing symlinks and preserved external W&B log targets;
recovered the damaged archived `fine_ch08_35_42/test_score_per_channel.npy` from an
exact-hash local duplicate, retaining its damaged original. Corrected executable
bits on 2,216 tracked files only where contents exactly match the Git index; left
content-modified files untouched. Recovery evidence is under
`migration/recovery-20261013/`. The historical
`pa_center_ch08_40_47/mlp_test_score_per_channel.npy` still mismatches the transfer
hash. The September 17 release archive was downloaded, checksum-verified and
inspected; this array is listed under EXCLUDED files, not included. Preserve the
current array; do not regenerate it and claim exact recovery. The user supplied
the current checkout path and GitHub URL, not an independent original copy.
No research training may start while transfer recovery remains unresolved.
The post-recovery full destination scan completed: **12,517 entries; 16 failures**.
Only one failure is a research array; fifteen are newer documentation/Git state
and snapshot differences relative to the old manifest. All other recorded artifacts
match. See `docs/research/transfer-recovery-2026-10-13.md`; do not describe the full
transfer as zero-failure or erase the original manifest. The original Git-status
snapshot also cannot be reconstructed exactly from the available copy.

13 October destination environment verified after user installation:
`/home/k_faz/miniforge3/envs/esa-thesis` has Python 3.9.23 and
PyTorch 2.6.0+cu124. `pip check` and `python -m esa_thesis doctor` pass.
CUDA detects NVIDIA GeForce RTX 2080 Ti; a finite 256x256 matrix multiplication
and CUDA synchronization passed. All 98 thesis tests passed in 12.106 seconds
on this destination. These are environment/regression checks, not a research
training run. The subsequent recovery and updated test count are recorded above.
The full transfer remains unresolved; do not launch experiments yet.

13 October research brief: the user requires all-channel anomaly/event evaluation,
three permanently fixed train/calibration/testing files for final comparisons,
a separate binary ground-truth label file, and a concise thesis results workbook.
They clarified: evaluate all-channel events; decide model inputs after research.
Do not equate this with authorization to train a 76-channel model. Before proposing
another experiment, review existing diagnostics and primary literature, explain
the mechanism and hypothesis, and define one controlled comparison and stopping
criteria. Avoid speculative model sweeps. The user wants concise explanations.

Audit and deliverables: `outputs/thesis-summary-20261013/deliverables/` contains
`ESA_thesis_results.xlsx` (one sheet, 65 method-period rows),
`ESA_Mission1_binary_labels.csv.gz` (14,728,320 timestamps across the existing
reference splits, all 76 channel labels, plus a communication-gap ignore flag),
and `research_protocol_review.md`. Both original prepared CSV SHA256 checks pass.
The reference dates are train 2000–2005, calibration 2006, testing 2007–2013;
these originally came from two source files and are now packaged separately as
recorded above. No new date split or pipeline replacement was implemented. The
independent-test status remains to be settled; inspected
test periods cannot become unseen by freezing files. Raw event IDs/intervals remain
authoritative for corrected event scoring; do not substitute pointwise F0.5.

13 October destination update: the user confirmed this is a fresh server and
prioritized preparing research reporting before environment setup.
Created `outputs/research-update-20261013/deliverables/ESA_research_update_2026-10-13.pptx`:
12 slides in the September 30 deck's layout, covering completed work since that
meeting, with source references and speaking notes. No research decisions changed.
Also updated the comprehensive workbook as a new copy in the same deliverables
folder: `ESA_all_methods_results_2026-10-13.xlsx` (25 sheets, 14 charts, 256
assessment records, including 112 added October records). It preserves the
September workbook, adds all three no-roll seeds and internal primary/secondary
policies, and includes calibration/event/channel-intervention diagnostics.
Formula checks, source metric arithmetic, original-cell preservation and rendered
updated ranges were checked. No model inference or training was performed.
The neutral `research-update-20261013` deliverables supersede the earlier October
reporting copies, which remain preserved. User preference: use academic language
and omit supervisor names and personal meeting references from Excel and reports.
Both revised packages were checked for the prohibited name, including notes and
metadata; workbook numeric values and formulas are unchanged.

Destination transfer verification completed: **12,517 entries checked, 25 failures**.
The log is `migration/destination-verification-2026-10-13.log`. Failures include
Git metadata, handoff/readme files, links, and two older score arrays. Causes have
not been resolved; do not describe the destination as verified or delete source
copies. Tracked `.gitattributes` and the October 12 backup report are also missing
relative to HEAD, and many executable bits differ. System Python remains
3.12.3; the separate Python 3.9 research environment and destination thesis tests
now pass as recorded above. Presentation tooling used the bundled artifact runtime.
Retain the original transfer manifest as evidence; reconcile failures before
creating a new manifest for any subsequent transfer, including today's new files.

The user is moving the **whole project directory to another PC**. Migration is
active again. Complete destination verification and environment checks before
research. Do not launch training during transfer. No research jobs were found
running at the source check on 12 October. Old PIDs, pending runs and commands in
historical notes are obsolete unless independently verified.

All experiments through 8 October below are finished; none of the new internal
candidates passed its advancement rule. Keep the established full-period baseline.
Next research work is **consolidating findings and defining a more representative
calibration/generalization protocol**, not another blind model/threshold sweep.
No replacement protocol has yet been implemented or approved as a specific run.
The user prefers autonomous execution within the agreed task, without repeated
permission questions. Preserve results and disclose experimental limitations.

Start on the new PC:

1. Read [MIGRATION.md](MIGRATION.md); verify the copied manifest **before imports,
   tests or Git commands that could modify the transferred tree**.
2. Recreate the Python environment; run doctor and the thesis tests.
3. Read the linked reports for the next task. Do not rerun completed experiments
   merely because an old history entry says “next” or “active”.

## 2. Thesis and scope

Kamyar Faz's thesis is satellite anomaly detection using ESA Mission 1 telemetry.
Supervisor Federico Butchellato supplied initial main/models/utils Python files
and requested a Transformer approach aiming for F0.5 >=0.85. Tentative title:
“Artificial Intelligence for Anomaly Detection in Satellite Telemetry under Space
Operational Constraints”. No final title/deadline; a complete thesis manuscript
has not been written. Do not claim the target is robustly achieved.

Current inputs are **eight channels 40–47**, not all 76 telemetry channels.
“System scope” means evaluating those predictions against all-system event IDs;
it does not mean the model consumed all 76 channels. Local scope uses events
annotated on the eight inputs. Use identical frozen predictions for both scopes.

Supervisor requirements: investigate informative channels, correlations and
preprocessing; compare methods on the **same real assessment data**, never
synthetic test data; present organized results by period/channel set and plots.
Artificial anomalies are training augmentation only. The current research focus
remains 40–47; do not silently switch channel sets or remove channel47.

## 3. Established full-period reference

- Train nominal windows from **2000–2005**; calibrate on **2006**.
- One frozen model/rule evaluated continuously over **all months 2007–2013**.
- Local: **F0.5 0.862907**, TP53/FP7/FN14, 67 annotated event IDs.
- System: **F0.5 0.805722**, TP54/FP7/FN37, 91 annotated event IDs.
- Main run: `results_longrun/long_baseline/ch40_47_train2000_2005_seed42`.
- See [dual-scope evaluation](docs/research/dual-scope-evaluation.md) and the
  baseline/reproduction reports indexed in [docs/README.md](docs/README.md).

This is a retrospective Transformer autoencoder, not a causal forecaster.
Later-period covariates and test results have been inspected: not untouched data.
Local >0.85 is neither a system score >0.85 nor evidence of robust generalization.
Do not directly compare full-period scores with small internal-pilot scores.

## 4. Model, metric and split rules

`esa_thesis/models.py:MultivariateAE` already uses temporal AND channel Transformer
encoders/decoders (1,333,392 parameters for eight inputs). Do not suggest “adding
Transformers” as though the baseline were a dense AE. The residual forecaster is
a different model; its earlier negative result is documented separately.

Correct evaluator: `esa_thesis/evaluation/esa.py`, version `esa-ew-id-duration-v1`.
It scores Anomaly + Rare Event annotation IDs and penalizes nominal alarm duration:

P = TP/(TP+FP) * (1 - nominal_false_alarm_seconds/nominal_seconds)
R = TP/(TP+FN)
F0.5 = 1.25*P*R/(0.25*P+R)

Use evaluator zero-denominator/neutral-annotation conventions and parity tests;
do not substitute pointwise F0.5 or the legacy evaluator. The selection cap is
**1% of all alarm samples**, not 1% of nominal false-alarm time.

Historical development folds2003/04/05: expanding training before each year,
Jan–Mar calibration, Apr–Dec assessment. Later internal pilots use:

| Fold | Training | Calibration | Real validation |
|---|---|---|---|
| 2001 | 2000 | Jan–Jun2001 | Jul–Dec2001 |
| 2002 | 2000–2001 | Jan–Jun2002 | Jul–Dec2002 |

Internal neural runs: five epochs, 25k nominal windows, seeds42/43/44, batch64,
short_no_roll augmentation, training-only median/IQR scaler, clipping ±10,
window256/stride16, score stride32, window-mean overlap aggregation then channel
maximum. Calibrate checkpoints/thresholds only on calibration, then freeze.
The primary policy is calibration_selected; fixed_epoch5 is secondary and cannot
be substituted after seeing its validation results. These periods have repeatedly
informed development and must not be presented as independent confirmation.

## 5. Recent completed experiments (do not restart)

Paths below are under `results_longrun/development/`. Scores are mean system F0.5
across seeds unless deterministic; all use the internal periods above.

| Experiment / folder | 2001 | 2002 | Decision |
|---|---:|---:|---|
| Joint control `inner_selection_cuda_v2` | 0 | 0 | Retain as matched internal control |
| Loss weight0.1 `inner_lambda01_cuda_v1` | 0 | 0.002600 | Failed; 2002 meanFP53→159 |
| Isolated channels `inner_isolation_cuda_v1` | 0 | 0 | Failed; 2002 meanFP53→128.67 |
| Score-change filter `score_change_v1` | 0 | 0.238095 | Failed; only seed44 detects id25 |
| Direct telemetry `telemetry_reference_v1` | 0 | 0 | Deterministic; silent both years |

Score-change mean hides seeds42/43=0 and seed44=0.714286 system/0.833333 local.
False alarms disappeared but recall was not reliable. No promotion/outer run.
The direct reference is one run per year, not three artificial replicates.

Reports, full CSVs and plots:
- [Internal selection](docs/research/inner-selection-results-2026-10-02.md).
- [Loss weight](docs/research/loss-weight-results-2026-10-03.md).
- [Channel isolation](docs/research/channel-isolation-results-2026-10-08.md).
- [Score change](docs/research/score-change-results-2026-10-08.md).
- [Direct telemetry](docs/research/telemetry-reference-results-2026-10-08.md).

**Checkpoint warning:** isolated-model tensors have the same keys as the joint AE,
but replay requires `ChannelIsolatedAE` from `esa_thesis/channel_isolation.py`.
Loading into MultivariateAE silently changes attention semantics. Isolation keeps
nominal parameter count but changes effective capacity; shared training losses
still couple updates. Do not describe it as eight separately fitted models.

## 6. What the diagnostics established

- Channel47's normal level changes and reconstruction errors rise in2002.
  Frozen-model interventions show its input level can affect other outputs.
  Correcting it reduces other-channel errors in most, not all, models. This is
  model sensitivity, not spacecraft causality; centering may hide real anomalies.
- Calibration detects a narrow subset of events. Later event scores can be much
  smaller; in2002 many nominal scores exceed missed-event peaks.
- Missed local validation events are not clipped on their annotated channels.
  Use **v2** event-input statistics; v1 used ID-union intervals and is historical.
- Removing the alarm cap improves6/10 calibration cases in2001 and0/10 in2002.
  19/20 grids offer no higher-recall row under the cap. No universal cap fix.
- No evidence supports another full-period training run for these candidates.

Read [channel intervention](docs/research/channel-level-coupling-2026-10-03.md),
[missed-event audit](docs/research/inner-event-audit-2026-10-08.md), and
[calibration audit](docs/research/calibration-selection-audit-2026-10-08.md).
Earlier broader inputs, reconstruction-only, specialist/OR alarms, masking,
observation-age and augmentation experiments are documented in the docs index.
Their results remain preserved; do not repeat them without reviewing those reports.

## 7. Code and artifact map

- `esa_thesis/`: active research; root `setup.py` is legacy TimeEval, not this package.
- `esa_thesis/inner_validation.py`: source-hashed shared pilot runner.
- `isolation_comparison.py`, `loss_weight_comparison.py`: wrappers using that runner.
- `score_change.py`, `telemetry_reference.py`: fixed statistical transforms.
- `scripts/research/`: diagnostics, report verification and transfer tool.
- `tests/thesis/`: regression/evaluator tests; 98 tests passed again on12 October2026.
- `docs/research/`: protocols/results; `tables/`: compact CSVs/plots/provenance.
- `results_longrun/`: checkpoints, optimizer states, arrays, full run manifests.
- `data/`: raw/prepared datasets and annotations; copy it, do not regenerate casually.
- `outputs/`, `output/`: presentations/workbooks and other artifacts; preserve both.
- `migration/`: local transfer manifest and prior environment snapshots; copy it.

Older protocols contain absolute source paths and immutable source hashes. Some
scripts have hard-coded output names and deliberately refuse existing directories.
Read code/CLI before executing. Relocation can break strict resume/hash checks;
never rewrite original provenance to force a resume. Create a new run or explicit
relocation adapter only when needed and document it. Replaying saved results does
not require resuming training. Historical finish scripts/PIDs are not migration jobs.

## 8. Transfer and reproducibility cautions

See [MIGRATION.md](MIGRATION.md). Copy **everything**, including `.git`, untracked
files and ignored datasets/results. This checkout has substantial local work;
GitHub-only restore is not guaranteed to contain it. Release backups are partial snapshots; the12October additive archive covers
the recent campaigns explicitly listed in its manifest. The12October source and additive artifact backup are documented in
[GitHub backup](docs/research/github-backup-2026-10-12.md); datasets and older
artifacts outside the archive still require the full folder/earlier releases. Keep the original until destination verification succeeds.

The source is Linux/Python3.9, PyTorch2.6.0+cu124, environment named timeeval.
Create a fresh destination environment (e.g. esa-thesis); names need not match.
Follow README/requirements-test.txt; don't copy Conda or run `pip install -e .`.
Source doctor on12October passed and detected CUDA/RTX3060, with an NVML warning;
this was not a GPU training test. Logs are under migration/.
Two artifact-build node_modules links point to the old Codex runtime; reinstall
artifact dependencies for rebuilding (finished artifacts remain portable).
See MIGRATION.md for all five external symlinks.
CPU tests are supported; check CUDA on the new PC before GPU work. macOS/native
Windows setup has not been validated here; Linux/WSL is the documented route.

Prior CSV decoding/label failures led to strict integrity checks. The prepared
training golden SHA256 is
`fe6812dd5d324fe2eb347a703894551f0a03f40253742a3cd35f42d02fdb80ad`.
Do not bypass corruption with permissive decoding or silently coerce invalid labels.

## 9. Full history and new-chat prompt

The previous1200-line handoff is preserved verbatim in
[historical handoff](docs/history/AI_HANDOFF-through-2026-10-08.md), including
supervisor references, early experiment results, restore order and detailed
chronology. It contains superseded plans/statuses. This document and the latest
verified result reports take precedence; history is evidence, not pending work.

Paste into the new chat:

> This is my ESA satellite anomaly-detection thesis project. Read AGENTS.md,
> AI_HANDOFF.md and MIGRATION.md first. We have moved computers: verify the copied
> manifest before changing files, then check the environment. Preserve all results.
> Read the latest linked reports and continue from the current research decision,
> not an old unfinished plan. Explain findings simply; do not ask me to repeat
> project history or automatically start another training sweep.
