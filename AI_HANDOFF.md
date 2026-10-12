# AI handoff — ESA satellite anomaly-detection thesis

Updated **12 October 2026**. This is the current entry point for a new assistant.
Read this before proposing work. The chat is not required to continue the project.

## 1. Current task and decision

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
