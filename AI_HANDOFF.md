# AI handoff — ESA satellite anomaly-detection thesis

Last updated: 22 September 2026. This is a portable summary of the working
conversation, not a transcript. Verify facts against saved artifacts if anything
has changed. Paths below are relative to this repository so the folder can move.

## Start here: current user intent

Migration was postponed and research resumed on 22 September 2026. The user
approved implementing the separate channels 14/21/29 Transformer and a combined
calibration workflow. Implementation and tests are complete; the user should run
the GPU experiment from `docs/research/specialist-detector.md`. No real training
has been started by the assistant and no specialist result is available yet.
The initial specialist run failed before training because of CSV corruption.
See docs/research/dataset-recovery-2026-09-22.md. The known-good CSV was restored;
use a fresh output ae_specialist_seed42_verified, including as the combination root.
No specialist training result exists yet.
The old transfer manifest is stale after these edits. Regenerate it only if
migration is requested again; the prior checks do not verify this changed tree.

The user values direct, practical help and explicit runnable commands. They have
asked for code, documentation and essential experiment artifacts to be backed up
to their GitHub repository. They explicitly agreed to exclude datasets from
GitHub because they can be downloaded again. Never confuse the essential GitHub
backup with the full working-folder copy.

Repository: https://github.com/kamyarfaz/ESA-ADB (default branch main).
Old structure/history is associated with previous_experiments. Do not reset or
replace branches. Local branch main was clean before this handoff was added.

## Thesis goal and provenance

- Satellite telemetry anomaly detection on ESA Mission 1.
- Supervisor Federico Butchellato supplied main.py, models.py and utils.py and
  asked for a Transformer-based model with F0.5 at least 0.85.
- Tentative title: Artificial Intelligence for Anomaly Detection in Satellite
  Telemetry under Space Operational Constraints. No final title or deadline.
- Code exists; a full LaTeX/PDF thesis manuscript has not been created here.
- Source review and attribution: `docs/thesis-research-context.md`.
- The supplied architecture is a starting point, not automatically an original
  thesis contribution. The user previously requested removal of a “supervised by
  Federico” line from the public README; do not reintroduce it there unprompted.

References supplied by the supervisor:
- https://arxiv.org/abs/2406.17826 (PDF: https://arxiv.org/pdf/2406.17826)
- https://ui.adsabs.harvard.edu/abs/2024arXiv240617826K/abstract
- https://github.com/kplabs-pl/ESA-ADB
- https://github.com/Google-Developers-Club-Guido-Carli/ESA-Spacecraft-Anomaly-Challenge
- https://openreview.net/pdf?id=FYEGPuUrpo
- https://www.researchgate.net/publication/375693831_Annotating_large_satellite_telemetry_dataset_for_ESA_international_AI_anomaly_detection_benchmark
- https://www.kaggle.com/competitions/esa-adb-challenge

## Important architectural correction

**The successful baseline already is a Transformer autoencoder.**
`esa_thesis/models.py:MultivariateAE` uses temporal and channel Transformer
encoders and decoders. Its eight-channel checkpoint strictly loads into the
current model; it has 1,333,392 parameters. An earlier assistant suggestion to
“add a Transformer reconstruction model” was mistaken and explicitly corrected.
The separate forecasting Transformer has 302,224 parameters and performs poorly.
Do not conflate forecasting with retrospective reconstruction.

## Evaluation and selection rules

Authoritative implementation: `esa_thesis/evaluation/esa.py`.
Metric version: `esa-ew-id-duration-v1`. It uses raw annotation-ID events,
Anomaly plus Rare Event categories, neutral treatment of other annotations,
and the official-style nominal-time penalty on event precision. Parity with the
bundled evaluator is tested. It is not sklearn pointwise F0.5 or point adjustment.

Chronological development folds in `esa_thesis/development.py`:

| Fold | Training ends before | Calibration | Assessment |
| --- | --- | --- | --- |
| 2003 | 2003-01-01 | Jan–Mar 2003 | Apr–Dec 2003 |
| 2004 | 2004-01-01 | Jan–Mar 2004 | Apr–Dec 2004 |
| 2005 | 2005-01-01 | Jan–Mar 2005 | Apr–Dec 2005 |

Intervals are [start,end). Expanding training starts in 2000. Calibration has
only 3, 1 and 4 selected events respectively; assessment has 6, 10 and 7.
Sparse calibration is a major limitation. Checkpoints and decision rules are
selected on calibration only, then frozen before assessment loading. All these
assessment folds have now informed development: do not call them untouched tests.
The benchmark test was examined in earlier historical comparisons; do not claim
it has never been accessed. Current development commands avoid it and the
reserved final 2006 validation interval.

AE baseline: channels 40–47, sequence 256, train stride 16, scoring stride 32,
training-only nominal median/IQR scaling with ±10 clipping, max 250,000 nominal
windows, pseudo-anomaly objective, float32, seed 42, batch 64, 40 epochs.
Score = per-channel window-mean squared residual, averaged across overlapping
windows, then maximum across channels. Fixed quantiles and gap/duration candidates
are in development.py. Select corrected calibration F0.5 every five epochs and
final epoch; earliest checkpoint wins ties, with 1% calibration prediction cap.
The reconstruction objective option disables only the pseudo-anomaly penalty.

## Verified experiment results

All table entries below are assessment event F0.5, not calibration scores.

| AE experiment | 2003 | 2004 | 2005 |
| --- | ---: | ---: | ---: |
| Baseline, pseudo-anomaly, channels 40–47 | 0.833170 | 0.428131 | 0.869055 |
| Expanded, append channels 14/21/29 | 0.909069 | 0.251365 | 0.148245 |
| Baseline channels, reconstruction-only | 0.525691 | 0.150184 | 0.868934 |

Baseline TPe/FPe/FNe: 2003 5/1/1; 2004 6/9/4; 2005 4/0/3.
Reconstruction-only: 2003 4/4/2; 2004 6/41/4; 2005 4/0/3.
Keep pseudo-anomaly training: removing it worsened this seed-42 comparison.
Tiny nominal reconstruction loss did not imply better anomaly detection.
Neither expanded channels nor reconstruction-only training gave consistent gains.
**The ≥0.85 target has not been robustly achieved across folds.**

Paths under `results_longrun/development/`:
- `ae_baseline_2003_seed42/2003`
- `ae_baseline_remaining_seed42/{2004,2005}`
- `ae_expanded_seed42/{2003,2004,2005}`
- `ae_reconstruction_seed42/{2003,2004,2005}`
- `forecast_2003_seed42/2003/{persistence,mlp,transformer}`
- `forecast_residual_2003_seed42/2003/transformer_residual`

2003 causal forecasts: persistence F0.5 0.042852; MLP 0.082774; Transformer
0.086696 (6 TP, 79 FP, 0 FN). Last-value-centered residual Transformer:
0.070614 (5 TP, 82 FP, 1 FN), selected epoch 15, calibration F0.5 0.057466.
It did not improve results; further residual forecasting runs were paused.
A separate causal alarm-confirmation experiment also failed to approach the AE.
Read `docs/research/forecast-development.md`, `causal-alarm-findings.md`, and
`residual-transformer.md` before proposing to repeat these experiments.

Earlier paired scoring with identical historical AE weights:
window_mean validation 0.882109, benchmark test 0.761655;
per_timestep validation 0.588188, benchmark test 0.671790.
Path: `results_longrun/scoring_comparison/paired_v1`.
Do not confuse the strong validation score with an achieved test target.

## Latest diagnoses and decisions

Read these reports in `docs/research/`:
- `expanded-channel-results.md`
- `channel-error-findings.md`
- `channel-regime-findings.md`
- `baseline-2004-findings.md`
- `reconstruction-objective.md`

Expanded 2004: 33 false alarms are dominated by original channels 41–46.
Expanded 2005: all 35 false alarms are dominated by channel 14 (8) or 21 (27),
all in October. Their sustained ranges are high relative to calibration.
Clipping is absent in channels 14/21/29 through 2005. Many false alarms lack
unusually large jumps. Score magnitudes overlap true-event alarms. This supports
investigating range sensitivity but does not establish physical causation.
Blindly dividing channel residuals by nominal error scale is not a demonstrated fix.

Baseline 2004: nine false alarms across six channels, lasting 16–80 minutes,
May–December. Nearest annotation is over 18 hours away. Peak scores overlap
true detections; simply raising threshold or duration is not a proven fix.
Missed IDs: id_91, id_92, id_94, id_97. Three are annotated only on 14/21/29,
outside baseline inputs; id_91 includes input channel 47. Missing channel coverage
is not proof of an absolute recall ceiling because correlated signals may exist.

Current approved research direction: a separate detector for channels 14/21/29,
retaining the eight-channel model. Implemented as `develop --channel-set specialist`.
`combine-specialist` retains the baseline rule and calibrates an optional specialist
OR branch: <=1% combined calibration prediction rate, no extra calibration false
alarm events, ties disable the branch. See `docs/research/specialist-detector.md`.
These constraints do not guarantee assessment performance. The specialist run and
combination have not yet been run on real data. Existing baselines remain intact.

## Code map and verification

- `esa_thesis/__main__.py`: CLI entry point (`python -m esa_thesis --help`).
- `models.py`: Transformer AE and legacy MLP.
- `development.py`: chronological AE experiments and objective/channel options.
- `development_recovery.py`: atomic full epoch checkpoints and strict resume checks.
- `forecasting.py`, `forecast_development.py`: causal forecasting experiments.
- `evaluation/esa.py`: corrected metric; `evaluation/compare_scoring.py`: scoring comparison.
- `scripts/research/`: read-only diagnoses and transfer verification.
- `docs/README.md`: research documentation index.

Full thesis test command: `python -m unittest discover -s tests/thesis -p 'test_*.py' -q`.
36 tests passed during migration preparation; 40 pass after the specialist workflow. The doctor passed dataset/header
checks; it does not certify full CSV contents or GPU training.
Python 3.9 environment, torch 2.6.0+cu124, numpy 1.26.4, pandas 2.3.3.
Dependencies and installation instructions are in README and requirements-test.txt.
Do not use the root pip editable install for the thesis; setup.py is old TimeEval.

AE `develop --resume` restores full epoch checkpoints and skips completed folds.
Legacy partial 2005 expanded run was archived under `interrupted/` and restarted
from epoch 1; the replacement finished. Forecast development has no resume.
Use tmux to survive logout. If already inside tmux, do not create a nested session.
Strict resume checks may reject relocation or code changes; never bypass them
silently. Completed runs do not need training resumed to read their results.

## Data integrity and migration

Full folder about 119 GiB; at least 150 GiB free recommended for a single copy.
Includes actual raw/prepared datasets and full local results, not just Git files.
Main datasets: `data/preprocessed/multivariate/ESA-Mission1-semi-supervised/`:
`84_months.train.csv` and `84_months.test.csv`, about 6.9 GiB each.
Official raw v1.0: https://zenodo.org/records/12528696 .
Mission1.zip MD5: 80750189d171f5f398fb3d96c49df12b.

Two corrupt numeric bytes in the train CSV were repaired using raw telemetry;
full validation followed. Preserve repair records and do not blindly rerun repairs.
Verified prepared-file SHA256:
- Train: fe6812dd5d324fe2eb347a703894551f0a03f40253742a3cd35f42d02fdb80ad
- Test: 9990c46dba0101074a19b7e58382bee787ea9a2051230a5b1716d207d8315f6c

`MIGRATION.md` explains full external-disk copy/archive, destination checksum
verification and environment recreation. `migration/manifest.json` is local,
ignored by Git, and describes the source snapshot. It is now stale because research code and Git metadata changed after migration
preparation. Regenerate it before any future transfer.
Verify the destination BEFORE running imports or Git that may alter caches.
Keep the original until destination verification reports zero failures.

Three absolute legacy W&B debug-log links have a preserved local target copy in
`migration/external-logs/`. They are not needed for training. Conda, credentials,
Codex chat history and supervisor originals outside this project are not included
merely by copying the folder. The project source/provenance documents are included.

## GitHub backup restore order

Source and instructions are on main. Essential binary artifacts are release assets:
1. https://github.com/kamyarfaz/ESA-ADB/releases/tag/migration-backup-2026-09-17
2. https://github.com/kamyarfaz/ESA-ADB/releases/tag/migration-update-2026-09-21
3. https://github.com/kamyarfaz/ESA-ADB/releases/tag/migration-update-2026-09-22

Restore and verify in that order; later snapshots overlay older files. Read each
release's RESTORE.md. September 22 adds 742 new/changed essential files, about
606 MB compressed, including reconstruction-only runs and diagnostic supplements.
Archive SHA256: 3033a1c1ffd70ede2e4e314d2bc619bc60074bb11b30c5e55925367752acd804.
Uploaded digest matched the local archive. Datasets, some older large score arrays,
and environments are deliberately not in these releases. The external-drive copy
contains more than the GitHub essential backup.

## First message to a replacement assistant

“Read AGENTS.md, AI_HANDOFF.md and MIGRATION.md in this project. We moved servers
and may have lost the old chat. Check whether migration actually happened. Research resumed with an approved
three-channel specialist experiment; read its protocol and inspect saved outputs
to determine whether it has run. Do not delete artifacts or assume results exist.”
