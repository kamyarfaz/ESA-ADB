# Corrected evaluation implementation and diagnostic baseline

Implemented/validated: 2026-09-16. This supersedes the earlier audit's description
of the **active** metric; historical reports retain their original values.

## Implemented

- `evaluation/esa.py`: fast event-ID and timestamp-duration evaluator, matching
  the event-wise portion of the pinned upstream ESA `ESAScores`. It implements
  the multiplicative nominal-time correction, overlapping/multipart event IDs,
  category exclusions, interval boundaries, and redundant-alarm precision.
  It does not implement affiliation or channel-localization metrics.
- `metrics.py`: requires an annotation evaluator; authoritative `esa_f05` and
  event counts come from that evaluator. Historical binary-mask event metrics
  are explicitly prefixed `legacy_`. Point-level diagnostics remain separate.
- `protocol.py`: last-three-calendar-month validation, data/source/configuration
  provenance, and refusal to resume unversioned or incompatible cached runs.
  File size/mtime signatures are not a substitute for dataset integrity checks.
- `thresholds.py`: validation-only grid and selection; no test input in the grid.
  A no-alarm candidate is included. The retained 1% validation predicted-rate cap
  is an experiment policy, not a universal ESA requirement.
- `training.py`: both AE and MLP checkpoint selection use corrected validation
  scores; final test metrics use the same evaluator. The default output root is
  `results_longrun/mission1_esa_ew_v1` to isolate new results from old caches.
- `recalibrate`: recalibrates cached AE/MLP scores, freezes each run's decision
  rule before opening its test scores, and writes a new output directory.
- Historical evaluation CLI commands require `--allow-legacy-metric` and warn
  that their output is historical. They remain available for reproduction only.

Scoring architecture and reconstruction behavior were deliberately not changed.
The full-series min-max ensemble remains experimental and is not a supported
input to corrected recalibration. Retrospective scoring, missing-value backfilling,
and previous test exposure remain limitations. This is not a causal deployment
certification or a claim of a fresh independently held-out benchmark result.

The reference implementation has undefined/fragile empty-event affiliation cases;
the new event-only evaluator explicitly returns zero when no positive events are
detected. With zero nominal duration it applies no nominal-time penalty. That
last convention is a documented extension, not an assertion about upstream's
zero-division behavior.

## Verification

Install `requirements-test.txt` for the pinned `portion` reference dependency.

```bash
PYTHONPATH=. python -m unittest discover -s tests/thesis -p 'test_*.py' -v
PYTHONPATH=. WANDB_MODE=disabled python tests/thesis/check_refactor.py
PYTHONPATH=. WANDB_MODE=disabled python tests/thesis/smoke_training.py
```

Passed:

- 17 unit tests, including the paired scoring workflow and 32 randomized regular/irregular-time parity cases
  against unmodified upstream code and boundary/category/empty-prediction checks.
- Comparison against the previous official audit on all 31 real saved masks:
  F0.5, precision, recall, and alarming precision agree within `1e-12`.
- Synthetic one-epoch AE+MLP training, selection, final evaluation, checkpoint
  resume, and incompatible-cache rejection. This was not full ESA training.
- Independent upstream evaluation of the recalibrated `pa_center_ch12_38_49`
  prediction mask, also agreeing within `1e-12`.

## Reproduce the diagnostic recalibration

```bash
python -m esa_thesis recalibrate \
  --source results_longrun/mission1_reconstruction_ae_sweep_optimized/20260519_155620 \
  --output results_longrun/audits/my_new_corrected_baseline
```

Existing output directories are rejected. `--runs NAME ...` restricts the run set;
`--score-kind mlp` selects cached MLP scores. `--validation-window cached` is an
explicit alternative historical-window diagnostic, not the benchmark split.

The completed AE baseline is at
`results_longrun/audits/f05_corrected_baseline_2026-09-16/` (local, ignored by Git).
Each run has a validation-only candidate table, frozen rule, output masks,
results, and provenance. The combined table is `results.csv`.

### Observations, not test-based model selection

All 31 cached AE runs were evaluated. The calibration interval is 2006-10-01 to
2007-01-01 with **five event IDs**; test is 2007-01-01 to 2014-01-01 with 91 IDs.

| Run | Validation F0.5 | Test F0.5 | Corrected test precision | Test recall |
| --- | ---: | ---: | ---: | ---: |
| `pa_center_ch12_38_49` | 0.8818 | 0.8380 | 0.9096 | 58/91 |
| `center_ch08_40_47` | 0.8799 | 0.7733 | 0.9985 | 37/91 |
| `pa_center_ch08_40_47` | 0.8821 | 0.7617 | 0.8567 | 48/91 |

The first row is the **largest observed test score**, not a justified final model
choice. Ranking solely by validation F0.5 selects `pa_center_ch08_40_47`, with test
score **0.7617**. The small validation event count and near ties show why earlier
chronological development folds are needed for model selection. None reached 0.85.

The old fixed mask for `pa_center_ch12_38_49` scored 0.7378 in the official audit;
its new mask scores 0.8380. This experiment changes the metric used for calibration,
the validation window, and the threshold/postprocessing grid together. It is not
an isolated causal estimate of the benefit of fixing the precision equation.
Weights/checkpoints remain those selected with the historical metric.

## Dataset issue discovered and repaired

The original local training CSV had two malformed numeric digits: a non-ASCII
byte in `channel_32` and `v` in a `channel_51` value. Both were restored from raw
telemetry, with source ZIP CRC checks and preprocessing-rule verification.
The repaired CSV is installed, and the original is preserved under `data/repairs/`.
Full comparison verified exactly two changed bytes; both train and test passed
complete structural validation. See [repair evidence and hashes](scoring-comparison.md#dataset-repair-completed-2026-09-16).

Cached-score recalibration reads only timestamp fields and warns on non-ASCII
content elsewhere. That workaround never established feature validity. The normal
training loader still parses all numeric features strictly. Initial temporary-copy
differences remain documented; the final installed copy passed validation and
checksum comparison. Full structural validation does not certify that every finite
value equals raw telemetry.

## Next controlled experiment

The [paired scoring command](scoring-comparison.md) compares current window-averaged
scoring with per-timestep residuals using the same weights and validation protocol. Use earlier
time folds to select channel groups and calibration settings. A causal patch
forecaster remains the next architecture experiment; 0.85 is not yet established.
