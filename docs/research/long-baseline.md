# Long-period eight-channel Transformer baseline

## Fixed protocol

User-authorized experiment; channels 40–47, pseudo-anomaly Transformer AE:

| Role | Interval (end exclusive) |
| --- | --- |
| Train | 2000-01-01 to 2006-01-01 |
| Calibrate | 2006-01-01 to 2007-01-01 |
| Assess frozen model | 2007-01-01 to 2014-01-01 |

Use every eligible globally nominal training window at stride 16 and length 256.
The historical 250,000-window cap is disabled explicitly for this command only.
All 87 paired label columns determine nominal status. This does not mean using
annotated periods for training, every possible stride-1 window, or fitting on test
data. Scaling is fitted on nominal training observations only. Existing prepared
CSV transformations remain unchanged. This is a longer baseline experiment, not
the result of a completed feature-selection study.

Defaults: 40 epochs, batch size 64, seed 42, float32. Calibration F0.5 selects
checkpoint and rule every five epochs and at the final epoch; earliest ties win.
The existing fixed threshold/postprocessing grid and prediction-rate constraint
are retained. The decision is saved before assessment values are loaded.

The primary result is the pooled 2007–2013 corrected event-wise F0.5. Annual rows
slice the same continuous prediction and frozen rule, without yearly tuning.
Year-boundary events can appear in adjacent yearly evaluations; do not sum annual
event counts or average annual F0.5 and call that the pooled result. Reconstruction
is retrospective, not an established real-time detector.

This changes both training years and window budget versus historical runs. Gains
cannot be attributed solely to longer duration. Test-period covariates and earlier
test results have been inspected; disclose this and do not call it untouched.

## Current data gate

The test CSV was restored on 23 September from independently checked raw-source
values. The reconstruction and installed file exactly match the historical golden
SHA256; the damaged original is preserved. The data gate is cleared. See
[integrity investigation and verified recovery](dataset-recovery-2026-09-23.md).
The actual `long-baseline --verify-data` preflight passed for both files after
restoration. The command below rechecks both CSV hashes before every start/resume.

## Run from the server GPU environment

If outside tmux, first start `tmux new -s esa-long-baseline`. If already inside
 tmux, use the existing session. Then:

```bash
conda activate timeeval
python -m esa_thesis long-baseline \
  --output results_longrun/long_baseline/ch40_47_train2000_2005_seed42 \
  --device cuda --batch-size 64 --epochs 40 --seed 42
```

Detach with Ctrl-b then d. Reattach with `tmux attach -t esa-long-baseline`.
Epoch checkpoints are saved. To resume, use the same command with `--resume`.
Keep source code, input data and settings unchanged while running: strict resume
provenance rejects changes. Existing output is never silently overwritten.

`--dry-run` prints the protocol without loading data or creating output.
`--verify-data` verifies the known whole-file CSV checksums without training or
creating output. Normal execution verifies both checksums before creating a run.
A failed checksum blocks training; never work around it with ignored parse errors.

## Outputs

- `protocol.json`: frozen settings, source hashes and input fingerprints.
- `full_period/history.csv`: losses and calibration scores.
- `full_period/checkpoint_last.pt`: complete epoch recovery checkpoint.
- `full_period/model_checkpoint.pt` and `frozen_rule.json`: selected model/rule.
- `full_period/assessment_score.npy` and `assessment_prediction.npy`.
- `pooled_results.csv` and `yearly_results.csv` (2007–2013).

The assistant session has no CUDA access; no full GPU training has been launched.
The 44-test suite passed, including a small end-to-end separate-test-file run,
no-cap nominal-window selection, dry-run boundaries and checksum rejection.
