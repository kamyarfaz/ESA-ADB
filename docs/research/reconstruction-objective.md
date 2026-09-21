# Transformer AE: reconstruction-only objective ablation

## Question and fixed protocol

Does the pseudo-anomaly objective improve the existing Transformer autoencoder?
The new `develop --objective reconstruction` option disables only the synthetic
corruption forward pass and its margin penalty. It minimizes nominal window
reconstruction MSE. The default remains `--objective pseudo-anomaly`.
Both options train `MultivariateAE`; this is an objective comparison, not a new
architecture. No performance improvement is claimed before running it.

Use baseline channels 40–47, folds 2003/2004/2005, seed 42, 40 epochs, batch size
64 and float32. Keep the existing training-only scaler, nominal-window selection,
optimizer, learning-rate schedule, window-mean scores, channel maximum, threshold
quantiles, gap/duration candidates and calibration-only checkpoint selection.
Report every fold, not just the best one. These are development assessments
already inspected during research; they are not an independent final test.
The benchmark test and reserved final validation interval remain unused here.

Same seed preserves initialization and initial data preparation. Removing the
corruption pass changes random-number consumption (including dropout and later
shuffle sequences), so this is not a bitwise paired stochastic training trajectory.
It also reduces computation per epoch; the comparison fixes epochs and nominal
training examples, not GPU time. The AE remains retrospective.

## Run

Inside an existing tmux session, run the following. If outside tmux, first use
`tmux new -s esa-reconstruction`. Do not nest tmux sessions.

```bash
conda activate timeeval
python -m esa_thesis develop \
  --objective reconstruction --channel-set baseline \
  --folds 2003 2004 2005 \
  --output results_longrun/development/ae_reconstruction_seed42 \
  --device cuda --batch-size 64 --epochs 40 --seed 42
```

Detach using Ctrl-b then d before logout. Add `--dry-run` to inspect the plan
without training. Use a new directory for the first run. After interruption,
repeat the exact command with `--resume`: full epoch checkpoints restore training
and completed folds are skipped. Do not change code, data or settings between
start and resume. A different objective cannot resume the original run.

## Comparison

Compare with the saved pseudo-anomaly baselines:

| Fold | Existing output | Assessment F0.5 | TPe / FPe / FNe |
| --- | --- | ---: | --- |
| 2003 | `ae_baseline_2003_seed42/2003` | 0.833170 | 5 / 1 / 1 |
| 2004 | `ae_baseline_remaining_seed42/2004` | 0.428131 | 6 / 9 / 4 |
| 2005 | `ae_baseline_remaining_seed42/2005` | 0.869055 | 4 / 0 / 3 |

All paths above are under `results_longrun/development/`. Compare event precision,
recall, false-positive duration, selected epoch and calibration performance as
well as assessment F0.5. Keep per-fold failures visible. Do not change thresholds
using assessment labels, or interpret a single ≥0.85 fold as robust success.

The protocol records the objective as `spec.pseudo`, with a distinct run/group
for reconstruction-only training. Existing defaults and old experiment outputs
are preserved. Tests cover explicit CLI selection, rejection of objective changes
on resume, and a synthetic reconstruction-only run that fails if pseudo-anomaly
generation is called. The workflow verifies decisions freeze before assessment.
