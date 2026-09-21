# Transformer with a carried-forward level

## Hypothesis and fixed comparison

The September 21 range diagnosis motivates testing sensitivity to sustained
telemetry levels. `transformer_residual` subtracts each channel's last observed
value from all 256 context samples, predicts 16 deviations with the existing patch
Transformer, then adds the last value back. Loss and anomaly scores compare the
restored prediction against the original scaled targets. A target jump is not
subtracted from its own prediction. No future values determine the reference.

The architecture, parameter count and seeded initial weights match `transformer`.
The only treatment is centering historical inputs and carrying the reference
level into the output. No zero-initialized head or additional tuning is introduced.
Translation equivariance holds in model input space; the existing ±10 preprocessing
clip can break this property for shifted raw inputs. A persistent anomaly may
become part of later contexts, as with other adaptive forecasters. Recovery of
all anomaly types is not guaranteed.

Use channels 40–47, seed 42, 40 epochs and batch size 64, with the existing
calibration-only checkpoint/threshold selection, threshold-only alarms, and
corrected event metric. The first comparison is fold 2003 against the saved
`forecast_2003_seed42/2003/transformer` control. This is a diagnostic-driven
experiment on already inspected development data, not independent validation.
It does not directly retest channels 14/21 or isolate AE failure causes.
Keep the eight-channel AE as the working baseline. Do not change thresholds
based on assessment or claim success from calibration F0.5 alone.

## Run safely across logout

From the repository root, start a tmux session:

```bash
tmux new -s esa-residual
```

Inside that session:

```bash
conda activate timeeval
set -o pipefail
mkdir -p logs
python -m esa_thesis forecast-develop \
  --folds 2003 --models transformer_residual \
  --output results_longrun/development/forecast_residual_2003_seed42 \
  --device cuda --batch-size 64 --epochs 40 --seed 42 \
  2>&1 | tee logs/forecast_residual_2003_seed42.log
```

Detach with Ctrl-b then d; reconnect with `tmux attach -t esa-residual`.
Use a new output directory. This forecasting runner does not support checkpoint
resume; tmux protects against terminal logout, not server failure. The existing
AE resume command is a separate implementation.

Add `--dry-run` to inspect the protocol without training. Nothing needs to be
rerun for the existing control if its recorded settings match the above. If it
must be reproduced, explicitly request `--models transformer transformer_residual`
in a fresh directory; they run sequentially with the same per-model seed.

## Review before extending

Compare saved assessment event precision, recall, TPe/FPe/FNe, false-positive
duration, calibration selection, and parameter count with the existing control.
Report any recall loss alongside fewer false alarms. Do not combine this change
with the earlier causal confirmation filter in this first comparison. Subsequent
2004/2005 runs should retain the same declared settings and be reported together;
a fold-2003 gain alone does not establish the thesis target of robust F0.5 ≥0.85.
No performance improvement is claimed before this run completes.

## Verification

The thesis tests cover identical seeded initial weights, per-channel translation
equivariance, retained sensitivity to a new target jump, prefix causality, coverage,
and a synthetic full training/calibration/assessment run for this model. The
synthetic workflow verifies that the rule exists before assessment is loaded and
that the benchmark test CSV is not accessed.
