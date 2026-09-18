# AE channel-coverage experiment

## Question

Does adding channels 14, 21, and 29 improve the existing pseudo-anomaly Transformer
AE across the three development folds without an unacceptable increase in false
alarms? The original model receives channels 40–47. Three missed 2004 event IDs
were annotated only on the added channels; this motivated the predeclared expansion.
It does not guarantee those events will be detected.

This choice was informed by development assessment errors, so the resulting scores
are further development results, not a fresh independent confirmation. The benchmark
test CSV and final 2006 validation interval remain unused.

## Controlled settings

`--channel-set baseline` (the default) retains the original eight channels.
`--channel-set expanded` preserves their ordering and appends 14, 21, and 29.
The global experiment list is not mutated. The selected list is recorded in both
the protocol and checkpoint.

The expansion uses the same AE implementation, pseudo-anomaly objective, 40 epochs,
batch size 64, seed 42, normal-window policy, scaler, window-mean scoring,
calibration grid, 1% calibration predicted-rate cap, and checkpoint selection rule.
The model necessarily has more channel embeddings and more channel tokens; its
compute cost and initialization are not numerically identical to the eight-channel
model. Calibration may select a different epoch and threshold as intended.

All folds still evaluate every supplied Anomaly and Rare Event ID in the period,
not only annotations on input channels. All label channels still determine nominal
training windows, so adding features does not relax the anomaly exclusion policy.
The AE remains retrospective; its metrics must not be described as causal streaming
performance.

## Run all fixed folds

From the updated project folder:

```bash
conda activate timeeval
python -m esa_thesis develop \
  --channel-set expanded \
  --folds 2003 2004 2005 \
  --output results_longrun/development/ae_expanded_seed42 \
  --device cuda --batch-size 64 --epochs 40 --seed 42
```

The folds run sequentially. Append `--dry-run` to inspect the configuration without
training or creating output. To run separately, specify one fold and a different
output directory per run. Existing output directories require `--resume`. Completed folds are skipped.
New runs save an atomic full checkpoint after each epoch, including optimizer,
scheduler, random states and the selected model. A disconnect loses at most the
in-progress epoch. Use identical arguments plus `--resume` to continue.

Old partial folds from before this feature have only selected model weights and
cannot resume faithfully. Add `--restart-incomplete` to preserve those files under
`interrupted/` and retrain only the incomplete fold from epoch 1. Completed folds
remain untouched. Original protocol files are retained, and a separate recovery
manifest records the upgraded implementation. Settings/data/source mismatches are
rejected; GPU bitwise determinism is still not guaranteed.

## Compare with the frozen baseline

| Fold | Baseline F0.5 | Detected / total | False-positive events |
| --- | ---: | --- | ---: |
| 2003 | 0.833170 | 5 / 6 | 1 |
| 2004 | 0.428131 | 6 / 10 | 9 |
| 2005 | 0.869055 | 4 / 7 | 0 |

Report all three folds, including regressions. Examine precision, recall, nominal
alarm duration, and whether the specific missing IDs are recovered; do not rely
only on the best fold. A gain on 2004 accompanied by losses elsewhere is not a
uniform improvement. Sparse calibration and the single random seed remain limitations.
No architecture, threshold grid, or alarm policy should be changed during this run.

## Verification

The complete suite passes 28 tests, including an explicit baseline/expanded CLI
check that verifies no mutation of the original channel set. A separate synthetic
one-epoch run exercised the 11-channel AE, pseudo-anomaly loss, scaler, checkpoint
save/load, calibration, and assessment with the benchmark test file absent.
Real-data GPU training has not been launched by the preparation step.

## Recover the interrupted September run and survive logout

`tmux` is installed on the current Linux server. Start a persistent session:

```bash
tmux new -s esa-expanded
```

Inside that session, run:

```bash
conda activate timeeval
cd ~/projects/ESA-ADB
python -m esa_thesis develop \
  --channel-set expanded --folds 2003 2004 2005 \
  --output results_longrun/development/ae_expanded_seed42 \
  --device cuda --batch-size 64 --epochs 40 --seed 42 \
  --resume --restart-incomplete
```

For this legacy run, 2003 and 2004 are complete and are skipped; 2005's partial
31-epoch run is archived and restarted because no optimizer state was saved.
Detach using **Ctrl+B, then D** before logging out. Reconnect with
`tmux attach -t esa-expanded`. Do not start duplicate sessions for the same output.
Tmux protects against terminal logout, not a server reboot; epoch checkpoints
provide recovery after process/server failure. Future retries need only `--resume`.
