# Short anomaly generator without temporal rolls

## Hypothesis and matched comparison

The 2004 training-only diagnostic found that short rolls produce almost the same
reconstruction error as their paired normal windows, while retaining a large
separation penalty. This motivates an ablation; it does not establish causality
or imply that the new method will improve F0.5.

Use `augmentation-comparison --comparison short-no-roll`:

- Control: existing `short_coordinated`, with 50% legacy and 50% short corruptions.
- Candidate: `short_no_roll`, with the same mixture, but the short branch chooses
  offset or noise with equal probability. The legacy branch still includes its
  original temporal rolls. Durations, channel-count laws and amplitude scaling
  remain unchanged.
- Same initialization, nominal windows, scaler, epoch-addressed shuffle/dropout,
  batch64, five epochs and a 25,000-window cap in each arm.
- Same channels40–47, window-mean score, threshold/postprocessing grid, 1% alarm
  fraction cap, calibration-only checkpoint selection and earliest ties.
- Corruption draws need not stay paired after a changed type choice consumes
  different randomness. Other stochastic streams remain isolated from corruption.
- Freeze both rules before each assessment; report identical predictions under
  local and system event scopes.

The original all-legacy baseline remains a separately reported historical
reference. Rerunning the current short-coordinated control makes the direct
ablation contemporaneous without adding a third training arm. Compare its
results with the earlier matching control and investigate material discrepancies.

## Campaign fixed before assessment

Run seeds42/43/44 on all three folds2003/04/05. Each fold trains on preceding
years, calibrates January–March, and assesses April–December. These are already
inspected development periods, not a new untouched final test.

First run seed42, then repeat unchanged with seeds43/44 and matching new output
names. Do not choose a seed, tune parameters or stop for a favorable individual
assessment score. Technical failures can stop the campaign for investigation.
Report every paired result, nominal loss, synthetic penalty, calibration scores,
local/system F0.5, recall and false alarms. A single improved fold is insufficient
for promotion. No 2007–2013 run is included in this pilot.

## Command

Inside an existing tmux session:

```bash
conda activate timeeval
cd ~/projects/ESA-ADB
python -m esa_thesis augmentation-comparison \
  --comparison short-no-roll \
  --folds 2003 2004 2005 \
  --output results_longrun/development/short_no_roll_seed42_v1 \
  --device cuda --batch-size 64 --epochs 5 \
  --max-train-windows 25000 --seed 42
```

For an interrupted run, repeat the identical command with `--resume`. This
restores completed epochs and skips completed folds; an interrupted epoch
restarts from its preceding complete checkpoint. Do not add `--resume` for a
directory that does not yet exist. Keep source/data/settings unchanged while a
run is active. Do not resume the older comparison directories under the new code.

## Verification and execution status

- 80 thesis tests passed, including new-mode support/branch checks, full
  interrupted-versus-uninterrupted CPU training for the new arm, and freezing both
  new-comparison rules before assessment.
- Existing `legacy` and `short_coordinated` outputs and Python/Torch RNG states
  match their pre-change HEAD implementation exactly for seeds42/43/44.
- CLI dry run confirms the intended arms, folds and budgets. CUDA unavailable
  in the assistant session; no real-data training launched here.
- Existing modes and default comparison remain unchanged. New outputs preserve
  earlier evidence. No GitHub push performed.
