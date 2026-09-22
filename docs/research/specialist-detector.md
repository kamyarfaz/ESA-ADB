# Separate channel-group Transformer experiment

## Predeclared comparison

Research resumed on 22 September 2026; migration is postponed. Train a separate
pseudo-anomaly Transformer AE on channels 14, 21 and 29, retaining the saved
baseline on channels 40–47. The model class, training objective, seed 42,
40 epochs, batch size 64, chronological folds, scaler and retrospective scoring
remain unchanged. The new model is smaller because its channel embedding and
channel attention operate on three channels. The original model is not retrained.

This tests whether missing-channel coverage can be added without the shared-model
interference seen in the eleven-channel experiment. It does not guarantee a
physical explanation or better F0.5. All folds are already inspected development
data, and calibration is sparse. No benchmark test or final validation is used.

## Training command

Inside tmux (do not nest another tmux session if already inside one):

```bash
conda activate timeeval
python -m esa_thesis develop \
  --channel-set specialist --objective pseudo-anomaly \
  --folds 2003 2004 2005 \
  --output results_longrun/development/ae_specialist_seed42_verified \
  --device cuda --batch-size 64 --epochs 40 --seed 42
```

Use a new output directory initially. After interruption, repeat the exact command
with `--resume`. Keep code, data and settings unchanged between start and resume.
Add `--dry-run` to inspect the plan. Training selects the specialist checkpoint
and its standalone decision using corrected calibration F0.5, as for the baseline.
It does not search checkpoints again using combined assessment performance.

## Combined decision: calibration only

After all specialist folds complete:

```bash
python -m esa_thesis combine-specialist \
  --specialist-root results_longrun/development/ae_specialist_seed42_verified \
  --folds 2003 2004 2005 \
  --output results_longrun/development/combined_specialist_seed42 \
  --device cuda --batch-size 64
```

The command reloads both selected checkpoints and recomputes calibration scores.
It verifies original calibration F0.5 within 1e-6 and checks source protocols,
channel order, architecture/training settings and the metric version. It does not
change the baseline threshold or postprocessing. Candidates are:

1. Baseline alone: an explicit off switch for the specialist branch.
2. Baseline OR specialist alarms. Specialist thresholds use the same fixed
   calibration quantiles; merge gaps 0/16/64 and minimum durations 1/8/32 are
   applied to the specialist mask before the OR. There is no post-OR gap filling.

Admissible combined masks have at most 1% predicted calibration samples and no
more false alarm events than the baseline on calibration. Maximize combined
calibration F0.5. Equal F0.5 favors baseline alone; remaining ties favor less
nominal alarm duration, higher threshold, smaller gap and smaller minimum duration.
The off switch is literal and remains off on assessment, even if assessment scores
exceed the calibration maximum. This is stronger than using the maximum calibration
score as an approximation to off.

The specialist checkpoint is selected for standalone calibration performance;
only its branch threshold and postprocessing are adjusted for combination.
The standalone specialist retains its own original frozen rule in the comparison.
Combined decisions and source hashes are written before reading assessment scores
or metrics. The tool then verifies each original assessment mask/metric and reports
baseline, specialist alone and combined metrics for every fold. It never selects
among them using assessment. Its output directory must be new; combined scoring
has no resume implementation. Use a new directory if it is interrupted.

## Interpretation and limitations

The OR cannot remove an existing baseline alarm sample, so point coverage cannot
decrease. Event precision, false-alarm counts and durations can change, including
through joining intervals. The constraints hold on calibration only, not future
data. Report TPe/FPe/FNe and false-positive duration as well as F0.5. Do not claim
success from one fold or from a lower event count caused by longer merged alarms.
The baseline and specialist both remain retrospective, not certified streaming
systems. No real-data improvement is claimed before the new run completes.

## Verification and artifacts

Tests cover specialist features, safe-off ties, calibration false-alarm/rate
constraints, nonfinite inputs, and freezing combined decisions before assessment
arrays are read. Existing AE integration and recovery tests remain applicable.
40 thesis tests pass. The combination checks the existing baseline protocol for
all three folds. It needs the existing baseline directories named in AI_HANDOFF.md;
these are not command-line-selectable in this first fixed experiment.

Training outputs use the established AE layout. Combination outputs include both
calibration score arrays, all calibration candidates, frozen rule/source hashes,
three assessment masks, and per-fold plus combined result tables. Fresh artifact
backups should be made after the experiment completes. The earlier external-drive
manifest is stale after code changes; regenerate it if migration is requested again.

## Failed initial launch

The initial output `ae_specialist_seed42` contains a pre-training CSV read failure.
The commands above use a new directory after [dataset recovery](dataset-recovery-2026-09-22.md).
