# Hidden-target overlap pilot

Prepared 25 September 2026; GPU experiment not launched by the assistant.

## Predeclared comparison

Keep inputs 40–47. Compare `masked` (original uniform segment placement) with
`masked_overlap` (uniform placement among valid starts that touch at least one
hidden target). Preserve the uniform segment-length distribution, corruption
types, number of channels, amplitudes, model, objective and inference procedure.
Temporal rolls on flat data can still be no-ops: this experiment guarantees
geometric overlap, not a change in hidden target values. It does not restrict
the entire segment to hidden patches. Placement changes random-number use, so
the two arms do not receive identical synthetic examples.

Use 25,000 nominal training windows before 2004, seed 42, five epochs per arm,
batch size 64. January–March 2004 selects each checkpoint/rule independently;
freeze both before April–December development assessment. This is a mechanism
pilot on an already-explored fold with just one calibration event, not independent
confirmation or a robust target claim. No 2007–2013 evaluation.

Record normal reconstruction MSE, synthetic MSE and synthetic penalty separately
in history.csv (batch means, matching the existing total-loss logging convention).
Report system and input-channel metrics from identical frozen predictions.

Consider continuation only if the candidate improves development system F0.5
without increasing false-positive events versus the matched control; report
recall and nominal false-alarm duration too. Meeting this exploratory criterion
does not replace the baseline: further predeclared chronological checks are
needed. Otherwise stop this variant. Do not choose a different arm per year,
retune to the assessment results, or automatically increase epochs.

## Run

Inside an existing tmux session, run directly. From a normal terminal, first use
`tmux new -s esa-overlap`; detach with Ctrl-b then d.

```bash
conda activate timeeval
python -m esa_thesis masked-pilot \
  --comparison hidden-overlap --folds 2004 \
  --output results_longrun/development/masked_overlap_2004_seed42_v1 \
  --device cuda --batch-size 64 --epochs 5 --max-windows 25000 --seed 42
```

Add `--resume` to the same command after interruption. Protocol/source hashes
must still match. Preserve old v1/v2 pilot results; this version's source change
prevents resuming them with the new code. New directories are required.

Validation: 59 thesis tests passed, including constrained placement, hidden-input
isolation, a real CPU training/checkpoint/resume check for the new arm, and
evaluation tests. The proposed command passed `--dry-run`. No full GPU result yet.
