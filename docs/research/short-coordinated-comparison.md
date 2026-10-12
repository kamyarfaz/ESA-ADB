# Short coordinated augmentation: matched development command

Implemented27September2026.77thesis tests pass and CLI dryrun passes. No full
GPU training launched by the assistant. This follows the recurring pre2006
short/wide annotation inventory; it is a hypothesis, not an established improvement.

## Protocol

Inputs40–47. Separate expanding training folds2003/2004/2005; each trains before
January1, calibratesJanuary–March, assessesApril–December. These periods have
been inspected: development data, not untouched testing. No2007–2013 access.
Botharms: identical modelinitialization, scaler, uniformly sampled25000globally
nominal windows, fiveepochs, batch64, seed42. Sameepoch-addressed shuffle and
dropout streams; corruption RNG isolated/restored with deterministicbatchseeds.
Epoch-level resume restores model/optimizer/scheduler and replays incompleteepoch.
CPUinterruption regression test matches uninterrupted weights/losses exactly;
GPUexact determinism is not guaranteed. This control is not identical to older
runs with a different randomstream, window budget or40epochs.

Arms: legacy generator versus50/50legacy+short-coordinated mixture. Same nominal
MSE + pseudo separation loss and optimizer. Original quantile/postprocessing
search and totalalarm1%cap retained to isolate the augmentation change. Eacharm
selects everyepoch on correctedcalibrationF0.5, earliesttie. BOTH rules frozen
before thatfoldassessment. No rules are changed using assessment outcomes.

Initial/pretraining/prefinal datahash guards include goldenpreparedtrainCSV and
annotations; source/settings/protocol checked onresume. Results appear only after
fold finalintegritycheck. This does not certify underlyinghardware health.

## Launch inside tmux

```bash
conda activate timeeval
cd ~/projects/ESA-ADB
python -m esa_thesis augmentation-comparison \
  --folds 2003 2004 2005 \
  --output results_longrun/development/short_coordinated_seed42_v1 \
  --device cuda --batch-size 64 --epochs 5 \
  --max-train-windows 25000 --seed 42
```

Repeat exactcommand with `--resume` after interruption; do not edit experiment
source meanwhile. Completedfolds skipped. Keep outputs in a newdirectory.
Rootresults.csv has12rows:3folds x2arms x2scopes. Perarm checkpoints, histories,
calibrationcandidate grids, frozenrules and assessmentscores/predictions retained.
Do not broaden training or choose per-yearwinners based on earlyfoldresults;
complete the fixed comparison and report everyfold.

Newtests cover noassessmentbeforebothfrozen, samplerlabel/budgetchecks, randomstate
restoration evenonexception, actual tinytraining and interruptedresume forbotharms.
Original73tests also pass. No claim F0.5improvement until the GPUrun completes.

For the subsequent controlled short-roll ablation, see [the separate protocol](short-no-roll-comparison.md). The default original comparison is unchanged.
