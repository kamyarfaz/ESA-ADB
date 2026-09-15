# Active research results

- `channel_correlation/`: channel correlation analysis and representative selections.
- `mission1_reconstruction_ae_sweep_optimized/`: training sweeps, checkpoints,
  cached scores, plots, summaries, and post-hoc evaluation artifacts.
- `latest_run.txt` inside a sweep root identifies the run used for resumption.

Keep completed runs intact. For changes to model or preprocessing configuration,
use a new `python -m esa_thesis train --output-root results_longrun/NAME` directory.
Cached outputs are not automatically invalidated when settings change.
Existing score names describe historical calculations, not independently verified
ESA/Kaggle scores. See `../docs/thesis-research-context.md`.
