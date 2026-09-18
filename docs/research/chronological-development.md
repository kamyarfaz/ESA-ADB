# Chronological Transformer AE development baseline

This runner retrains the existing pseudo-anomaly Transformer AE from scratch on
three expanding historical training sets. It retains the configured eight input
channels 40–47 and window-mean scoring. It never opens the benchmark test CSV.
The final October–December 2006 calibration interval is also unused.

## Fixed folds

All intervals include their start and exclude their end. Training starts on
2000-01-01. Event counts include Anomaly and Rare Event IDs intersecting each period.

| Fold | Train before | Calibration | Assessment | Calibration / assessment IDs |
| --- | --- | --- | --- | --- |
| 2003 | 2003-01-01 | Jan–Mar 2003 | Apr–Dec 2003 | 3 / 6 |
| 2004 | 2004-01-01 | Jan–Mar 2004 | Apr–Dec 2004 | 1 / 10 |
| 2005 | 2005-01-01 | Jan–Mar 2005 | Apr–Dec 2005 | 4 / 7 |

Nine-month assessment periods were chosen for broader event coverage, before any
model results. Calibration remains sparse, particularly in 2004. Report every fold
and precision/recall/counts, not only a mean F0.5. Event IDs crossing boundaries can
appear in both adjacent intervals; these are temporal assessments, not independent
random samples. Later folds may train on earlier assessment observations, which is
intentional expanding-window development.

## Run the first fold

From the repository root in the thesis environment:

```bash
conda activate timeeval
python -m esa_thesis develop \
  --folds 2003 \
  --output results_longrun/development/ae_baseline_2003_seed42 \
  --device cuda --batch-size 64 --epochs 40 --seed 42
```

Append `--dry-run` to inspect the plan without creating output or starting training.
Run the remaining fixed folds after checking that the first completes correctly:

```bash
python -m esa_thesis develop \
  --folds 2004 2005 \
  --output results_longrun/development/ae_baseline_remaining_seed42 \
  --device cuda --batch-size 64 --epochs 40 --seed 42
```

A one-epoch run is a pipeline smoke check, not a comparable research result.
Existing output directories require `--resume`. New runs save full epoch checkpoints.
Older partial runs without them require `--resume --restart-incomplete`, which
archives the partial fold before restarting it; completed folds are skipped.
See the [recovery and tmux instructions](expanded-channel-ae.md#recover-the-interrupted-september-run-and-survive-logout).
It uses float32 training, without AMP, and a fixed seed. Exact GPU bitwise
reproducibility is not guaranteed. Each fold caps nominal windows using the existing
250,000-window policy; the configuration and source hashes are recorded.

## Selection and artifacts

The scaler is fitted only on nominal training observations. All model windows are
contained in their own period; invalid/non-finite data raise errors instead of being
filled across split boundaries. Lack of nominal windows aborts training.

Every five epochs and at the final epoch, corrected ESA event-wise F0.5 selects the
checkpoint and calibration decision rule under the 1% predicted-rate cap. Equal
checkpoint scores retain the earlier epoch. The threshold grid and postprocessing
settings are fixed in the protocol. The selected checkpoint and rule are written
before assessment observations are loaded. Assessment never changes that decision.

Outputs include `protocol.json`, per-fold `history.csv`, `model_checkpoint.pt`,
`calibration_candidates.csv`, `frozen_rule.json`, assessment scores/masks and
`results.json`, plus combined `results.csv`. Keep these local experiment outputs
out of Git. The checkpoint is compatible with the existing scoring-comparison tool,
but do not run historical fold checkpoints on the benchmark test to guide selection.

This is a new development baseline, not an exact reproduction of old training:
splits, checkpoint metric, calibration grid, batch size, and numerical precision
are explicitly fixed here. Future architecture comparisons must use this same
protocol. The AE remains retrospective and sees its reconstruction target. A causal
forecaster must also document its different information and delay budget.

Tests cover period boundaries, invalid data rejection, no benchmark test file,
and freezing the rule before loading assessment data, including a synthetic
one-epoch pseudo-anomaly training run. Full ESA GPU training is user-run.

## Channel coverage ablation

The default remains `--channel-set baseline`. The separately documented
[expanded-channel experiment](expanded-channel-ae.md) adds channels 14, 21, and 29
using `--channel-set expanded`, with the same training and evaluation settings.
