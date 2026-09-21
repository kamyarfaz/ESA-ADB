# Causal forecasting development comparison

Implemented 2026-09-17. This compares three predeclared forecasting detectors on
the existing chronological folds and channels 40–47. It does not open the benchmark
test CSV or the reserved final 2006 validation interval. No real-data performance
improvement is claimed before running this experiment.

## Models and shared protocol

| Model | Forecast |
| --- | --- |
| Persistence | Repeat the last observed value for all 16 future samples; no learned weights |
| MLP | Shared channel-independent network with two 128-unit hidden layers |
| Transformer | Shared channel-independent encoder, 16-sample patches, width 128, 8 heads, 2 layers, direct 16-step forecast head |

All receive 256 historical samples (128 minutes) and forecast the next 16 samples
(8 minutes). A new forecast is issued every 16 samples. The context ends before
the first target; forecasts never receive target values. Transformer attention
can span every historical patch because all patches are already observed.
There is no cross-channel attention in this initial forecasting candidate.

The training-only robust scaler and clipping at ±10 standardized units match the
existing data policy. Both context and targets must be nominal across supplied
label channels. All models use the same training period and nominal-window policy;
up to 250,000 windows are selected using the declared seed. Persistence uses the
same scaler but has no optimizer or learned checkpoint selection. The learned
models use MSE, AdamW, cosine decay, float32, and no pseudo-anomaly objective.

Each timestamp receives the maximum squared forecast error across channels when
that observation arrives. No errors from later timestamps are averaged into it.
Disjoint forecast blocks cover every sample after the initial context, including
a partial final block. Scores are recorded at target timestamps, not context ends.
The first 256 samples of each calibration/assessment period produce no alarms;
events there remain in full-period evaluation. Quantiles use only covered scores.
This intentionally avoids importing context across split boundaries.

## Alarm policy and comparison limits

**Forecast alarms are threshold-only.** The AE's retrospective gap filling and
minimum-duration filters can alter earlier labels using future observations, so
they are not applied here. A future streaming duration rule would need separately
specified detection delay and no backdating. Thresholds use the same fixed quantile
list and corrected ESA event F0.5 with a 1% calibration predicted-rate cap.

Every five epochs and at the final epoch, calibration selects the checkpoint and
threshold. Equal checkpoint scores retain the earlier epoch. The frozen rule is
written before assessment observations are loaded. Assessment does not change the
rule. No model is automatically selected by its assessment score.

The three forecasters have matched target coverage and alarm policies. Comparison
with the existing AE is a comparison of detector designs, not architecture alone:
information availability, objective, scoring, warm-up, and postprocessing differ.
Keep those differences explicit in the thesis. Sparse calibration, especially the
single 2004 event, remains a limitation; all three folds must be reported.
Causal input/scoring is not by itself a deployment latency or hardware certification.

## First run

From the updated repository in the thesis environment:

```bash
conda activate timeeval
python -m esa_thesis forecast-develop \
  --folds 2003 \
  --models persistence mlp transformer \
  --output results_longrun/development/forecast_2003_seed42 \
  --device cuda --batch-size 64 --epochs 40 --seed 42
```

Models run sequentially. Persistence performs calibration and assessment without
training; each learned model trains for 40 epochs. Add `--dry-run` to inspect the
plan without reading datasets, writing output, or checking CUDA availability.
`--device cpu` is available. Use a fresh output directory: resume is not implemented.
If interrupted, preserve completed results and rerun only missing models in a new
directory with `--models`. A one-epoch smoke run is not a comparable experiment.

After the first fold completes correctly, keep settings fixed and run:

```bash
python -m esa_thesis forecast-develop \
  --folds 2004 2005 \
  --models persistence mlp transformer \
  --output results_longrun/development/forecast_remaining_seed42 \
  --device cuda --batch-size 64 --epochs 40 --seed 42
```

## Artifacts

`protocol.json` records settings, software versions, source hashes, annotation
hashes, and input file size/mtime. Size/mtime is provenance, not a dataset checksum.
Outputs are grouped by fold/model with history, checkpoint/scaler, calibration
candidates and scores, frozen rule, assessment scores/coverage/predictions, and
results. Combined `results.csv` is updated after every completed model.
Forecast checkpoints are not compatible with the AE `compare-scoring` command.
Elapsed seconds are wall-clock run timing, not a controlled inference benchmark.
GPU bitwise determinism is not guaranteed despite fixed seeds.

## Verification

The full thesis suite passes 24 tests, including:

- Exact persistence errors on a ramp, target timestamp alignment, and partial-tail coverage.
- Prefix invariance for all three models: changing later observations cannot alter
  earlier scores; scoring a truncated series preserves its earlier scores.
- Nominal filtering includes future targets; no anomalous-window fallback.
- Synthetic end-to-end runs for all models, with the rule present before assessment
  loading and a nonexistent benchmark test path.

```bash
python -m unittest discover -s tests/thesis -p 'test_*.py' -q
```

## Optional level-residual variant

The explicit `--models transformer_residual` option adds a matched level-centered
Transformer. The original default model list is unchanged. See the
[experiment protocol and tmux command](residual-transformer.md).
