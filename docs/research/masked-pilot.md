# Matched masked-reconstruction pilot — channels 40–47

## Current follow-up: 2004 and 2005

The 2003 run is complete and preserved. Version2 adds fold selection and automatic
dual-scope assessment reporting; the model, masking and training algorithm are unchanged.

| Fold | Training period | Calibration | Development assessment |
| --- | --- | --- | --- |
| 2004 | 2000–2003 | Jan–Mar2004 | Apr–Dec2004 |
| 2005 | 2000–2004 | Jan–Mar2005 | Apr–Dec2005 |

Each fold has fresh baseline and masked models, sharing25,000nominal windows,
fiveepochs and seed42. Scaling uses only that fold's training data. Both models
select checkpoints/thresholds with the system-level calibration metric, then
report local and system assessment scores from identical frozen predictions.
The `calibration_f05` column remains system-level regardless of assessment scope.

Results are under `2004/` and `2005/`; root `results.csv` aggregates system scores
and `dual_scope_results.csv` includes both scopes. Other-channel annotations remain
neutral in the local metric. No new input channels or threshold optimizations.
Use the NEW output below, not the completed v1 directory. Source/version checks
intentionally reject resuming v1 with v2. New v2 runs support epoch recovery and
skipping completed folds when `--resume` is added to the unchanged command.

Validation after extension:58thesis tests passed; two-fold dry-run passed.
No full GPU follow-up was launched by the assistant.

## Original design (first evaluated on 2003)

## Purpose

Test whether hiding inputs reduces the current Transformer's ability to reconstruct
anomalous patterns too easily. This is a hypothesis, not a diagnosed failure or
promised improvement. The unsuccessful q99 score normalization is not included.

## Frozen pilot design

| Component | Both arms |
| --- | --- |
| Inputs | Channels 40–47, original order |
| Training | 2000–2002, globally nominal windows |
| Window budget | Same seeded sample of up to 25,000 stride-16 windows |
| Scaler | Median/IQR on nominal training samples, existing clipping |
| Epochs | Five per arm, same optimizer and cosine schedule |
| Calibration | January–March 2003, every epoch; earliest F0.5 tie |
| Development assessment | April–December 2003, after both rules are frozen |
| Metric | Existing corrected event-ID/duration F0.5 |
| Threshold grid | Existing percentiles, merge gaps0/16/64, minimum durations1/8/32 |
| Seed | 42 |

**Baseline arm:** existing Transformer and pseudo-anomaly separation loss.
**Masked arm:** identical base weights at initialization plus a zero-initialized
128-parameter mask token. Hide 25% of temporal patches on all eight channels
simultaneously. Positions0,4,8,12 are one phase; the other three phases cover the
other patches. Train phases cycle deterministically by epoch and batch.

Both encoder and decoder receive masked content. Positional and channel identity
remain available. Reconstruction and the same pseudo-anomaly separation objective
are computed on hidden patches only. At inference, four complementary passes
reconstruct each target patch exactly once while hidden. Both arms then use
window-mean errors, overlap averaging and maximum across channels.

This compares two complete reconstruction approaches: masking necessarily changes
which training targets are scored and requires four inference passes. Equal window
and epoch budgets are **not equal compute budgets**. No architecture/seed sweep,
new channels or OR ensemble is part of the pilot.

## Run

From the project root in tmux:

```bash
conda activate timeeval
python -m esa_thesis masked-pilot \
  --folds 2004 2005 \
  --output results_longrun/development/masked_pilot_2004_2005_seed42_v2 \
  --device cuda --batch-size 64
```

Progress prints every100training/scoring batches and at epoch boundaries. The
training CSV checksum is checked before loading, after training/calibration loading,
and before accepting final results. Initial reading/checksumming can take minutes.
No assessment CSV is accessed. The prepared training CSV contains all pilot periods.

After interruption, add `--resume` to the identical command. Full epoch state,
optimizer, scheduler, best weights and RNG are restored. An incomplete epoch restarts
from its preceding checkpoint; completed arms skip training. Assessment can rerun
from the frozen decisions. Source/config/annotations must match the original protocol.
Use a new output for deliberate configuration changes.

## Outputs and interpretation

Each arm saves history, last/best checkpoints, calibration candidates and its frozen
rule. Both frozen files exist before development assessment is loaded. Root
`results.csv` is written only after the final data-integrity check. Shared window
indices and complete configuration/source/annotation hashes are saved.

Compare the two arms to each other, **not directly with the full-run0.8057**.
This is a small single-seed development pilot and may undertrain either model.
2003 has few events and was explored previously; it is not independent confirmation.
A positive result requires additional chronological checks before a full run.
A poor result does not prove that all masked architectures fail. No automatic
promotion or expansion to more training runs is implemented.

## Validation

Full thesis suite:54tests passed. A subsequent targeted masked-pilot suite (five
tests, including one newly added) passed. Tests cover:

- identical base initialization;
- no prediction dependence or gradient through hidden input content;
- four-phase complete nonoverlapping target coverage;
- hidden-only training loss and hidden-only inference error;
- seeded nominal-window selection;
- real tiny CPU training and completed-checkpoint recovery.

Dry-run passed. Full server/GPU pilot has not been run by the assistant.
