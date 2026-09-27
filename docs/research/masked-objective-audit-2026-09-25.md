# Training objective audit — channels 40–47

Completed 25 September 2026. No training or calibration/assessment inference was
launched. The probe used 128 seeded nominal windows selected from the saved 2004
pilot training indices (training before 2004), identical synthetic inputs for both
arms, saved epoch-1 and epoch-5 weights, and all four mask phases. The training
CSV passed its golden SHA-256 check before loading and after inference.

## Observations

| Model | Epoch | Mean normal reconstruction error | Mean synthetic reconstruction error | Mean synthetic penalty |
|---|---:|---:|---:|---:|
| Baseline | 1 | 0.306911 | 6.591958 | 4.547519 |
| Baseline | 5 | 0.184762 | 8.859540 | 3.150109 |
| Masked | 1 | 0.430128 | 4.455412 | 6.171932 |
| Masked | 5 | 0.320988 | 9.069595 | 3.624144 |

The objective rewards accurate normal reconstruction and large synthetic errors.
Both components improve in this fixed sample, while saved real-event calibration
F0.5 falls from 1.0 to 0 for baseline and 0.171616 to 0 for masked. This supports
investigating mismatch between the synthetic objective and real detection. It
does not identify a unique cause or establish that all synthetic augmentation is
harmful. The earlier reconstruction-only experiment also had mixed results.

These are diagnostic evaluation-mode measurements, not recovered historical
training losses: dropout is disabled, sample size is 128, and the penalty uses
batches of 16 rather than the training batch size of 64. Baseline phase repeats
are duplicates; masked phase measurements are dependent observations of the
same windows. Neither arm has 512 independent sampled windows. These error
summaries are not deployed event scores or F0.5 estimates.

## Mask and augmentation interaction

Exact enumeration of the configured uniform segment lengths (12–84 samples),
uniform valid starts, and equally weighted four mask phases gives a **15.1935%**
probability that the synthetic segment touches no hidden target. In the fixed
training probe, **17.3828%** of window/phase pairs have unchanged hidden target
values. The latter includes transformations that leave values unchanged, such
as rolling a flat segment; it is not an estimate of segment geometry alone.

Visible corruption can still change predictions of hidden targets. Therefore,
no hidden-target change does not mean zero loss or gradient. However, those
examples ask the model to reconstruct unchanged hidden targets badly when
visible context is corrupted. This differs from detecting reconstruction error
at the corrupted target itself and merits a controlled ablation.

## Decision

Keep the original full-period baseline and all existing results. Do not tune to
assessment or run another long sweep. A justified next small experiment would
change only masked augmentation placement so it affects hidden targets, with
the original masked configuration as the control. This remains a hypothesis;
it must be specified before running, and success must include real-event
calibration/development metrics, not merely lower synthetic loss. No such
experiment was launched by this audit.

## Reproduction and evidence

- Script: `scripts/research/masked_objective_audit.py`; run from repository root
  with the research Python environment and `PYTHONPATH=.`. CPU inference only;
  no optimizer steps. The script is fixed to the existing 2004 pilot artifacts.
- Measurements: `tables/masked-objective-audit-2026-09-24/training_probe.json`.
  The directory date reflects when the audit started.
- Exact geometry: `tables/masked-objective-audit-2026-09-24/mask_overlap_geometry.json`.
- Saved calibration history: [previous diagnosis](masked-calibration-diagnosis-2026-09-24.md).
