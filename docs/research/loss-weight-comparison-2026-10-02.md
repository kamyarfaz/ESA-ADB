# Matched artificial-anomaly loss-weight comparison

Approved 2 October 2026. Completed and verified 3 October 2026.
The advancement gate failed; see [results](loss-weight-results-2026-10-03.md).

Hypothesis: reducing the pseudo-anomaly loss weight improves reconstruction of
normal patterns without losing useful anomaly separation. Diagnostic plots make
this worth testing, but do not establish causality. This is not a guaranteed
F0.5 improvement or a repeat of the earlier penalty-zero reconstruction-only run.

## One intended change

Control loss: `normal_MSE + 1.0 * pseudo_penalty`.
Candidate loss: `normal_MSE + 0.1 * pseudo_penalty`.
The hinge margin, corruption generator, model, scaler and all other settings
remain unchanged. The candidate uses the original inner-validation training
function without modifying it. A process-local context sets the multiplier,
records it in the protocol, then restores the default on exit or failure.

Control: `results_longrun/development/inner_selection_cuda_v2`.
Candidate: `results_longrun/development/inner_lambda01_cuda_v1`.
All control source-file hashes and completion manifests must verify before use.
The final comparison also requires identical training starts, scalers, data
boundaries, validation timestamps, runtime and settings except the multiplier.

Six candidate jobs: seeds42/43/44, folds2001/2002, five epochs,25,000 nominal
windows, batch64, CUDA/float32, channels40–47. Fold2001 trains on2000, calibrates
Jan–Jun2001, validates Jul–Dec2001. Fold2002 trains on2000–2001, calibrates
Jan–Jun2002, validates Jul–Dec2002. Both arms use identical real validation data;
synthetic data remains confined to training.

## Selection and decision

Primary comparison uses `calibration_selected` for both weights. Original
threshold grid,1% total-alarm cap and earliest checkpoint tie rule are retained.
The unchanged runner also exports fixed-epoch5 records; these are secondary and
must not be chosen after viewing results. Its `decision.json` compares checkpoint
policies and is NOT the loss-weight decision. Use `loss_weight_decision.json`.

Advance only if lambda0.1 improves mean system F0.5 in both internal folds without
increasing mean false-event count in either. Ties retain the control. Report all
seeds, both annotation scopes, recall and false-alarm duration; do not cherry-pick
the best seed or replace the existing full-period baseline. These internal
periods have already informed development and are not untouched tests. No outer
2003–2005 or2007–2013 assessment is part of this run.

## Command and recovery

```bash
python -m esa_thesis.loss_weight_comparison \
  --control results_longrun/development/inner_selection_cuda_v2 \
  --output results_longrun/development/inner_lambda01_cuda_v1
```

Append `--resume` with identical arguments/source to recover. The sidecar
`inner_lambda01_cuda_v1.comparison.json` locks the comparison before training;
the runner also locks its full source/configuration protocol in the output.
Epoch checkpoints and completed-run manifests preserve progress.

87 thesis tests passed, including loss decomposition/first-batch equality,
restoring the default after exceptions, and existing resume/freeze protections.
Dry-run control verification passed. No GitHub push requested or performed.
