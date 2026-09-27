# Fixed-baseline inference repeatability check

**Passed exactly on CPU.** Two fresh processes loaded the existing2004 epoch35
baseline and the same cached calibration values. No training or parameter changes.
CUDA was unavailable in the assistant environment; GPU inference was not tested.

## Fixed sample and execution

- Inputs: channels40–47, saved scaler and threshold3.132171630859375.
- Four2048-sample calibration blocks, centered where possible around January1
  05:00, January24 08:00, February14 15:00 and March1 12:00UTC. The first block
  is clipped to the start of calibration; starts align to stride32.
- These cover the real-event context, both false-alarm contexts and an additional
  normal-period context. This is a diagnostic selection, not independent validation.
- Window length256, stride32:57windows per block,228total. Batch size16.
- CPU, one PyTorch thread, deterministic algorithms enabled, evaluation/inference
  mode. All checkpoint weights loaded; no gradients, optimizer or checkpoint update.

## Comparison

| Stage | Differing elements | Maximum absolute difference |
|---|---:|---:|
| Prepared raw values |0|0|
| Scaled values |0|0|
| Timestamps |0|0|
| Reconstructed values |0|0|
| Per-window channel errors |0|0|
| Aggregated scores |0|0|
| Frozen predictions |0|0|

Input tensor, model-state and source-file hashes agree between processes. Each
worker also checks input/model immutability and source hashes after inference.
Recomputed scaling agrees exactly with the cached scaled values.

This is bitwise repeatability for this sample and execution mode. It does not
establish accuracy, verify GPU behavior, validate every historical result or
repair the intermittent fault. Block-edge aggregation has less context than
continuous full-calibration scoring; no new event-level F0.5 was computed or
compared with the historical full-period metric.

## State after this check

The existing baseline passed this bounded check. The observation-age sidecar
build remains unaccepted because its separate final integrity gate failed.
Continuing that prototype requires its own successful output verification; this
inference result must not be used to bypass those checks. No new training run.

Reproduction: `PYTHONPATH=. python scripts/research/inference_repeatability.py
--output NEW_DIRECTORY` (one shell line, research environment, repository root).
Full artifacts: `results_longrun/diagnostics/baseline_2004_repeatability_v1`.
Compact comparison: `tables/inference-repeatability-2026-09-25/comparison.json`.
