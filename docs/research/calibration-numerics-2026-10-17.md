# Frozen calibration numerical audit

This audit investigates 2002/seed44 calibration only. It does not change the
checkpoint, threshold, postprocessing, acceptance tolerance or historical scores.
No training or benchmark assessment was performed.

## Checks and interpretation

The original training path calls `augmentation_comparison.score`, which uses the
same `window_errors` and `aggregate` functions as the diagnostic. Their complete
file hashes, model/config/scaler/threshold/runtime hashes, raw dataset checksum and
all recorded run artifact hashes were verified. `development.py` has changed since
the historical run, so its whole-file check initially stopped this audit. The exact
historical version was recovered from Git commit e473c858bd0b27457afcc0b05b7e278a5350ca67
and matched the historical SHA256. Its `load_period` function has an identical AST
to the current loader. This is a verified function-level exception, not a claim
that the whole module is unchanged. Initial failure evidence is preserved.

Two fresh calls to the original CUDA scoring function were bitwise identical to
each other and to the previously saved CUDA replay. All three still fail the
unchanged score tolerance at128/521280 samples, with historical scores between
0.912977457 and0.926272094 (well below threshold72.162323). The largest absolute
difference within those128 samples is0.000011622906. All CUDA masks match exactly.
This establishes repeatability for these executions, not all future executions or
cross-platform identity. The maximum absolute discrepancy over the entire series
is0.000047683716, which occurs under a different score-dependent tolerance.

The CPU replay differs at 32 samples where the historical score equals the
threshold 72.16232299804688. Its score is higher by one float32 step
(0.00000762939453125). The unchanged strict `>` comparison therefore produces a
different mask. Rounding, shifting the threshold or splicing CPU/GPU scores would
change the frozen procedure and was not done.

Historical, prior CUDA and prior CPU predictions all produce the same corrected
calibration event metrics: F0.5=0.7691731680767528, precision=0.9998783232567259,
recall=0.4 and nominal false-alarm time fraction=0.00012167674327412392. This is a
numerical-sensitivity check of an old calibration result, not new performance.
Metric equality does not override failed score or exact-mask checks.

PyTorch documents that mathematically equivalent floating-point operations can
produce different values on CPU/GPU and across platforms. This supports numerical
execution differences as a plausible explanation; it does not identify the unique
kernel or prove that matching historical hardware will resolve the discrepancy.
Sources: [numerical accuracy](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html)
and [reproducibility](https://github.com/pytorch/pytorch/blob/main/docs/source/notes/randomness.md).

## Decision

The original full channel-replay gate remains failed. Preserve original arrays and
all six verified validation stages; do not replace historical scores with a new
execution or describe this audit as a corrected detector. Further blind device,
precision or tolerance searches are not justified by this evidence.

We can continue designing the next controlled development comparison while this
historical cross-platform issue remains documented. Any future comparison must
freeze its runtime and hardware, persist baseline/candidate scores and predictions,
and check repeated inference before evaluation. Historical numerical equivalence
and a new controlled comparison are separate claims. There is no newly approved
method, calibration rule or training sweep in this audit.

The seven corrected event-evaluator parity tests passed. Compact numerical and
metric evidence accompanies this report; the script and arrays are preserved.
