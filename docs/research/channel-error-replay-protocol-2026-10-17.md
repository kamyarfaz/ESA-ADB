# Channel-resolved frozen-model replay

Executes the diagnostic specified in `development-evidence-review-2026-10-16.md`.
No fitting, new threshold, input selection or outer-test evaluation is performed.

## Numerical replay investigation

The initial CUDA replay of 2001/seed42 calibration matched scores within the
declared rtol=1e-5/atol=1e-6 but failed the exact-mask gate at32/521280 timestamps.
Those timestamps were06April2001 08:00:00–08:15:30UTC. The original float32 score
equalled the frozen threshold184.55422973632812; a one-ULP increase crossed it.
The maximum score discrepancy over the period was3.4332275390625e-05.
Corrected calibration F0.5 stayed2/3, but the alarm-interval count increased1→2;
therefore matching F0.5 alone was not accepted as sufficient replay evidence.

The run stopped before event diagnostics. A separate CPU replay with identical
weights, preprocessing, batch size and frozen decision matched the score tolerance
and reproduced the original mask exactly. This is consistent with execution-path
floating-point differences, not proof of a unique hardware/kernel cause.

The failed v1 directory, runner snapshot and log are preserved. The v2 execution
retains both original gates: score agreement within the same tolerance and exact
frozen-mask agreement. It tries CUDA first; when a mask differs, it repeats the
same stage on CPU. If CPU fails either gate, it stops. It never adjusts thresholds
or chooses a device for a favorable metric. Stage metadata records the device.
There is no decision-equivalence claim for a failed GPU reconstruction.

After15 verified stages, v2 stopped on2002/seed44 calibration because128 GPU
scores exceeded the same tolerance (maximum absolute discrepancy4.7683716e-05,
maximum relative discrepancy1.2730769e-05). V3 applies the same CPU fallback to
either score-tolerance or mask failures. Both acceptance gates remain unchanged.
The15 completed v2 stages are reused with file-hash verification; failed outputs
and the v2 runner snapshot are preserved. This corrects the fallback coverage,
not the model, threshold or scientific acceptance criteria.

## Measurements

Six primary joint checkpoints: folds2001/2002, seeds42/43/44. Each is replayed on
its original calibration and development validation segments, and on its training
period for nominal monthly error distributions. Scalar scores must reproduce from
the channel maximum before diagnostic tables are written. Calibration masks are
replayed from original saved scores/rules; validation masks additionally match
the saved binary arrays. Windows, scaling, clipping and aggregation are unchanged.

For each input channel's own annotated interval: error peak, minimum and median,
nominal upper-tail fraction at the peak, lower-tail fraction at the minimum,
median difference from nominal, and attribution of the scalar maximum. Nominal
samples exclude all annotation categories/intervals, including communication gaps.
This is a retrospective nominal reference, not a fitted or deployable detector.
Training summaries do not produce event-performance measurements.

No minimum or channel rank becomes a new anomaly rule. Interpret all channels,
events and seeds together; do not select favorable cases. Ties in maximum-channel
attribution use NumPy's first-index convention and do not establish unique cause.
Error distributions are for overlapping windows; rows are not independent samples.

Outputs: `results_longrun/diagnostics/channel_error_replay_20261017_v3/`.
Investigation: `results_longrun/diagnostics/channel_error_precision_20261017_v1/`.
Source: `scripts/research/channel_error_replay.py` and
`scripts/research/investigate_channel_replay_precision.py`.

## Supplementary continuation after failed calibration gate

On 17 October the user instructed continuation. A separately labeled diagnostic
completed only 2002/seed44 validation and training, with original two CPU threads
and the same CUDA inference settings. Validation independently requires the same
scalar tolerance and exact saved mask. This permits a complete six-model
validation description, not a claim that the original calibration gate passed.
No acceptance criterion was relaxed. `supplementary_status.json` records six
validation stages, five calibration stages, six training summaries and the
unresolved calibration failure. See `channel-error-validation-results-2026-10-17.md`.
