# Frozen 2004 baseline diagnostic

Prepared 25 September 2026. No new training. Inputs40–47; historical checkpoint
epoch35. Training references before2004 (up to2,000 nominal windows per year,
8,000 total); full January–March2004 calibration. No development assessment or
benchmark evaluation. Reuse the saved scaler and threshold3.132171630859375.

```bash
conda activate timeeval
python -m esa_thesis diagnose-channels \
  --source-run results_longrun/development/ae_baseline_remaining_seed42 \
  --fold 2004 \
  --output results_longrun/diagnostics/baseline_2004_epoch35_v2 \
  --device cuda --batch-size 64
```

Run inside the existing tmux session. Add `--resume` to the same command after
interruption, provided source/protocol settings have not changed. Older diagnostic
caches cannot resume under this updated protocol; keep them intact.

Expected reproduction: calibration F0.5=0.38390663095893557, TP1 FP2 FN0.
The script checks the golden training CSV checksum before and after the diagnostic
and fails if frozen calibration F0.5/precision/recall/counts differ beyond1e-6.
The older development protocol has no source hashes to compare with current
inference dependencies. Current sources, checkpoint, rule, annotations and source
protocol are fingerprinted; metric reproduction is required, not proof of bytewise
historical inference equivalence.

Outputs include training error distributions, cached channel scores, event table,
`calibration_alarms.csv`, and `alarm_00.png` etc. Each alarm plot has eight channel
score traces and eight training-scaled prepared input traces, with four hours of
context on either side. These are prepared telemetry values, not original native
sampling or verified physical units. Peak-error channels are diagnostic associations,
not causal explanations or feature importance. No new thresholds are selected.

Final `results.json` is written only after reproduction and integrity gates.
GPU inference has not been launched by the assistant. The command dry-run and
59-test thesis suite pass; historical sampling excludes2004 and later years.

## Loader repair after interrupted v1

The v1 attempt stopped before inference inside NumPy label membership validation
with an out-of-bounds index549755813888. The current CSV matched the golden hash
before and after successfully reloading all4,207,680 training rows. Cause remains
uncertain. Validation now checks each label column within each chunk against
0/1/2/3 before narrowing storage; no bad values are ignored or repaired. This
avoids the large lookup-table operation and lowers label storage memory.
60tests pass, including invalid large integers, negatives, fractions, NaNs and
strings. Use the new v2 output above: v1 source hashes no longer match. Preserve
v1; it contains only the protocol. No diagnostic GPU inference was run here.
