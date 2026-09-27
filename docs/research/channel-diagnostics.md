# Fixed-checkpoint diagnostics: channels 40–47

Use after the completed long baseline. No training, normalization selection or
assessment-file access. It reproduces the frozen 2006 score before accepting results.

```bash
conda activate timeeval
python -m esa_thesis diagnose-channels \
  --source-run results_longrun/long_baseline/ch40_47_train2000_2005_seed42 \
  --output results_longrun/diagnostics/ch40_47_epoch10_v2 \
  --device cuda --batch-size 64
```

Run in an existing tmux session, or create one if outside tmux. Add `--resume` to
the same command after interruption; completed training/calibration inference stages
are reused, but an interrupted stage restarts. No per-batch recovery. Existing output
without `--resume` is rejected. Configuration, source and checkpoint must match.

- Uses the saved scaler, weights and frozen rule; does not fit on calibration.
- Samples up to 2,000 nominal windows in each training year 2000–2005. Evenly
  spaced among eligible stride-16 windows, entirely inside each year. This is a
  deterministic temporal sample, not all windows or independent validation.
- Scores all 2006 calibration windows, preserving the baseline's temporal averaging.
- Saves per-window/per-channel errors, calibration channel scores, timestamps,
  event detections/misses, dominant-channel counts, quantiles and event plots.
- Checks original model/config/data/evaluator/threshold source hashes, verifies only
  the training CSV golden checksum, records checkpoint/annotation hashes, and
  rejects changed-data runs. Does not read the assessment CSV or assessment arrays.
- Requires frozen calibration F0.5, precision, recall and TP/FP/FN to match within
  1e-6. Stops rather than silently accepting a different baseline result.

Important interpretation: training error quantiles are **per-window** whereas
calibration quantiles are **overlap-averaged per timestamp**. They are diagnostic
summaries, not directly interchangeable estimates of drift. Dominant channel is
argmax (first channel wins ties); nominal alarm counts use the prepared global
label mask, whereas official false-positive duration uses raw annotation unions.
Event detection uses interval overlap even when a short raw annotation contains
no sample timestamp. A missing inside-sample peak is not a missed event by itself.

Outputs: `results.json`, `channel_error_summary.csv`, `calibration_events.csv`,
`calibration_events.png`, and three NPZ caches. Result JSON is written last.
Preserve caches outside source Git. Review the outputs before specifying the single
training-normalized-max scoring comparison. No improved F0.5 is claimed by this run.

Validation: 47 thesis tests passed, including exact aggregation parity with existing
scoring, nominal/year-boundary sampling and off-grid event attribution. Actual source
checkpoint dry-run passes. Full GPU inference must run on the server.

After the 24 September CSV integrity incident, use v2, not resume of v1.
The diagnostic now repeats the full training CSV hash before accepting results.
The old training-error cache is preserved but is not trusted for the retry.
