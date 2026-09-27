# 2003 calibration reproduction: boundary isolated

Original paired scoring code rerun on cached calibration inputs with the saved
checkpoint onCPU. Scores match the diagnostic scorer exactly, maxdifference0,
all samples covered. File hashes before/after unchanged. No assessment access.

Frozen threshold6.974297523498535.32 consecutive samples from2003-02-27 02:24:00
to02:39:30 score6.974298000335693, the immediately adjacent float32 above it
(delta4.76837158203125e-7). With strict greater-than comparison, the alarm starts
02:24 instead of02:40, ending03:44 in either case. This explains960additional
nominal alarm seconds and F0.5 .7141835329272609 versus historical.7142410612656628.
Event counts remain1TP/0FP/2FN.

Diagnostic perturbation threshold+1e-6 exactly restores the historical metrics,
but is NOT adopted. The old GPU calibration scores are unavailable, so CPU/GPU
rounding is a plausible explanation, not a proven root cause. Exact agreement
between scoring implementations rules out an observed difference between those
two implementations on these cached inputs. It does not certify hardware health,
raw input integrity beyond the prior initial checksum, or original GPU inference.

Next: run the unchanged diagnostic on the GPU with a new output directory:

```bash
conda activate timeeval
python -m esa_thesis diagnose-channels \
  --source-run results_longrun/development/ae_baseline_2003_seed42 \
  --fold 2003 \
  --output results_longrun/diagnostics/baseline_2003_epoch25_gpu_v1 \
  --device cuda --batch-size 64
```

No training needed. Preserve failedCPU output. Do not relax tolerance, modify
frozen thresholds or accept the missing2003 grid/cap cells until verified.
[Evidence](tables/calibration-boundary-2003-2026-09-26/result.json).
