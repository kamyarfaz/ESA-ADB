# Fixed q99 channel normalization — 24 September 2026

Scope: channels 40–47 only; saved epoch-10 Transformer; nominal sampled training
window errors from 2000–2005; calibration 2006. No assessment data or retraining.

## Prespecified change

For each channel c, scale[c] = max(percentile(training window errors[c],99),1e-8).
The alternative score is max_c(aggregated window-mean error[c] / scale[c]).
No centering; no percentile/floor search. Existing temporal aggregation is fixed.
Both versions use the existing percentile/merge/min-duration grid, corrected
ESA evaluator, and <=1% predicted fraction selection. Original wins method ties.
Protocol was written before running calibration sweeps. Scales are channel weights,
not probabilities. Sampled training-window errors and overlap-averaged calibration
errors have different distributions; normalization is an ablation, not a formal
probability calibration.

| Calibration metric | Original max | Training-q99 normalized max |
| --- | ---: | ---: |
| F0.5 | 0.882148 | 0.882099 |
| Detected IDs | 6 | 6 |
| False-positive events | 0 | 0 |
| Missed IDs | 4 | 4 |
| False-positive duration (seconds) | 10030.695 | 12441.642 |

The normalized alternative adds 2410.947 seconds (about40.18minutes) of nominal
alarm duration. The same six IDs are detected; there is no recovery of a new ID.
Difference in F0.5 is tiny, but it provides no reason to promote the alternative.
Keep the original model, score and threshold. Do not tune more percentiles to
chase a small improvement on this already-used ten-event calibration set. No
assessment inference or earlier-fold GPU comparison is justified for this candidate.
Assessment reference remains0.805722; it was neither rerun nor selected against.

## Validation and reproduction

50 thesis tests pass. Added tests cover scale floor, invalid inputs, invariance to
per-channel changes of error units and training-only scale fitting. Production
comparison reproduced the original calibration prediction mask exactly and checked
cached metrics. Cache hashes are recorded and checked again after comparison;
annotation and evaluation-source hashes match the completed diagnostic.

Run from the repository root:

```bash
python -m esa_thesis.evaluation.compare_channel_normalization \
  --diagnostic results_longrun/diagnostics/ch40_47_epoch10_v2 \
  --output NEW_OUTPUT_DIRECTORY
```

Original outputs: results_longrun/diagnostics/ch40_47_q99_comparison_v1.
Compact evidence: tables/channel-normalization-2026-09-24/.

Next research should address an evidenced modelling limitation on these same eight
channels, rather than retuning this normalization. Masked reconstruction and the
realism of pseudo-anomaly corruption remain hypotheses, not approved winners.
Any architecture pilot needs a matched chronological protocol and the existing
baseline reference; the long-run checkpoint must not assess years it trained on.
