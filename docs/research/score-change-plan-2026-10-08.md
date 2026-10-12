# Fixed-checkpoint score-change pilot

Protocol fixed before execution. Use only the six joint-AE calibration-selected
checkpoints from inner_selection_cuda_v2, with their cached scores. No retraining,
checkpoint reselection, channel removal, ensemble, or outer-period evaluation.

Transform x_t = log(1 + nonnegative saved reconstruction score_t).
Let alpha = 1 - 2^(-1/2880), one-day half-life at 30 seconds/sample.
Candidate score d_t = abs(x_t - m_(t-1)); then
m_t = (1-alpha)*m_(t-1) + alpha*x_t.
Initialize m_-1 from the first calibration x; the first candidate score is zero.
Keep all samples. Carry the final calibration state into validation; never reset
at that boundary. Update on every sample, including alarms. No labels affect
state updates. One day is a fixed engineering choice, not a tuned optimum.
No scale division: log compression limits the effect of extreme raw score values.

Use the existing quantile grid, merge gaps 0/16/64, minimum durations 1/8/32,
corrected system event F0.5 selection, and 1% calibration alarm cap. Freeze each
rule before loading that run's validation scores. Compare original frozen alarms
and transformed alarms on exactly the same real Jul–Dec 2001 and 2002 periods,
three seeds each; report local and system scopes, false events and duration.

Advance only if candidate mean system F0.5 improves in both folds without higher
mean FPe in either. Ties retain original. No automatic promotion/outer run.

This filter is history-only in score space, but its upstream AE scores are
retrospective and overlapping windows can use future telemetry. Cached calibration
and validation scoring are separate segments; carrying filter state does not
remove their upstream boundary effects. Do not claim an operationally causal
pipeline. Adaptation can absorb sustained anomalies and low-score movements may
be harmless. These periods already informed development; no untouched-test claim.
