# Representation and input decision after calibration consolidation

The evidence does not justify another full-period training run yet. Keep the
Transformer autoencoder reference, eight inputs and fixed data files. This is a
scientific stopping decision for the completed comparison, not a permission hold.

| Question | Existing evidence | Decision |
|---|---|---|
| Will a lower global threshold fix misses? | New fixed-quantile comparison detects more but produces 87.2% occupancy in 2002. Earlier event peaks rank below many nominal scores. | Reject this candidate; no neighboring-quantile search. |
| Should inputs immediately expand to all 76? | Adding 14/21/29 recovered selected development events but increased false alarms to 33 and 35 in 2004/05. | All-channel evaluation is mandatory; all-channel model input is not yet justified. |
| Is correlation enough to select inputs? | Training-only correlations vary by year; stable groups can still drift and cause false alarms. | Use correlation as descriptive evidence, not a performance guarantee or automatic channel selector. |
| Can channel isolation alone fix interference? | Three-seed matched isolation failed its advancement gate. | Do not repeat that experiment. |
| Did clipping hide the missed local events? | Verified event-specific audit finds no clipping on their annotated input channels. | Do not remove clipping as an unsupported general fix. |
| Is preparation perfectly operational? | Training-only retention audit finds label-dependent restoration changes five values at two timestamps; other historical preparation includes backfilling/counter issues. | Keep benchmark preparation immutable; any operational preprocessing needs a new version and separate claim. |
| Does recent-score adaptation already solve drift? | Completed score-change filter only succeeds in one seed; direct telemetry reference is silent. | Do not relabel these completed negative experiments as new candidates. |

The next scientifically useful design problem is distinguishing nominal operating
range changes from abnormal changes while retaining absolute-level anomaly evidence.
Neither unconditional centering nor unrestricted adaptation is safe to assume:
both may erase sustained anomalies. Existing residual-forecasting, normalization,
scoring and augmentation comparisons must be incorporated before a new design.
The present evidence does not identify a validated architecture for that problem.

Before any subsequent training experiment, declare one representation change,
a matched control, training-only fitting, chronological calibration, fixed seeds,
resource budget and failure criteria. Require improved system event F0.5 together
with false-event and false-duration control across both development years. Preserve
absolute-level information when testing relative features, and evaluate both
upward and downward event signatures. No claim of independent generalization can
use development periods or the already inspected benchmark as unseen data.

Sources within the project: `expanded-channel-results.md`,
`preprocessing-and-correlation-policy.md`, `channel-regime-findings.md`,
`channel-isolation-results-2026-10-08.md`, `inner-event-audit-2026-10-08.md`,
`score-change-results-2026-10-08.md`, `telemetry-reference-results-2026-10-08.md`,
and `nominal-quantile-results-2026-10-14.md`. Older documents' suggested next steps
are historical hypotheses; the latest completed outcomes take precedence.
