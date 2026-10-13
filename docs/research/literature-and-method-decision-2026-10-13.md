# Research evidence and method decision

Reviewed 13 October 2026. This is a targeted primary-source review, not a
systematic review or a new performance claim. Claims below are limited to the
sections actually accessible. Local results take precedence over a paper's
headline score when deciding whether a technique addresses this project's failure.

| Source | Relevant evidence | Implication and limit here |
|---|---|---|
| [Kotowski et al., ESA benchmark](https://arxiv.org/pdf/2406.17826), dataset and preprocessing sections | Annotations distinguish IDs, classes, rare events and channels. The benchmark describes chronological evaluation, a nominal-duration correction, forward holding and label-based restoration of short events. | Preserve event-ID scoring and document benchmark preprocessing separately from a deployable label-independent pipeline. Our one-year 2006 calibration differs from the paper's last-three-month split; published scores are not matched controls. |
| [Kim et al., rigorous evaluation](https://arxiv.org/abs/2109.05257), abstract | Point adjustment can substantially inflate apparent performance and distort rankings. | Do not substitute point-adjusted F1 or binary sample F0.5 for the repository's parity-tested corrected ESA evaluator. This paper does not establish that our evaluator or model is invalid. |
| [Dragoi et al., AnoShift](https://proceedings.neurips.cc/paper_files/paper/2022/hash/d3bcbcb2a7b0b4716bf24ce4b2ea8d60-Abstract.html), abstract | A time-indexed anomaly benchmark distinguishes nearby and more distant distributions and observes degradation over time. | Report chronological transfer separately from calibration fit. Network-intrusion evidence motivates a question; it does not validate a satellite detector. |
| [Barber et al., conformal prediction beyond exchangeability](https://arxiv.org/abs/2202.13415), abstract | Exchangeability and algorithm symmetry support conventional coverage results; weighted methods address departures under specified conditions. | A nominal quantile threshold is a useful control, but overlapping windows and drift do not automatically inherit IID coverage guarantees. Event F0.5 is not a conformal coverage guarantee. |
| [Barber and Pananjady, time-series conformal inference](https://proceedings.mlr.press/v313/barber26a.html), abstract | Bounds depend on temporal dependence, including a switch coefficient, with results over stationary beta-mixing processes. | Do not claim arbitrary-drift or event-level false-alarm control from a fixed quantile. Applicability assumptions require examination. |
| [Xu and Boström, conformal anomaly thresholding](https://proceedings.mlr.press/v329/xu26a.html), abstract only; full PDF unavailable through the research browser | Proposes calibration-based thresholds using latent features and reports lower false alarms. | Worth keeping as a follow-up reference. The accessible abstract is insufficient to reproduce it or assert suitability for ESA; do not implement a guessed version. |
| [Allegrini and Pompei, hierarchical ESA ensemble](https://arxiv.org/html/2605.06681v1), sections 3–4 | Uses statistical and shapelet features, supervised base/stacking models, separate aggregation data, and challenge evaluation. | Shape and local statistics may provide information absent from reconstruction error. Adopting the whole system would change supervision, representation and aggregation simultaneously; its challenge scores cannot be ranked against our reused seven-year benchmark. |
| [Nguyen et al., deployable ESA benchmark study](https://arxiv.org/html/2607.07335v1), sections III–V | Compares supervised neural models with statistical methods; supervised and unsupervised preprocessing uses different sampling intervals. The displayed event-precision equation lacks our nominal-duration correction. | Useful for computational baselines, not numerical evidence that a model exceeds our baseline under the same protocol. A matched local implementation and metric audit would be required. |

## What the project evidence now supports

The established reference remains the eight-input Transformer autoencoder with
system F0.5 0.805722. Its local score 0.862907 is not the all-channel result.
Removing short rolls improved mean scores in the historical 2003–2005 comparison,
but 2004 remained variable. The subsequent internal attempts did not pass their
advancement gates. These findings are preserved, not replaced by the new audits.

The new calibration studies use only existing development annotations and saved
calibration scores. The key distinction is between three different problems:

1. **Information coverage:** events outside the annotated input channels may be
   harder to observe. An annotation outside an input is not proof of physical
   unobservability, so counts alone cannot choose an optimal subset.
2. **Score separation:** some missed events have lower reconstruction scores
   than many nominal samples. Any fixed monotone transform preserves this ordering;
   it cannot repair that ranking. Context-dependent scores could change ordering,
   but the already-tested score-change method did not transfer reliably.
3. **Selection support:** a high calibration F0.5 may rest on one temporal
   component and a small set of classes. More threshold trials do not supply
   new event evidence. The 1% cap is an alarm-occupancy policy, not a theorem
   controlling false alarms in future years.

## Method decisions

| Candidate action | Decision | Reason |
|---|---|---|
| Another Transformer architecture, loss-weight sweep or seed search | Reject for this phase | Does not isolate the measured coverage, ranking and calibration-support failures; similar recent variations failed. |
| Relax thresholds/cap or apply a monotone score normalization | Reject as the main fix | Cannot resolve low anomaly/nominal ranking; cap removal already failed to help the 2002 calibration optimum. |
| Conformal thresholding | Defer, retain as a possible calibration control | Needs an explicit dependence/drift argument and does not fix missing information. No universal operational guarantee is assumed. |
| Immediately expand to 76 inputs | Defer | Increased coverage can also increase nuisance scores; earlier broader/specialist comparisons did not establish reliable gains. Training-only input evidence must precede a matched comparison. |
| Copy the supervised hierarchical ensemble | Defer | Its real-anomaly supervision and multi-stage fitting change the question and require a larger label budget and stricter temporal separation. |
| Quantify preprocessing retention and calibration support | Execute now | Small, falsifiable diagnostics can prevent spending another training campaign on an unsupported explanation. |
| A compact shape/statistics representation alongside reconstruction | Candidate for a later matched pilot, not adopted | Mechanistically could detect changed waveform behavior without requiring high reconstruction error. First establish label-independent signal retention and a fixed temporal protocol. |

The decision is to strengthen the measurement and selection protocol before a
new model comparison. Negative diagnostic results are research outcomes: no
improvement is claimed, no final-test threshold is chosen, and no candidate is
promoted merely because a paper reports a high score.

Local evidence: [calibration plan](calibration-consolidation-plan-2026-10-13.md),
[calibration cap audit](calibration-selection-audit-2026-10-08.md),
[missed-event audit](inner-event-audit-2026-10-08.md),
[short-roll comparison](short-no-roll-multiseed-2026-10-02.md), and
[preprocessing reconstruction plan](preprocessing-retention-plan-2026-10-13.md).
