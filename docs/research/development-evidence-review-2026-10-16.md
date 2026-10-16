# Development evidence and next diagnostic decision

## Decision

Do not repeat channel expansion, lower thresholds, or replace the architecture
without locating the remaining score failure. The next bounded comparison is
**channel-resolved reconstruction error versus the existing channel-maximum
score**, using the six frozen internal joint models. This is a diagnostic replay,
not a new deployed detector, a threshold search or a training run. Its protocol is
below; it has not yet been executed.

## Evidence established before the benchmark period

Recomputed from original-hash raw annotations restricted to 2000–2005:

| Period | All unique event IDs | IDs annotated on inputs 40–47 | Outside inputs only |
|---|---:|---:|---:|
| 2000 | 44 | 10 | 34 |
| 2001 | 14 | 7 | 7 |
| 2002 | 8 | 6 | 2 |
| 2003 | 9 | 7 | 2 |
| 2004 | 11 | 8 | 3 |
| 2005 | 10 | 9 | 1 |
| Pooled 2000–2005 | 95 | 46 | 49 |

The coverage gap exists before 2006: 49/95 IDs lack annotations on the eight
inputs. Pooled IDs are deduplicated; annual rows are not additive. These are
annotation coverage counts, not detection rates, usable training anomalies, or
proof that unannotated channels lack correlated signals. The current training
objective uses globally nominal windows and artificial anomalies, not these real
anomaly windows as positive training examples.

The verified historical internal score audit covers all three seeds in each fold:

| Real development assessment | Input-annotated IDs | Peak score / frozen threshold | Fraction of nominal scores at least event peak |
|---|---:|---:|---:|
| Jul–Dec 2001 | 3 | 0.0105–0.1218 | 0–42.50% |
| Jul–Dec 2002 | 2 | 0.8326–0.9632 | 49.07–67.50% |

Ranges span event/seed pairs, not confidence intervals. None of these IDs is
detected by the primary joint models. In 2002 many nominal timestamps outrank
even the event maximum, so lowering a single global threshold cannot separate
these examples cleanly. In 2001 the pooled range does not support one uniform
diagnosis: some peaks rank highly but remain far below the calibrated threshold.
These nominal fractions are retrospective diagnostics, not event FPR or deployable
alarm rates. Source hashes were checked; the score values are reused from the
verified prior audit rather than recomputed by new inference.

## Earlier experiments constrain the next step

| Already tested | Observed limitation | Consequence |
|---|---|---|
| Inputs 14/21/29 added | Recovered targeted development IDs; false events reached 33 and 35 in 2004/05 | More coverage alone is insufficient; do not rerun this as a new idea |
| Fixed training-q99 channel residual scaling | Same six 2006 calibration IDs; more nominal alarm time | Do not retune normalization percentiles |
| Channel isolation, three seeds | No recovered internal events; more 2002 false events | Removing channel attention alone is not supported |
| Masked reconstruction | 2003 gain; substantial 2004 deterioration; 2005 essentially tied | Masking is not an established improvement |
| Level-residual forecasting | 2003 F0.5 decreased from 0.086696 to 0.070614 | Do not assume centering solves level drift |
| Two-sided recent-score deviation | Only one of three seeds detects one 2002 event; 2001 remains silent | Do not optimize half-life on the same outcomes |
| Direct telemetry deviation | Silent in both internal assessment periods | Reconstruction is not the only possible source of calibration failure |
| Nominal q=0.999 calibration | Candidate 2002 alarm occupancy 87.2% | Do not launch neighboring-quantile sweeps |

Sources: `expanded-channel-results.md`, `channel-normalization-results-2026-09-24.md`,
`channel-isolation-results-2026-10-08.md`, `masked-pilot-followup-2026-09-24.md`,
`residual-transformer.md`, `score-change-results-2026-10-08.md`,
`telemetry-reference-results-2026-10-08.md`, `nominal-quantile-results-2026-10-14.md`.
These experiments use different budgets and periods; the table is an evidence
inventory, not a numerical ranking of techniques.

## Primary-source literature check

1. [ESA-ADB benchmark](https://arxiv.org/abs/2406.17826): the benchmark uses real,
   annotated satellite telemetry and was designed to address evaluation gaps in
   existing anomaly-detection datasets. It supplies context, not evidence that
   any proposed representation will meet this project's target.
2. [RevIN, official author implementation and paper link](https://github.com/ts-kim/RevIN):
   per-instance centering/scaling and subsequent inversion address distribution
   shift in forecasting. This supports examining level sensitivity, but forecast
   accuracy is not anomaly discrimination. Our inference: removing a window's
   level can also remove evidence of a sustained level anomaly; restoration alone
   does not prove that an autoencoder detects such anomalies. No RevIN experiment
   or claimed improvement is justified solely by this citation. The OpenReview
   page was access-blocked; this review inspected the authors' implementation
   description, not inaccessible full-paper experiments.
3. [Nguyen et al., 2026 preprint](https://arxiv.org/html/2607.07335v1): in its
   Mission 1 results, expanding the channel set helps the supervised GAT but
   reduces ECOD performance. Thus expansion is method-dependent, not a general
   guarantee. Its stated precision equation lacks our nominal-duration penalty;
   its results are not directly interchangeable with our corrected metric. Its
   validation-percentile approach has already motivated a completed negative
   diagnostic here. It is not grounds to repeat that diagnostic.

No reviewed source demonstrates that a new architecture will solve this project's
specific calibration/score-transfer failure. Literature mechanisms and local
evidence are kept separate from performance claims.

## One frozen-model diagnostic comparison

Question: does maximum aggregation obscure useful error changes on annotated
input channels, or are those channels' errors themselves poorly separated from
nominal behavior? The prior scalar-score audit cannot answer this; the earlier
channel-47 intervention established sensitivity, not per-event separation.

- Use only `inner_selection_cuda_v2`, folds 2001/2002, seeds 42/43/44 and each
  saved primary calibration-selected checkpoint. No favorable epoch/seed choice.
- Preserve inputs 40–47, saved scalers, clipping, windows, stride and window-mean
  aggregation. Reconstruct per-channel errors with the existing joint model.
- Read only each fold's original training/calibration and real development
  assessment periods. Exclude 2006 and 2007–2013. Verify manifests and input
  hashes before inference; save to a new directory.
- First take the channel maximum and require reproduction of saved scalar scores
  within a stated numerical tolerance (rtol 1e-5, atol 1e-6). Replay the original
  threshold/postprocessing and require exact saved-mask agreement. Stop and
  investigate a mismatch; never recalibrate to conceal it.
- For every annotated input-channel interval, measure its own maximum error and
  the fraction of nominal scores on that same channel exceeding it. Use actual
  channel intervals, not an ID-wide union assigned to every channel. Report
  all events/seeds, upward and downward changes, and the channel contributing
  the existing maximum during those intervals. Keep per-channel diagnostic ranks
  separate from deployment metrics; do not build an annotation-aware detector.
- Report nominal per-channel error distributions by calendar month and split to
  distinguish persistent drift from isolated peaks. Fit no new model or scale.
- No new F0.5 result is expected: original predictions remain unchanged. No
  channel weights, thresholds, ensembles or input selection are derived from
  development assessment ranks in this diagnostic.

If annotated-channel errors are also poorly separated, another aggregation or
threshold tweak lacks support; investigate representation/objective mismatch.
If useful evidence is consistently hidden by another channel, a separately
specified training-fitted score comparison may be justified. Either finding
requires a new explicit protocol before training or changing deployed predictions.
This conditional diagnostic is narrower than assuming a new architecture works.

## Artifacts and verification

`scripts/research/consolidate_development_evidence.py` produced
`tables/development-evidence-2026-10-16/`. All annual coverage counts reconcile
with the previous diversity audit; pooled counts use unique IDs. Original
annotation and internal event-audit hashes match the migration manifest.
No training, inference, new threshold evaluation or benchmark-score reading was
performed in this consolidation. Existing reports were read as historical evidence.
