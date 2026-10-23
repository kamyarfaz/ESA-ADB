# Documentation index

## Thesis research

- [Completed three-combination comparison](research/combinations-full-results-2026-10-22.md): verified 2007–2013 annual and pooled metrics, controls, and limitations.

- [Project guide](../README.md): layout and commands.
- [Research assessment](thesis-research-context.md): scientific findings and limitations.
- [Source migration](source-migration.json): old and new research filenames.
- [Repository organization](repository-organization.md): folder responsibilities.

- [F0.5 evaluation audit](research/f05-evaluation-audit.md): official-metric comparison on saved predictions.
- [F0.5 improvement plan](research/f05-improvement-plan.md): prioritized, evidence-based experiments.

- [Corrected evaluation implementation](research/corrected-evaluation.md): tests, recalibration results, and data-integrity finding.

- [Controlled scoring comparison](research/scoring-comparison.md): paired inference, validation-only selection, and GPU command.

- [Chronological development baseline](research/chronological-development.md): fixed folds, isolated assessment, and training commands.

- [Development baseline findings](research/development-baseline-findings.md): verified missed events, false-alarm attribution, and next experiment specification.

- [Forecasting development comparison](research/forecast-development.md): causal scoring, matched controls, and run commands.

- [Causal alarm findings](research/causal-alarm-findings.md): frozen-score confirmation experiment, false alarms, and detection delays.

- [Expanded-channel AE experiment](research/expanded-channel-ae.md): fixed 11-channel comparison against the existing baseline.

- [Expanded-channel results](research/expanded-channel-results.md): completed three-fold comparison and verified event recovery.

- [Channel-error diagnosis](research/channel-error-findings.md): training residual scales and verified false-alarm channel attribution.

## Original benchmark

- [Benchmark instructions](benchmark-original.md).
- `index.rst`, `api/`, `concepts/`, `dev/`, `user/`: original Sphinx documentation.
- `assets/`: benchmark illustration and TimeEval logo.
- `_static/`: Sphinx styling; `conf.py`, `Makefile`, `requirements.txt`: docs tooling.

- [Channel operating ranges](research/channel-regime-findings.md): 2005 range shifts, clipping checks, and alarm-context diagnostics.
- [Residual Transformer experiment](research/residual-transformer.md): history-only level centering and a matched forecasting comparison.
- [Reconstruction-only AE ablation](research/reconstruction-objective.md): test the pseudo-anomaly objective with architecture and evaluation fixed.
- [2004 baseline alarm audit](research/baseline-2004-findings.md): frozen false alarms, score overlap and missed-event coverage.
- [Separate specialist detector](research/specialist-detector.md): channels 14/21/29 and frozen-baseline combined calibration.
- [Data-first supervisor plan](research/data-first-plan.md): training-only channel audit, small-subset screening and presentation structure.
- [All-input training review](research/all-input-review-2026-09-23.md): all 76 telemetry channels and 698 telecommand types in 2000–2002.
- [Counter preprocessing trace](research/counter-preprocessing-trace-2026-09-23.md): why raw changes disappear from nominal prepared data.
- [Seven-year descriptive review](research/seven-year-input-review-2026-09-23.md): 2000–2006 yearly inventory and telemetry atlas, separate from model selection.

- [Full 14-year Mission 1 review](research/full-mission-input-review-2026-09-23.md): all 774 inputs across 2000–2013, with annual statistics and telemetry plots.

- [Long-period baseline](research/long-baseline.md): channels 40–47; train 2000–2005, calibrate 2006, assess 2007–2013; uncapped nominal windows and recovery.

- [Historical channel experiment inventory](research/historical-channel-review-2026-09-23.md): nine completed correlation/representative variants and the preprocessing/correlation work still needed.

- [Preprocessing and corrected training-period correlation review](research/preprocessing-and-correlation-policy.md) — all 76 telemetry channels, 2000–2005, with yearly matrices and diagnostic groups.

- [Completed longer baseline results](research/long-baseline-results-2026-09-24.md) — pooled and yearly assessment, selected checkpoint and limitations.

- [Training scaling review](research/scaling-review-2026-09-24.md) — exact scaling/clipping audit for all 87 prepared inputs.

- [Counter aggregation review](research/counter-aggregation-2026-09-24.md) — causal binning, change retention and missingness limitations.

- [Channels 40–47: data and improvement review](research/ch40-47-improvement-review-2026-09-24.md) — current scope, measured findings, primary research sources and prioritized scoring diagnostic.

- [Channels 40–47 diagnostic command](research/channel-diagnostics.md) — fixed-checkpoint inference, progress and stage recovery.

- [Channel-normalization comparison](research/channel-normalization-results-2026-09-24.md) — completed calibration-only ablation; original scoring retained.

- [Masked-reconstruction pilot](research/masked-pilot.md) — matched small comparison on channels 40–47, with epoch recovery.

- [Input-channel versus system-level evaluation](research/dual-scope-evaluation.md) — supervisor request and same-prediction baseline comparison.

- [Completed 2003 masked pilot](research/masked-pilot-results-2026-09-24.md) — matched baseline comparison and both evaluation scopes.

- [Masked pilot follow-up](research/masked-pilot-followup-2026-09-24.md) — three-year comparison; masked improvement did not hold consistently.
- [2004 calibration diagnosis](research/masked-calibration-diagnosis-2026-09-24.md) — single-event calibration and deteriorating detection despite falling training loss.
- [Training objective audit](research/masked-objective-audit-2026-09-25.md) — fixed training probe and synthetic corruption versus hidden-target overlap.
- [Hidden-target overlap pilot](research/masked-overlap-pilot.md) — controlled placement comparison, protocol and run command.
- [Hidden-target overlap results](research/masked-overlap-results-2026-09-25.md) — marginal gain, poor calibration; variant paused.
- [Baseline calibration event review](research/baseline-event-review-2026-09-25.md) — all six input-annotated events detected; missed system events and score margins.
- [Baseline consistency across 2003–2005](research/baseline-years-review-2026-09-25.md) — sparse calibration, false-alarm episodes and local/system misses.
- [Frozen 2004 calibration diagnostic](research/baseline-2004-diagnostic.md) — checkpoint-preserving per-channel plots for calibration alarms.
- [Completed 2004 calibration diagnostic](research/baseline-2004-diagnostic-results-2026-09-25.md) — differing false-alarm patterns and repeated prepared values.
- [Native timestamp review](research/native-alarm-review-2026-09-25.md) — longer observation intervals and held values around the January24 false alarm.
- [Cadence frequency audit — provisional](research/cadence-frequency-review-2026-09-25.md) — repeated CRC/timestamp failures block acceptance of new findings.
- [Direct integrity investigation](research/integrity-investigation-2026-09-25.md) — two guarded cadence audits reproduced exactly; underlying intermittent fault unresolved.
- [Observation-age prototype](research/observation-age-prototype.md) — tested causal timing features; full data build stopped by integrity checks.
- [Single-calculation isolation](research/single-age-calculation-2026-09-25.md) — NumPy and scalar Python agree; intermittent pipeline failure not reproduced.
- [Baseline inference repeatability](research/inference-repeatability-2026-09-25.md) — two independent CPU runs matched exactly across228 windows.

- [Full-period timing comparison: protocol, recovery, and run command](research/full-period-timing-comparison.md)

- [Full-period timing comparison results](research/timing-comparison-results-2026-09-26.md)

- [2006 calibration stability audit](research/calibration-stability-2006-2026-09-26.md)

- [2005 calibration missed-event diagnostic](research/baseline-2005-diagnostic-2026-09-26.md)

- [Alarm-cap and threshold-grid audit](research/alarm-cap-audit-2026-09-26.md)

- [Partial calibration grid/cap comparison](research/grid-cap-comparison-2026-09-26.md)

- [2003 calibration threshold-boundary investigation](research/calibration-boundary-2003-2026-09-26.md)

- [Completed calibration grid/cap comparison](research/grid-cap-comparison-complete-2026-09-26.md)

- [Grid/cap development assessment: candidate rejected](research/grid-cap-development-2026-09-26.md)

- [Training/calibration score stability](research/training-calibration-stability-2026-09-26.md)

- [Nominal-window reconstruction inspection](research/nominal-window-inspection-2026-09-26.md)

- [Existing checkpoint comparison on nominal windows](research/nominal-reconstruction-comparison-2026-09-26.md)

- [Both models on 2005 calibration events](research/calibration-model-comparison-2005-2026-09-26.md)

- [id_98 local channel and augmentation review](research/id98-local-review-2026-09-26.md)

- [Baseline repeat verification](research/baseline-repeat-2026-09-27.md)
- [Local event-shape inventory and proposed experiment](research/local-event-shapes-2026-09-27.md)

- [Optional short coordinated augmentation](research/short-coordinated-augmentation.md)

- [Short coordinated augmentation comparison command](research/short-coordinated-comparison.md)

- [Short coordinated augmentation pilot results](research/short-coordinated-results-2026-09-27.md)

- [Three-seed short coordinated augmentation review (29 September)](research/short-coordinated-multiseed-2026-09-29.md)

- [2004 augmentation diagnosis and training-only objective probe](research/augmentation-2004-diagnosis-2026-09-29.md)

- [Short corruption without temporal rolls pilot](research/short-no-roll-comparison.md)

- [Three-seed no-roll results](research/short-no-roll-multiseed-2026-10-02.md)
- [No-roll 2004 calibration diagnosis](research/short-no-roll-2004-diagnosis-2026-10-02.md)
- [Chronological inner-selection comparison protocol](research/inner-validation-plan-2026-10-02.md)
- [Completed inner-selection results and false-alarm comparison](research/inner-selection-results-2026-10-02.md)
- [Channel reconstruction and calibration-score diagnosis](research/inner-score-diagnosis-2026-10-02.md)
- [Matched artificial-anomaly loss-weight comparison](research/loss-weight-comparison-2026-10-02.md)

- [Loss-weight comparison results](research/loss-weight-results-2026-10-03.md): verified matched real-data comparison, all seeds, and failed advancement gate.

- [Loss-weight temporal diagnosis](research/loss-weight-diagnosis-2026-10-03.md): frozen monthly scores, event ratios and prolonged false alarms.

- [Frozen channel replay](research/loss-weight-channel-replay-2026-10-03.md): channel 47 level changes, reconstruction errors, and exact six-model score reproduction.

- [Channel-level intervention](research/channel-level-coupling-2026-10-03.md): frozen-model evidence of cross-channel sensitivity and limits of centering.

- [Channel-isolation experiment](research/channel-isolation-plan-2026-10-08.md): parameter-matched diagonal channel attention with a frozen joint-model control.

- [Verified channel-isolation results](research/channel-isolation-results-2026-10-08.md).

- [Missed-event audit](research/inner-event-audit-2026-10-08.md): calibration transfer, score ranking, and 13 prepared-telemetry event plots.

- [Score-change pilot](research/score-change-results-2026-10-08.md): fixed-checkpoint temporal scoring, failed advancement gate, verified paired results.

- [Direct telemetry reference results](research/telemetry-reference-results-2026-10-08.md): deterministic comparison, missed events, and verified calibration-transfer failure.

- [Calibration selection audit](research/calibration-selection-audit-2026-10-08.md): alarm-cap restrictions versus precision/recall trade-offs across 20 saved grids.

## Move to a new PC or chat

Read [current AI handoff](../AI_HANDOFF.md) and [migration guide](../MIGRATION.md).
The [archived handoff](history/AI_HANDOFF-through-2026-10-08.md) preserves the full
earlier chronology; its pending-run notes are historical.

- [False-alarm reduction: both models, verified negative results](research/false-alarm-reduction-results-2026-10-23.md)
