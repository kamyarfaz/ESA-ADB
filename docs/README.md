# Documentation index

## Thesis research

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
