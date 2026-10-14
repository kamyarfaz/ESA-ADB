# Fixed nominal-quantile comparison

The predeclared candidate failed advancement. Lowering the threshold recovered
some events but did not produce an acceptable detector across both years.
The established full-period baseline remains unchanged.

| Development year | Threshold policy | System F0.5 mean ± sample SD | Mean false events | Mean nominal false-alarm days | Total alarm occupancy |
|---|---|---:|---:|---:|---:|
| 2001 | Original epoch-five control | 0 ± 0 | 0 | 0 | 0% |
| 2001 | Nominal q=0.999 | 0.384075 ± 0.105221 | 1.33 | 0.0540 | 0.0926% |
| 2002 | Original epoch-five control | 0 ± 0 | 100 | 67.4337 | 36.7411% |
| 2002 | Nominal q=0.999 | 0.006370 ± 0.001437 | 34.67 | 156.0429 | 87.2041% |

Three paired seeds per year. These are July–December development periods, not
the 2007–2013 benchmark. This control uses epoch five by design, not the historical
primary calibration-selected checkpoint. Compare only matched rows here.

## Interpretation

The lower threshold helps recover weak event scores in 2001, but adds false alarms
to a previously silent control. In 2002, nominal scores frequently exceed the
calibration threshold, making alarms nearly continuous. Fewer false *events*
therefore do not mean less false-alarm *time*: long alarms merge into fewer intervals.
The corrected duration penalty captures this failure. Candidate mean local recall
is 1 in 2002, yet precision and F0.5 remain very low.

Both years fail the predeclared gate: 2001 increases false-event count and duration;
2002 increases false-alarm duration. No nearby quantile search, cap relaxation,
checkpoint substitution, full-period rerun or candidate promotion follows.
This closes this specific fixed-quantile hypothesis, not every possible calibration
method. Together with earlier score-ranking diagnostics, the result motivates
studying representation and operating-range transfer before further threshold tuning.

## Verification and limitations

All six controls reproduce saved calibration metrics, validation prediction arrays,
and both-scope validation metrics. Every consumed historical dependency matches
its original transfer manifest. Twelve rules (six matched pairs) were written and
SHA256-sealed before reading validation arrays. No model weights were loaded.
The known mismatching historical MLP array is absent from the dependency list.
The fixed train/calibration/testing and labels package also passed checksum verification.

The implementation and evaluator suite passes 120 tests, including neutral-gap
exclusion, closed annotation boundaries, empty/nonfinite rejection, and advancement
criteria. Full seed-level tables, thresholds, occupancies and provenance:
[tables](tables/nominal-quantile-2026-10-14/results.csv),
[summary](tables/nominal-quantile-2026-10-14/summary.csv),
[verification](tables/nominal-quantile-2026-10-14/verification_and_decision.json).
Predictions remain in `results_longrun/development/nominal_quantile_20261014_v1/`.

These repeatedly inspected development years are not independent confirmation.
Nominal masking uses annotations, and postprocessing is inherited from the original
calibrated control; the candidate is not wholly unsupervised or label-free.
The 99.9th percentile reference is motivated by
[Nguyen et al., IV-B](https://arxiv.org/html/2607.07335v1), with an explicitly adapted
nominal-only restriction. No finite-sample false-alarm guarantee is claimed.
