# F0.5 evaluation audit

Date: 2026-09-15. Reviewed research commit: `2de1a60`.

## Finding: the active thesis metric is not the official ESA metric

`esa_thesis/metrics.py:fbeta` correctly implements the F-beta equation:

`F0.5 = 1.25 * P * R / (0.25 * P + R)`.

However, `metrics()` supplies this precision:

`P_legacy = TPe / (TPe + FPe + FPt/Nt)`.

The official corrected event-wise precision is instead:

`P_corrected = TPe / (TPe + FPe) * (1 - FP_nominal_duration / nominal_duration)`.

`R = TPe / (TPe + FNe)` uses detected/missed annotated event IDs. On a regular
sample grid, `FPt/Nt` approximates the duration ratio, but resampled labels and
closed interval boundaries can still differ from raw annotations.

The local bundled `timeeval/metrics/ESA_ADB_metrics.py` implements the official
multiplicative correction. Its `ESAScores(betas=0.5).score(...)` returns the desired
`EW_F_0.50` value. The active thesis training code does not call that class.
The current upstream implementation confirms the same correction:
[official source](https://github.com/kplabs-pl/ESA-ADB/blob/main/timeeval/metrics/ESA_ADB_metrics.py).

The defect is not a harmless rename. When predictions cover the entire nominal
time range, the official correction sets precision to zero. The additive form
can remain close to one if there are many true events. A saturation flag is a
separate selection heuristic; it does not repair the metric.

## Additional differences that matter

1. **Event identity:** current `find_events` counts contiguous binary segments;
   ESA scoring groups annotation intervals by ID. Different events may overlap;
   one event may have several intervals. These are not equivalent label formats.
2. **Scope:** the default research loader uses all per-channel labels even when
   only a few feature channels are selected. A subset input is not automatically
   a subset evaluation. The official channel-aware metric is a separate measure
   of correctly localized affected channels, not subset event recall.
3. **Categories:** `get_y_any` treats every positive label code as anomalous,
   potentially including communication gaps. A deliberate category policy is
   required. There are no gap IDs in the raw test interval audited here.
4. **Protocol:** the current 20% training-tail validation is around 17 months;
   the ESA protocol uses the last three calendar months. Both are chronological,
   but their scores are not interchangeable benchmark results.
5. **Selection:** checkpoint selection and threshold ranking use the incorrect
   custom metric. Some utilities also report explicitly test-selected diagnostics.
   Current primary mask selection uses validation, which is an improvement, but
   the test has already been inspected repeatedly across experiments.
6. **Timing:** retrospective reconstruction and future-based filling can improve
   offline apparent localization. Causal performance needs separately timestamped
   predictions and an explicit detection-delay policy.

Implementation correction must cover the main metric and duplicated calculations
in `evaluation/legacy_scores.py`, `legacy_evaluation.py`, `mlp_ensemble.py`, plus
other subset/reselection utilities. The latter also have scope-specific event
counting and cannot be fixed safely with a blanket formula replacement.
Centralize the evaluation definition rather than maintaining several variants.

## Read-only audit procedure

- Verified every timestamp in the 7,364,161-row test CSV is on a continuous
  30-second grid from 2007-01-01 00:00:00 to 2014-01-01 00:00:00.
- Joined raw `labels.csv` and `anomaly_types.csv`, retained intervals overlapping
  that range, clipped them to the scoring range, removed channel duplication,
  and retained IDs. There are **91 event IDs: 55 anomalies and 36 rare events**.
  All annotated channels are target channels in `channels.csv`.
- Loaded each saved primary `test_pred_mask.npy` without changing predictions,
  thresholds, weights, or ground truth. Converted changes in binary predictions
  to timestamp/value pairs, retaining the start and final sample endpoints.
- Called the bundled, unmodified ESAScores with `betas=0.5`, explicit `full_range`,
  and `select_labels={'Category': ['Anomaly', 'Rare Event']}`.
- Imported its modules under a private audit namespace to avoid the unrelated
  legacy TimeEval package initializer/dependency conflict. The metric source and
  its dependencies were not reimplemented or edited. Its source hash and the
  prediction-file hashes are recorded in the local audit artifacts.

This produces **diagnostic scores from the official metric implementation on old
saved predictions**. It does not make their model selection, split, cached
provenance, or operational timing a fully compliant new benchmark experiment.

## Results

| Saved primary mask | Stored custom F0.5 | Formula-only correction | Official implementation, raw IDs |
| --- | ---: | ---: | ---: |
| `pa_clust_med_t03_6ch` | 0.8452 | 0.7373 | 0.7245 |
| `pa_center_ch12_38_49` | 0.7458 | 0.7198 | 0.7378 |
| `center_ch12_38_49` | 0.7548 | 0.7277 | 0.7373 |
| `center_ch08_40_47` | 0.6965 | 0.6959 | 0.7103 |

All **31** primary masks were audited. Recomputing the legacy metrics from the
cached ground-truth/prediction arrays matched their stored summaries (no mismatches).
The highest diagnostic annotation-ID F0.5 among these masks is **0.7378**, for
`pa_center_ch12_38_49`; this observation must not be used as a new test-selected
model choice. None of these 31 masks reaches 0.85 under this scoring scope.
The highest-scoring mask has corrected precision 0.9413 and event recall 36/91
(0.3956); its principal limitation is missed events, rather than low precision.

The pseudo-anomaly six-channel model's stored score of 0.8452 becomes 0.7373 after
only correcting its precision formula and **0.7245** using raw annotation IDs and
the official duration-based implementation. Its validation predicted rate is about
1.00%, but its test predicted rate is **19.33%**. Calibration/distribution stability
is therefore a concrete issue to investigate, not merely an architectural hypothesis.

A synthetic 89-event always-positive detector returned **0.9911** under the current
custom metric and **0.0000** under ESAScores. This establishes the formula defect;
the pipeline's saturation filter may reject this candidate, but does not make the
formula correct.

The formula-only and annotation-ID columns answer different questions; corrections
are not guaranteed to move a score monotonically when event identity also changes.
These calculations did not tune new thresholds or train any model.

Local artifacts (excluded from Git) are under
`results_longrun/audits/f05_2026-09-15/`: `saved_mask_audit.csv`,
`audit_metadata.json`, `summary_consistency.json`, and the saved `audit.py` script.
The script requires the local data and saved masks plus the existing research
packages and `portion`; it deliberately reads the official metric in isolation.


## Required implementation before claiming the target

1. Add a production adapter around a pinned official metric implementation,
   with explicit annotation scope, categories, full time range, and metric version.
2. Validate perfect predictions, all-positive/all-negative predictions, fragmented
   alarms, overlapping IDs, category exclusions, point events, split boundaries,
   and irregular timestamps. The official class also calculates affiliation
   metrics; inspect empty-event edge cases rather than assuming those always work.
3. Use one adapter for validation checkpoint selection, thresholds, ensembles,
   and final reporting. Rename old outputs `legacy_*`; never overwrite old results.
4. Align the declared split and event scope; freeze the development/calibration
   process before a final evaluation. New thresholds alone cannot undo historical
   test-informed development.
5. Report corrected event precision/recall, nominal alarm duration, false alarms
   per day, event timing, and channel localization alongside F0.5.

No production metric or training behavior was changed during this audit.
The improvement experiments are in [the research plan](f05-improvement-plan.md).
