# Chronological selection audit before further model changes

Status: all six runs completed and verified on 2 October in
`results_longrun/development/inner_selection_cuda_v2`. Fixed epoch5 failed the
advancement gate. See [results](inner-selection-results-2026-10-02.md).

Run with `python -m esa_thesis inner-validation --output NEW_DIRECTORY --device cuda
--threads 2`. Append `--resume` with the identical output, source and arguments
to restore a stopped run. Epochs, seeds, folds and budget are fixed in the runner.
Every epoch saves weights, calibration scores/candidates and a full resume state.
Completion manifests protect saved results when skipping completed jobs.

The restricted session initially exposed no GPU; a host-level check found CUDA.
An incomplete CPU attempt and a pretraining GPU attempt are preserved separately
as `inner_selection_cpu_v1` and `inner_selection_cuda_v1`. Neither is part of the
planned comparison. The v2 directory uses the corrected resumed-summary handling.

## What the earlier data provides

Inventory uses Anomaly and Rare Event annotation IDs, separately for all system
channels and inputs 40–47. All analyzed periods end before January 2003; no outer
assessment telemetry or scores are used. IDs crossing quarter boundaries count
in both quarters, so quarter counts must not be summed as unique-event totals.

| Internal fold | Nominal training | Real calibration | Calibration IDs: system / inputs | Real internal validation | Validation IDs: system / inputs |
| --- | --- | --- | ---: | --- | ---: |
| 2001 | 2000, before Jan 2001 | Jan–Jun 2001 | 7 / 4 | Jul–Dec 2001 | 7 / 3 |
| 2002 | 2000–2001, before Jan 2002 | Jan–Jun 2002 | 5 / 4 | Jul–Dec 2002 | 3 / 2 |

No annotation ID is shared between calibration and validation within either
proposed fold. Q4 2002 has no selected event IDs: retain it to measure false
alarms, report its event F0.5 as uninformative for detection, and include its
nominal duration in the full half-year evaluation. Do not remove quiet periods.
These counts establish annotation coverage, not data quality or independence.

## Representativeness: evidence and limits

The current Q1 2004 calibration has only one selected ID, unlike these earlier
half-year periods. Existing nominal-data diagnostics also show that stable median
errors conceal changing upper tails: no-roll seed44 selected-checkpoint nominal
window q99 is 2.177 in January and 8.324 in February. This is consistent with
normal behavior that the model reconstructs unevenly, but does not identify
physical spacecraft modes. Those plots compare observed periods, not validated
operating-state labels. More events alone will not solve score overlap.

Before running the proposed folds, check prepared CSV integrity, date coverage,
finite values, nominal-window availability, per-channel clipping, and monthly
training/calibration range and residual summaries. Fit scalers only on each
fold's nominal training data. Keep every input channel 40–47. Do not fit a
scaler, choose channels, or compute threshold references from validation data.

## One comparison, not another architecture search

Question: does choosing checkpoints from a sparse event F0.5 signal help compared
with a predetermined training budget?

Hold the architecture, scoring and short_no_roll training generator fixed. This
choice isolates selection within the just-completed candidate; it does not
promote that generator over the existing baseline. Train five epochs with at most
25,000 globally nominal windows, batch64, float32, seeds42/43/44. Two internal
folds require six training jobs, not separate training for each selection policy.
Use identical weights and calibration scores when comparing policies:

- **Existing policy:** select the epoch with the highest corrected calibration
  F0.5 under the unchanged 1% total-alarm cap; earliest epoch wins ties.
- **Fixed-budget policy:** always use epoch5; calibrate its threshold using the
  same grid, metric, postprocessing, cap and threshold tie ordering.

If all eligible calibration scores are zero, keep the documented policy's
selection and mark it `uninformative_calibration`; never silently switch epoch,
relax the cap or use validation to pick a threshold. Save all five checkpoints
and calibration candidates so the comparison is auditable. Freeze both policies
before loading internal validation. Synthetic data remains training-only.

Primary output is system event F0.5; also report input-scope F0.5, event TP/FP/FN,
precision/recall, nominal false-alarm duration/rate and alarms per day. The same
frozen predictions serve both annotation scopes. Report all three seeds, their
mean and sample SD, and each half-year separately. Do not select the best seed.

Evidence for proceeding requires the fixed policy to improve mean system F0.5
in **both** internal folds, with no increase in mean false-alarm events in either
fold. Ties retain the existing policy. Report recall tradeoffs regardless of this
gate. This is a conservative advancement criterion, not a statistical guarantee
or permission to claim the thesis target. If it fails, retain the baseline and
investigate score separation rather than expand a hyperparameter search.

## Preserve the external comparison

The current April–December 2003, 2004 and 2005 assessments remain exactly as
defined, with the same real observations and annotation scopes. Their January–
March calibration intervals are not silently expanded. The 2000–2005 training,
2006 calibration, 2007–2013 full-period reference also remains unchanged.

Internal six-month calibration is a separate selection study with earlier
training cutoffs; it does **not** prove that a policy will fix sparse Q1 2004
calibration. Any later assessment must label this protocol difference. Results
already inspected have influenced research, so neither these historical internal
periods nor the outer development periods should be presented as a pristine test.
Later folds use earlier observations chronologically; results across folds are
dependent. No old model may be used if its fitting data extend into an internal
validation period: earlier-cutoff models must be trained afresh.

## Reproducibility

Inventory script: `scripts/research/inventory_inner_validation.py`.
Counts, event IDs, carried-in events and source hashes:
`tables/inner-validation-2026-10-02/coverage.csv` and `provenance.json`.
Both annotation source hashes were stable before/after inventory. All 32 rows
end no later than January2003; within-fold calibration/validation ID intersections
are empty. No new F0.5 results, source training changes or GitHub push.
