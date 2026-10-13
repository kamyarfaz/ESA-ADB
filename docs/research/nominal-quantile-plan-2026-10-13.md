# One matched calibration-policy comparison

Declared before generating rules or loading validation scores, 13 October 2026.

## Hypothesis

The new influence audit shows that the 2001 event-selected threshold depends on
one overlapping component; removing it makes every alarm-cap-eligible candidate
uninformative. Test whether using only the nominal calibration tail for the
threshold reduces this dependence enough to improve chronological transfer.
This may fail because it cannot improve the ranking of a fixed score. Such a
failure is informative and stops this calibration-only approach before retraining.

The fixed nominal-tail reference is motivated by quantile-based thresholding
discussed in [Nguyen et al., section IV-B](https://arxiv.org/html/2607.07335v1).
We choose exactly q=0.999 for this diagnostic; there is no quantile search.
This is an empirical threshold, **not a claimed conformal guarantee**, and the
paper's published results are not a matched comparison with our metric.

## Fixed design

- Use the six completed joint-control runs in `inner_selection_cuda_v2`,
  years 2001/2002, seeds 42/43/44; original train cutoffs and real half-year
  calibration/assessment periods remain fixed.
- Use **epoch 5 for both arms**, isolating threshold policy from checkpoint
  selection. This does not replace the historical primary checkpoint policy.
- Control: original `fixed_epoch5` calibration-selected threshold and its
  postprocessing. Candidate: the 99.9th percentile (NumPy linear interpolation)
  of nominal calibration scores, with the **same control postprocessing**.
  Consequently the candidate is not wholly label-free: nominal labels exclude
  annotations, and inherited postprocessing was previously calibrated.
- Nominal timestamps exclude all supplied raw annotation intervals, including
  neutral communication gaps. Use closed interval membership on the existing
  30-second grid. At least one nominal score must exist; reject nonfinite scores.
- Do not apply another search, cap relaxation, per-year tuning or fallback.
  The original 1% total-alarm cap applies to the control's selection. The
  candidate has a fixed nominal quantile instead; report total occupancy and
  empirical nominal alarm duration explicitly. A nominal quantile does not
  guarantee an event alarm rate or an occupancy cap after postprocessing.
- Verify all dependencies against the original manifest. Save all six candidate
  rules and a single hash-protected decisions file before loading any subsequent
  assessment score array. Never load model weights or run inference/training.
- Reproduce original fixed-epoch calibration/assessment predictions and event
  metrics before accepting comparison results. Report corrected system and
  input-channel event scores, TP/FP/FN, precision/recall, false-alarm duration,
  total occupancy and all seeds. No pointwise F0.5 substitution.

## Advancement and stopping

Advance only if candidate mean system F0.5 is strictly higher in **both** years,
with no increase in mean false-event count or nominal false-alarm duration in
either year. Report every paired result and sample SD. A tie fails advancement.
This is a conservative research decision rule, not a significance test. No
full-period run or ensemble follows automatically even if it passes.

If it fails, stop quantile/cap tuning here. Preserve the baseline and turn the
next protocol toward representation/observability rather than trying nearby
quantiles until one works. These repeatedly inspected years are development
data; this comparison cannot establish independent generalization or the thesis
target. The fixed train/calibration/testing reference files remain unchanged.
