# Short coordinated augmentation: three-seed development review

## Conclusion

Do not promote the candidate to a full-period run yet. Gains in 2003 and 2005
are accompanied by a substantial 2004 regression and sensitivity to random seed.
The thesis system F0.5 target has not been robustly achieved.

## Matched protocol

Channels 40–47; seeds 42, 43 and 44; five epochs; up to 25,000 globally nominal
training windows per fold. Train before January of the fold year, calibrate
January–March, assess April–December. Compare legacy pseudo-anomalies with the
50/50 legacy plus short coordinated mixture. Saved protocols match exactly apart
from seed. Both arms retain the original scoring, threshold grid and alarm cap.
These are previously inspected development periods, not an untouched test.
These controls have a smaller budget than the historical 40-epoch baseline.

## System event F0.5

Each cell is mean ± sample standard deviation over three seeds. These are
arithmetic summaries, not pooled event metrics or confidence intervals.

| Assessment year (April–December) | Legacy control | Short coordinated |
|---|---:|---:|
| 2003 | 0.642418 ± 0.171621 | 0.836292 ± 0.123532 |
| 2004 | 0.415934 ± 0.364273 | 0.241707 ± 0.123166 |
| 2005 | 0.848011 ± 0.036528 | 0.931322 ± 0.065321 |

Across nine seed/year pairs: six increases, two decreases, one exact tie.
Counting wins alone conceals the magnitude of the 2004 seed44 regression
(0.832848 to 0.209270). For 2004, legacy versus candidate event TP/FP/FN:

| Seed | Legacy | Candidate |
|---|---|---|
| 42 | 4 / 13 / 6 | 5 / 9 / 5 |
| 43 | 3 / 18 / 7 | 6 / 45 / 4 |
| 44 | 5 / 0 / 5 | 6 / 27 / 4 |

The candidate detects more 2004 events but produces many more false alarms in
seeds43/44. This hurts precision, which F0.5 emphasizes. Calibration is also
weak: candidate seed42 has zero calibration F0.5 at every epoch; legacy seed43
selects a zero-score checkpoint. Earliest ties are respected. The observed
assessment improvements cannot be used to select a seed or retune a rule while
calling those same periods independent validation.

## Verification and saved outputs

Reconstructed all 18 predictions exactly from saved scores and frozen rules.
Recomputed all 36 local/system metric rows with the corrected evaluator within
absolute tolerance 1e-9. Checked five completed epochs and earliest-best
calibration checkpoint selection for every arm. This review did not rerun model
inference, deserialize checkpoints, or rehash the full prepared dataset.

Source runs: results_longrun/development/short_coordinated_seed{42,43,44}_v1.
Compact results and verification record:
[tables](tables/short-coordinated-multiseed-2026-09-29/).
Local and system summaries are both included; the same predictions are used for
both scopes. Source artifacts remain untouched. No new training or GitHub push.

## Next step

Keep the existing baseline. Diagnose 2004 training/calibration behavior across
all three seeds: nominal versus anomalous score distributions, channel residuals,
and calibration checkpoint stability. Use training/calibration for any subsequent
rule or model changes. Assessment failures can be documented as development
findings, but do not tune directly against them. Define one bounded comparison
only after this diagnosis; do not launch another long run or pick the best seed.
