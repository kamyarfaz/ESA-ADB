# Full-period short-roll removal — user-requested extension

On 14 October the user explicitly requested assessment for every year 2007–2013
and a pooled seven-year F0.5. This overrides the previous decision not to extend
the unstable pilot automatically. It does not claim the pilot passed its gate.

Train channels 40–47 on globally nominal windows from 2000–2005. Use immutable
`data/frozen/mission1-reference-v1/train.csv`, `calibration.csv` (2006) and
`testing.csv` (2007–2013), plus the verified raw annotations and binary labels.
Every file and the manifest are checksum-gated before fitting; data and annotations
are rechecked at completion. Exclude the historical damaged MLP array entirely.

Run one fixed seed (42), 40 epochs, batch64, all eligible nominal windows with
length256/stride16, matching the established long-baseline budget. Use the same
Transformer AE, training-only median/IQR scaling, clipping, optimizer, learning
rate schedule, pseudo-anomaly objective and scoring. The augmentation is the
existing `short_no_roll` mixture: 50% legacy branch and 50% short offset/noise
branch. Legacy branch rolls remain. No representation or threshold-policy change.

Select checkpoint and threshold using corrected all-channel 2006 event F0.5 at
epochs5,10,...,40, earliest tie, original percentile/postprocessing grid and 1%
total-alarm cap. Save the frozen rule before assessment observations are loaded.
No annual refitting, threshold selection or seed selection using 2007–2013.

Score the full assessment continuously using window-mean reconstruction scoring.
Apply postprocessing once across all years, then slice predictions by calendar
year for reporting. Report corrected event F0.5, precision/recall and TP/FP/FN
for all-channel and input-channel scopes, seven annual rows plus one pooled row.
The pooled row is directly scored over the full interval, not the mean of annual
scores. Event IDs spanning years may occur in multiple annual denominators but
are counted once in the pooled denominator.

This is a single-seed reused-benchmark extension, not independent validation or
a guaranteed improvement. The historical legacy baseline is context, not a pure
short-roll ablation: its augmentation differs beyond the short branch. A matched
short-coordinated full-period control would be required to attribute a full-period
difference specifically to removing short rolls. No such extra run is silently
added to the user's requested candidate assessment.

Output: `results_longrun/short_no_roll_full_20261014_seed42_v1/`.
Run: `python -m esa_thesis.full_short_no_roll --output results_longrun/short_no_roll_full_20261014_seed42_v1`.
Resume the same command with `--resume`; source/data/settings must remain identical.
Atomic epoch checkpoints preserve optimizer, scheduler and random states. The
implementation tests separate immutable input files, exact interrupted CPU replay,
and pooled event-ID accounting. Existing baseline behavior retains its defaults.
