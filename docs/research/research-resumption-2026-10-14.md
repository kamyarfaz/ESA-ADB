# Research resumption and evaluation protocol — 14 October 2026

The user explicitly authorized autonomous research, commands and training on
14 October, superseding the earlier stop. Authorization does not make every
possible training run scientifically justified. The first experiment is the
already specified nominal-quantile diagnostic, with no new training or test tuning.

## Integrity and scope

The comprehensive destination-state archive is verified and published. The old
migration remains non-exact: one historical MLP array is excluded from trusted
research. It is not an input to the six-run diagnostic. All diagnostic dependencies
must match their original migration SHA256 before use. Original artifacts remain
unchanged; results go into a new directory. The historical mismatch does not
invalidate independently verified inputs, but no claim of original recovery follows.

## Fixed evaluation contract

Use `data/frozen/mission1-reference-v1`: train 2000–2005, calibration 2006,
testing 2007–2013, with a separate binary label file and gap-ignore flag.
Reverify before consumption; do not regenerate or edit these files. The historical
reference is a retrospective benchmark: its test years have already been inspected.
Freezing them cannot restore independence. An independent claim would require
additional genuinely unused data or a separately justified external protocol.

All-channel annotation IDs are the primary evaluation scope; input-channel scope
is secondary. Score Anomaly and Rare Event IDs with the corrected duration-penalized
ESA evaluator. Preserve neutral communication-gap handling. Binary label files
support alignment and inspection; pointwise F0.5 does not replace event scoring.

For the immediate matched diagnostic retain eight inputs, channels 40–47, because
changing inputs would confound the threshold-policy comparison. This does not
establish them as the final optimal input set. Any later input change must be
justified using training/development evidence, with one predeclared comparison,
not selected using the 2007–2013 benchmark scores.

## First controlled comparison

Execute `nominal-quantile-plan-2026-10-13.md` exactly: two development years,
three seeds, epoch five in both arms, q=0.999, identical postprocessing, all six
rules frozen before validation-score loading. Reproduce control predictions and
corrected metrics. Reject advancement unless mean system F0.5 improves in both
years without increasing false-event count or nominal false-alarm duration.
No nearby quantiles or automatic full-period training follow a failure.

Primary-source review reconfirmed the 99.9th percentile validation threshold in
Nguyen et al., section IV-B: https://arxiv.org/html/2607.07335v1 . Our explicit
nominal-only restriction is an adaptation; the paper's metric/splits/models do
not establish expected performance here. Benchmark context:
https://arxiv.org/abs/2406.17826 . No conformal or event false-alarm guarantee is
claimed for this empirical quantile on dependent, shifting telemetry.

## Reporting and stopping

Retain every paired seed, both scopes, precision, recall, event F0.5, TP/FP/FN,
nominal false-alarm duration and total occupancy, with mean and sample SD.
A failed gate closes this calibration-only candidate. The baseline stays in place.
Then consolidate representation and input-observability evidence before approving
another training design. Never launch a model sweep merely because access is granted.
