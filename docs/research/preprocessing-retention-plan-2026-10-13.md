# Training-only reconstruction of the existing eight inputs

The calibration influence audit rules out treating repeated checkpoint/threshold
search as sufficient evidence of progress. Before changing representation, measure
another already documented limitation: annotation-dependent restoration of short
raw events during 30-second preprocessing.

This diagnostic is declared before computing its results. Use channels 40–47,
the existing inputs, and exactly [2000-01-01, 2003-01-01), the first outer fold's
training interval. The raw archives must be unpickled in full, then immediately
restricted to this interval. Do not summarize later raw observations or read
benchmark testing.csv. No channel choice follows from this diagnostic.

For every prepared training timestamp, reconstruct (A) zero-order hold with the
historical initial-value convention, and (B) that same hold plus the historical
annotation-restoration rule. Compare both against the verified prepared source
using rtol=1e-6, atol=1e-12, to accommodate raw float32/CSV decimal differences.
Check that the full legacy reconstruction agrees before attributing differences
to restoration. Preserve any unexplained mismatch as a failed reconstruction,
not as a newly discovered anomaly. Keep output to summaries and changed rows.

Record restoration locations, numeric/label changes, affected event IDs and
per-channel/per-year counts. A changed value is not a detected event; do not
derive F0.5 improvement from these counts. This comparison isolates restoration
and does not claim a causal end-to-end pipeline: initial backfill, offline
window scoring and postprocessing remain distinct issues.

The benchmark paper describes the restoration step explicitly in its
preprocessing section (pp.25–26):
https://arxiv.org/pdf/2406.17826 . The local implementation is
`notebooks/data-prep/Mission1_semisupervised_prep_from_raw.py:130`.
No historical CSVs, frozen reference files, labels or models are changed.
