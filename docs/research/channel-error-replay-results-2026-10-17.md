# Frozen channel-error replay: incomplete at numerical verification gate

**Status: five of six models verified; the full diagnostic is not complete.**
Do not interpret the five-model subset as a successful six-model comparison or
select a scoring method from it. No detector, threshold or historical result changed.

The diagnostic completed15 of18 stages: training error summaries and exact
calibration/validation prediction replay for2001 seeds42/43/44 and2002 seeds42/43.
All ten calibration/validation masks for those five models passed exact replay.
The full thesis suite passed124 tests; focused annotation-interval tests passed.
Dependency and input hashes passed. Those checks do not override a failed real-data
replay gate on the sixth model.

## Why execution stopped

The final model,2002/seed44, failed calibration replay on the current execution
paths. GPU scores exceeded the fixed numerical tolerance at128/521280 samples.
CPU scores passed that tolerance, but32/521280 binary predictions differed when
the original threshold and postprocessing were applied. The script stopped before
producing its event diagnostics. The threshold was neither moved nor rounded.

An earlier2001/seed42 GPU mismatch was resolved by CPU replay. For that first
case,32 values equal to the saved threshold increased by one float32 ULP and
created another predicted alarm interval; corrected F0.5 stayed unchanged.
That is evidence that matching only the final metric would conceal a replay
difference. It does not establish the exact cause of the final model's mismatch.
See `channel-error-replay-protocol-2026-10-17.md` for the preserved investigations
and unchanged acceptance gates.

## What the partial output can and cannot establish

Completed channel summaries show heterogeneous behavior: some event intervals
have rare channel-specific error peaks, while others have ordinary or reduced
errors. These are provisional descriptive observations from an incomplete cohort,
not evidence that replacing maximum aggregation will improve detection. No new
system F0.5 or adopted scoring rule follows from this run.

The next requirement is numerical reproducibility of the remaining frozen
checkpoint, ideally with the original inference environment/hardware or a separately
documented numerical-equivalence study. Do not relax the acceptance tolerance,
shift a threshold, omit the seed or label five models as six to finish the table.
The current data does not yet answer the full aggregation-versus-representation
question. New neural training is not justified by this incomplete diagnostic.

## Preserved evidence

Compact evidence is in `tables/channel-error-replay-2026-10-17/`:
`status.json`, protocol, failure log, first numerical investigation, and explicitly
named `partial_events.csv` / `partial_monthly.csv`.
All15 stage outputs were rehashed before copying the partial summaries.
Arrays and per-stage manifests remain in
`results_longrun/diagnostics/channel_error_replay_20261017_v3/`.
Failedv1/v2 directories and logs remain intact. V3 has `stopped.json`, not a
successful `complete.json`. There is no running replay process to resume blindly.
