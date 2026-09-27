# Observation-age preprocessing prototype

25 September2026. Implementation and62 thesis tests pass. **No complete verified
sidecar dataset exists yet; no model or predictions were changed.**

## Representation

For each existing telemetry input40–47, record time elapsed since its latest
native observation at the prepared grid timestamp. Encode
`log1p(age_seconds / training_median_native_interval_seconds)`. Cadence is estimated
only from2000–2003 training timestamps (90s for40,30s for41–47). No label or
calibration-selected threshold enters this transformation.

An availability indicator distinguishes the initial grid point with no preceding
measurement from a fresh reading. Age0 at that point is only a placeholder; no
future observation is substituted. The prototype records timing metadata alongside
the same eight physical channels. It does not add other telemetry channels,
change their values, remove rows, suppress alarms, or change evaluation scope.

Functions: `esa_thesis/observation_age.py`. Tests cover exact timestamps, sparse
sampling, initial unavailability, empty history, nonmonotonic inputs and future
observation independence.

## Guarded builder

`scripts/research/prepare_observation_age.py --output NEW_DIRECTORY` reads trusted
local native archives and verifies compressed, decoded and timestamp hashes against
the independently reproduced cadence audit. It checks computed age against a
separate pandas as-of lookup on10000 grid points per channel, round-trips saved
arrays, and rechecks output hashes before writing a completion protocol.

Training interval2000–2003; calibrationJanuary–March2004. No assessment. Optional
`--decoder unzip` uses independent command-line decompression while retaining the
same expected hashes and all downstream checks. It is not a bypass for failed
integrity checks. Existing directories are never overwritten.

## Actual build results

1. Python build stopped on channel42 decoded digest mismatch.
2. Separate CPU2-affinity build stopped on channel41 decoded digest mismatch.
3. Independent `unzip -p ... | sha256sum` for channel41 matched the expected
   decoded hash. This supports testing a distinct decoder, not declaring recovery.
4. The independent-decoder build passed initial decoding checks but stopped when
   a buffer hash changed during parsing of channel42.

All partial directories remain preserved under `results_longrun/diagnostics/`
(`observation_age_2004_v1`, `observation_age_2004_v2`,
`observation_age_2004_unzip_v1`). None has an accepted completion protocol; do not
use their partial arrays for training. Failure logs are in
`tables/observation-age-prototype-2026-09-25/`.

The failure is not confined to Python ZIP decompression. No raw data was altered,
and the checks must not be weakened to obtain a passing build. The underlying
intermittent read/computation problem remains unresolved. The prototype is ready
at code level, but a complete verified data build is required before a small
matched modelling experiment. No new training command is appropriate yet.

## Further isolation and bounded repetition

Controlled SHA256 checks on synthetic bytes, compressed bytes and four repeated
unpickles of channel42 all agreed with independent `sha256sum`. A native C probe
using256MB, eight patterns and three verification passes per pattern found no
mismatch. These limited clean tests do not rule out an intermittent fault.

Fresh per-channel processes initially passed40–42 but failed the expected native
timestamp hash for43. `--isolate-channels` preserves each worker's logs and stops
on failure; this did not resolve the problem.

A separate bounded builder, `scripts/research/build_age_verified.py`, required two
independent workers to produce matching protocol and output hashes per channel,
with at most four attempts per channel and all attempts logged. It made22 attempts,
five exiting with errors. Two otherwise successful channel44 outputs disagreed
at one feature value: grid index4420357 was0.14900924265384674 versus
0.14902326464653015. A third worker matched the second. This shows why a single
worker passing source checks is insufficient; it does not identify the root cause.

All channels eventually had matching pairs, **but the final recheck of those files
failed** (`Independent copies changed`). The top-level completion protocol was
therefore not written. There is still no accepted full sidecar dataset. Do not
combine partial files manually or relax a check to force completion.

Evidence is preserved under `observation_age_2004_isolated_v1` and
`observation_age_2004_verified_v1` in the diagnostics directory, with compact logs
and the disagreement record in the documentation tables. Raw data is unchanged.
No model has been trained on these outputs. The user requested direct investigation;
no administrator handoff or system changes were performed.
