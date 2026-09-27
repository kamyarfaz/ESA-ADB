# Direct integrity investigation and reproduced cadence audit

The user requested direct investigation rather than a handoff to an administrator.
All checks were read-only with respect to raw/prepared datasets and system state.
Diagnostic scripts, logs and new result directories were created; no data repair,
driver change, reboot or destructive filesystem operation was performed.

## Evidence

- Repeated compressed SHA256 values were stable per channel43/44 across12 probes
  in the Conda interpreter and12 in system Python. Successful decoded hashes also
  agreed. Conda probes had0/12 CRC failures; system Python had2/12 (channel43).
  These probes read bytes into memory before decompression. Therefore the failure
  is not exclusive to the `timeeval` environment or the pandas CSV parser.
- Independent `unzip -t` checks passed for both archives on that attempt.
- An eight-channel guarded audit failed CRC on47. Another failed a repeated hash
  check on decoded in-memory bytes after parsing. No intended modification of those
  bytes occurs. This suggests an intermittent read/computation/memory-path problem;
  the observations do not distinguish the underlying hardware/software cause.
- Readable kernel logs show the already-known NVIDIA driver/library mismatch and
  some process crashes; the filtered log did not establish a storage or memory
  fault. No claim that a specific DIMM, CPU or NVMe drive is defective.
- Filesystem has about82GB available (96% used); RAM approximately56GB available
  at inspection. These facts do not explain a CRC failure by themselves.

## Guarded audit and independent comparison

The audit now snapshots compressed bytes, validates ZIP CRC, hashes compressed
and decoded bytes, checks those hashes again after deserialization, validates
timestamp order, fingerprints the inspected timestamp arrays and saves source
metadata. Failed runs stop without a completed protocol/result set. It never
skips invalid records or silently retries within a run.

Two complete runs (`audit_b` and `audit_cpu2`) produced byte-identical frequency
and training-window tables and identical protocol/source fingerprints. One was
run with CPU affinity restricted to logical CPU2 as a diagnostic comparison;
that success does NOT certify CPU2 or establish another core as defective.
Failed attempts are retained alongside the successful ones, not hidden.

Confirmed bounded-audit tables are in
`tables/cadence-frequency-confirmed-2026-09-25/`, with hashes in `verification.json`.
The original provisional directory remains intact. The initial missing-history
window is now counted separately. This is repeatability evidence, not proof that
the machine is fixed or that all historical results are verified.

## What the audit establishes

For channels41–46, readings older than twice their30s training median cadence
occur at0.1117% of training grid times and0.0591% of calibration grid times.
Channel47 is similar (0.1201%/0.0595%). Thus the cadence issue is rare, but is not
entirely new in calibration. These whole-period statistics include annotated
periods, so they are not nominal-training exposure percentages.

In the8000-window wholly nominal reference sample, two windows per channel41–47
contain readings older than60s. One additional window lacks an initial native
observation and is explicitly separated. The two-window groups have higher
reconstruction errors, but are too small for a robust effect estimate. The
reference sample is not the actual set of all windows used by the optimizer.

## Research decision

We can use the reproduced audit to motivate an observation-age/missingness
hypothesis. We cannot claim it explains the February false alarm or guarantees
an F0.5 improvement. Do not suppress alarms whenever measurements are old,
relabel data or start an unverified long training run. A next preprocessing
comparison should expose timing quality explicitly and preserve channels40–47,
with training-only design and unchanged evaluation. No such model was trained.

The intermittent fault remains unresolved. Future data-processing work needs
integrity checks and repeatability checks, rather than assuming a successful read
means the underlying issue has disappeared. This does not require asking the user
to contact an administrator before every next step.

## Reproduction

`scripts/research/archive_read_probe.py` accepts an output JSON path and tests
channels43/44 repeatedly using only the Python standard library.
`scripts/research/cadence_frequency_review.py --output NEW_DIRECTORY` runs the
guarded audit in the research environment. These scripts were executed, not only
syntax checked. Logs and failed/successful runs are in
`outputs/integrity-review-20260925/`.
