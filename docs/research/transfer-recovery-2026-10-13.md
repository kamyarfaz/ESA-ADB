# Destination recovery and verification — 13 October 2026

The full original transfer manifest was checked again on this server after
recovery: **12,517 entries checked; 12,501 match; 16 differ**. The log is
`migration/recovery-20261013/full-verification.log`. This is not a zero-failure
transfer and does not authorize deleting the source copy.

## Recovered

- `.gitattributes` and `docs/research/github-backup-2026-10-12.md` restored from the
  current committed HEAD without changing the Git index.
- Eight missing symlinks restored with the exact recorded targets. External W&B
  log targets recovered from the preserved `migration/external-logs/` copy.
- The archived `fine_ch08_35_42/test_score_per_channel.npy` restored from a local
  duplicate whose SHA256 exactly matches the original manifest. Its damaged
  predecessor remains in `migration/recovery-20261013/`.
- Executable bits corrected on 2,216 tracked files only where contents exactly
  match their Git index blobs. Two content-modified files were left untouched.
- Git object connectivity check passed. All recovery paths, before/after hashes
  and permission changes are recorded under `migration/recovery-20261013/`.

## Remaining differences

Fifteen entries are documentation, Git state or Git-status snapshots. The original
manifest records an earlier HEAD (`e473c858...`); the current HEAD is its descendant
`0fd67c8...`. Current README and .gitignore exactly match committed HEAD. The handoff
contains subsequent verified work. Preserve this newer state rather than resetting
Git metadata or replacing current documentation to force an old manifest to pass.
The old Git-status snapshot is not available in its original form.

One research array still differs:

`results_longrun/mission1_reconstruction_ae_sweep_optimized/20260519_155620/pa_center_ch08_40_47/mlp_test_score_per_channel.npy`

- Expected SHA256: `280a2c4ed680b776297bf8124e5f794204caadb51b705e5fbb74cffdb0a42652`
- Current SHA256: `edc0a6612e099ae9ed4175a76101f8c645f7c9c8a01cc7e17ebf34a4c0e65ba8`
- Size: 235,653,280 bytes (same size, different contents).

The September 17 essential archive was downloaded and verified against published
archive SHA256 `4cffcc4f8b5ac173cd642672fa52d296e91147bf158a927a9bca1aedee8f454c`.
The target occurs in its EXCLUDED manifest list and is absent from the archive.
September 21/22 and October 12 release manifests and the saved September 27
manifest also provide no recoverable copy. GitHub source is not a backup of this
array. The current file is preserved unchanged. Do not use it as verified evidence
or regenerate it and call that exact recovery. An independent original/external-drive
copy is still needed. The user supplied the current checkout and GitHub URL;
neither is an independent original for resolving this discrepancy.

## Verified data preparation

All other recorded research/data artifacts match the original transfer hashes in
this check. The established source CSV checks passed independently. The separate
frozen reference package was built only from those verified CSVs and the independently
verified binary-label export; it does not use the unresolved MLP array.
See [frozen reference data](frozen-reference-2026-10-13.md). Four output files passed
post-write SHA256 checks. The complete thesis suite, including five new packaging
integrity tests, passes **103 tests** on the destination environment.

No research experiment was launched. Keep the original manifest as evidence.
Do not replace it with current hashes to erase a discrepancy. Regenerate a transfer
manifest only after reconciliation and immediately before a subsequent copy, per
`MIGRATION.md`.
