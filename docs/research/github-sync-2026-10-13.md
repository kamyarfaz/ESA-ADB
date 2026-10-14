# Project synchronization — 13 October 2026

The user requested updating both GitHub and the local project. The source branch
was fetched and checked before changes: local `main` and `origin/main` both
pointed to `0fd67c8b08d33286f2b383a5f8ddbb2150d39106`. There were no incoming
source commits to merge. Source synchronization is separate from Release assets.

## Source update

This update preserves current code, tests, reports, compact diagnostic tables,
neutral presentation/workbooks, and reproducible build sources. Existing local
documentation edits are retained. Office locks and superseded generated reports
stay out of source Git. Research is stopped; a saved research plan is not an
instruction to run it. The complete thesis suite, including three new snapshot safety tests, passed
117 tests in 10.746 seconds before publication.

## Comprehensive project-file backup

Target release:
[project-snapshot-2026-10-13](https://github.com/kamyarfaz/ESA-ADB/releases/tag/project-snapshot-2026-10-13).

Completed **14 October 2026**. The published Release contains 50 assets: 44
archive parts plus the manifest, checksums, restore instructions, verifier and
verification records. All uploaded asset sizes and GitHub SHA256 digests match
the local files, including the remote verification record itself.

The archive independently passed verification of all 10,526 entries: 9,477
regular files totaling 186,891,293,981 bytes (174.06 GiB), plus directories and
symlinks. The compressed archive is 46,685,754,202 bytes (43.48 GiB).
Archive SHA256: `b076811626786e4af490249d9755dc7561a78bdccba5e50eff60103e8cc1950c`.
The verification record is [saved here](github-snapshot-verification-2026-10-14.json).
Remote verification uses GitHub's asset digests; a complete remote archive was
not downloaded again. Every local archived file was independently rehashed.

The snapshot records source commit `751fc1a`; completion documentation and this
verification record were committed subsequently and are available in source Git.
The dated snapshot manifest describes its immutable archive, not those later
source documentation changes. Regenerate a transfer manifest before any new
whole-directory server copy, per MIGRATION.md.

All 3,667 entries in older Release manifests exist locally. Three older versions
were preserved separately under `archive/snapshots/github-release-versions/`
without replacing the newer local files. The 19 files in supplemental September
archives also match their local copies.
It includes datasets, frozen splits, labels, saved arrays/weights, historical
archives, output/build files and migration evidence, including ignored files.

Excluded operational material is listed in the manifest: `.git` (source history
is provided by GitHub), installed `node_modules`, Python bytecode/cache folders,
and temporary Office locks. External symlink targets are not silently copied;
links are preserved and reported. Recreate environments as documented in README.
GitHub credentials live outside this project and are not part of this backup.

The original migration manifest is preserved. This new snapshot fingerprints the
current destination state; it does **not** claim the old migration became exact.
The known historical MLP array remains unverified against its original hash and
is backed up only as its current preserved bytes. See
[transfer recovery](transfer-recovery-2026-10-13.md).

Restore into a new directory, verify every archived regular file and symlink,
then compare against an existing checkout before merging. No local files or old
release assets are deleted by this synchronization. Keep an independent full
external-drive copy as well.
