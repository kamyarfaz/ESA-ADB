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

The snapshot is being prepared as split `.tar.zst` assets with a per-file
SHA256 manifest, part checksums, restore instructions and a verification tool.
Publication is complete only when the final verification record is present.
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
