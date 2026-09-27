# GitHub backup — 27 September 2026

Source update includes current research modules, diagnostic scripts, tests,
reports, compact result tables, plots, and AI handoff.73thesis tests pass.
Short coordinated augmentation is implemented/tested but not wired into a
training CLI and has not been trained. Baseline systemF0.5 remains0.80572.

Essential artifacts are packaged separately in the release
[research-update-2026-09-27](https://github.com/kamyarfaz/ESA-ADB/releases/tag/research-update-2026-09-27).
Archive `esa-research-artifacts-20260927.tar.gz` contains213files (~977MB unpacked):
recent selected checkpoints, protocols/results and assessment scores/predictions,
plus the accepted full-period observation-age sidecars. Each file was hashed
before/after packing and verified by reading it back from the archive.
`BACKUP_MANIFEST.json` inside lists all paths/sizes/SHA256 values; an external
`.sha256` file verifies the archive before extracting into a project checkout.
Existing release backups remain necessary for older artifacts. See the manifest
for exact coverage; this is not a full working-directory mirror.

Excluded: raw/prepared dataset (download/setup perREADME), most large diagnostic
caches, duplicate failed timing workers, and optimizer resume states. Retain the
local project/external-drive copy for those. Restored selected models can be
scored; omitted optimizer states do not support exact interrupted-training resume.
Historical source-hash protocols may require their original source revision;
never silently rewrite them to allow a restart. No migration manifest regenerated
because this is a GitHub publication, not a full filesystem transfer.
