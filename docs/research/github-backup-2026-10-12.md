# GitHub research backup — 12 October 2026

The source update includes current thesis modules, tests, diagnostic and review
scripts, protocols/results, compact tables/plots, finished meeting workbooks,
slides/PDF, and the updated portable AI handoff.98tests passed on12October.

Large artifacts are separate from source Git in release
[research-update-2026-10-12](https://github.com/kamyarfaz/ESA-ADB/releases/tag/research-update-2026-10-12).
The archive `esa-research-artifacts-20261012.tar.gz` is an **additive** backup.
It includes complete recent inner-selection/loss-weight/channel-isolation runs,
short-augmentation campaigns, score-change/direct-telemetry pilots, their listed
diagnostics, and recent presentation/workbook build sources. Included run files
retain optimizer states and all saved epochs. Interrupted inner-run attempts are
preserved too, not treated as accepted experiments.

`BACKUP_MANIFEST.json` inside the archive lists exact file coverage, byte sizes
and SHA256 hashes. A separate manifest and archive SHA256 file are release assets.
Packing verifies every archived file against both the source and archive bytes.

Excluded: datasets, old experiment trees outside the manifest, Conda/Node runtime
installations, temporary renders outside the listed output folders, and `.git`.
Earlier releases remain necessary for older artifacts. This archive is not a
complete mirror of the167GiB working folder. Keep the full external-drive copy.

## Restore

From a checkout of this source revision (Linux; `gh` authenticated if needed):

```bash
mkdir -p /tmp/esa-october-restore
cd /tmp/esa-october-restore
gh release download research-update-2026-10-12 --repo kamyarfaz/ESA-ADB \
  --pattern 'esa-research-artifacts-20261012.tar.gz*'
sha256sum -c esa-research-artifacts-20261012.tar.gz.sha256
# Replace the destination below with your actual fresh project checkout:
tar -xzf esa-research-artifacts-20261012.tar.gz -C /path/to/ESA-ADB
```

Inspect archive paths/manifest before extracting into an existing working copy;
do not overwrite locally changed experiments without comparing them. Historical
protocol paths/source hashes remain provenance; relocation does not automatically
permit interrupted-run resume. Recreate runtimes using README/MIGRATION.md.
Isolated-model weights require ChannelIsolatedAE despite matching joint tensor keys.

A Git commit or any file change invalidates the older full-transfer manifest.
Generate the final transfer manifest after publication; verify on the destination
before deleting the original folder. No destination verification is implied here.
