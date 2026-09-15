# Publication scope

Only this guide and `pre_organization_source_2026-09-15.zip` (the small regression source fixture) are published. Other backups described below exist only in the original local workspace.

# Historical research archive

`legacy_experiments_2026-09-15.zip` preserves 342 source and small result/configuration
files from the former `trash/` folder. It is approximately 4 MB compressed.

The reviewed folder contained 1,998 files totaling 43,606,940,000 bytes (40.61 GiB).
After archive integrity verification, the folder was deleted at the user's request.
Old checkpoints, score arrays, plots, and large prediction CSVs were discarded.
They cannot be recovered from this archive; models would need retraining.

The archive contains:

- `README.md`: scope and interpretation notes.
- `MANIFEST.json`: every former file's path, size, retention status, and retained-file hash.
- `files/`: original relative paths for source and text records below 5 MB.

These experiments are historical references, not maintained executable entry points or
validated thesis results. Their metrics and selection procedures vary. In particular,
the early supervisor-port score of approximately 0.982 accompanied a predicted anomaly
rate of approximately 99.997%, illustrating the custom metric's weakness.

CRC and SHA-256 checks verified all retained files before deletion. No active source
dependency on `trash/` was found. Current datasets and results outside `trash/` were
left in place. The original mission runners in the archive also match Git HEAD.

## Organization snapshot

`pre_organization_source_2026-09-15.zip` preserves root research sources before module extraction (including a SHA-256 manifest). The regression check uses this archive.

`snapshots/main_v1/` is the former `main/` directory, moved intact with its older results. Its scripts and absolute paths are historical, not maintained entry points. `inventories/` holds previous directory listings. Active results remain in `results_longrun/`.

`generated/build_before_organization/` preserves the former root `build/` output. It is generated packaging output, not active source.
