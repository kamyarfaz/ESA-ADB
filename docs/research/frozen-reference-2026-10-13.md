# Frozen historical reference data

The existing reference periods are packaged as separate data files in
`data/frozen/mission1-reference-v1/`. This packages the established experiment;
it does not define a new unseen test set or launch an experiment.

| File | Period, end exclusive | Rows |
|---|---|---:|
| train.csv | 2000-01-01 to 2006-01-01 | 6,312,960 |
| calibration.csv | 2006-01-01 to 2007-01-01 | 1,051,200 |
| testing.csv | 2007-01-01 to 2014-01-01 | 7,364,160 |
| labels.csv.gz | All three periods; split column identifies each subset | 14,728,320 |

All telemetry, telecommand and original annotation columns are retained byte for
byte. No model input channels were selected. The duplicate 2007 endpoint in the
original train source and final 2014 endpoint in the test source are excluded,
matching the established half-open reference protocol. Consecutive timestamps are
checked at 30-second spacing without gaps or overlaps.

The separate labels file was independently checked before packaging. It contains
`timestamp,split,is_anomaly,is_ignored`. A target is positive if any of the 76
telemetry channels has an Anomaly or Rare Event label. Gap-only rows are flagged
for exclusion from normal/pointwise interpretation. Original annotation columns
inside the three data files retain their original 0/1/2/3 encoding. Keep exact raw
annotation IDs and intervals for the corrected event evaluator; binary pointwise
F0.5 is not the thesis evaluation metric.

The source CSVs are checked against their established golden SHA256 digests.
The label file is checked against the previously verified export hash. A separate
manifest records all four output hashes, sizes, column order, intervals and row
counts. Output files and manifest are read-only. The packager refuses to overwrite
an existing version. Failed preparation remains in an `.incomplete` directory;
only a successful preparation publishes the versioned directory.

Run the integrity gate before consuming this package:

```bash
conda activate esa-thesis
cd /home/k_faz/projects/ESA-ADB
python scripts/research/freeze_reference_data.py verify
```

`create` is for initial packaging only; do not recreate or modify this version.
Any deliberate data-policy change requires a new version and a documented reason.
Read-only permissions discourage accidental edits; SHA256 verification detects
content changes. Preserve the manifest digest independently in the project record.

Verified manifest SHA256:
`a49a8c42c23a5d72d91a48596f485d762a805cc6101ad037c01a36c8037bf099`.
All four output files passed the post-write checksum verification on this server.

The current historical training runners still read the original two source CSVs.
They were not silently redirected. A future approved runner must consume these
fixed files and pass verification before fitting or scoring. Select preprocessing
and input channels using permitted training/development information, select the
checkpoint and alarm rules on calibration, then freeze all choices before testing.
The test years were previously inspected and must be reported as reused benchmark
assessment, not independent unseen confirmation.

Five focused tests cover exact boundary assignment, refusal to overwrite,
source-checksum failure, missing timestamps, altered output and wrong label hash.
The complete suite passes 103 tests on the destination research environment.
No scientific model experiment was run. Transfer recovery remains a separate gate:
one historical MLP score array still requires an original source copy.
