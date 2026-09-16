# Local datasets

- `ESA-Mission1/`, `ESA-Mission2/`: source mission telemetry, telecommands, and annotations.
- `preprocessed/`: benchmark metadata and generated model input CSVs.

The thesis currently uses Mission 1's 84-month split under
`preprocessed/multivariate/ESA-Mission1-semi-supervised/`.
Source datasets are retained at their original locations. Do not mix experiment
outputs into these directories or overwrite source annotations.
Dataset preprocessing tools are indexed in `../scripts/README.md`.

`repairs/` contains local integrity reports and preserved damaged-file backups.
These artifacts, like the datasets themselves, are excluded from Git.
