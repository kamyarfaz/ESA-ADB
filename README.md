# Anomaly Detection in Satellite Telemetry

Thesis research by **Kamyar Faz**, exploring a
Transformer autoencoder and an MLP forecaster on ESA Mission 1 telemetry.
The objective is to improve anomaly detection under space operational constraints,
with a target **F0.5 ≥ 0.85**. That target is a research objective, **not a verified
result of this repository**.

This repository extends the [ESA Anomaly Detection Benchmark](https://github.com/kplabs-pl/ESA-ADB).
`esa_thesis/` contains the active research code; the original benchmark framework
is retained for preprocessing, reference algorithms, and evaluation research.
`main` contains the organized project. [`previous_experiments`](https://github.com/kamyarfaz/ESA-ADB/tree/previous_experiments)
preserves the earlier layout.

## 1. Clone and check the software

The documented target is **Linux with Conda/Miniconda and Python 3.9**.
The thesis pipeline uses PyTorch directly; Docker is only needed for the separate
original benchmark algorithms. CPU works for startup and regression checks;
full experiments benefit from an NVIDIA GPU and substantial RAM/storage.
The author's training hardware is an RTX 3060, i9-10980XE, and about 64 GB RAM;
this is context, not a tested minimum specification.

```bash
git clone --depth 1 --branch main https://github.com/kamyarfaz/ESA-ADB.git
cd ESA-ADB
conda create -n esa-thesis python=3.9 pip -y
conda activate esa-thesis
python -m pip install --upgrade pip

# CPU installation, suitable for software/regression checks:
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-test.txt

python -m esa_thesis --help
python -m esa_thesis doctor --skip-data
PYTHONPATH=. WANDB_MODE=disabled python tests/thesis/check_refactor.py
```

For an NVIDIA training machine, replace the CPU installation command with:

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

Choose one build when setting up the environment. An existing CPU install can be
replaced by adding `--force-reinstall` to the CUDA command. These build choices
follow the [PyTorch 2.6 installation instructions](https://pytorch.org/get-started/previous-versions/#v260).
The NVIDIA driver must support the chosen build. Check detection with
`python -m esa_thesis doctor --skip-data`; the code otherwise selects CPU automatically.

Run commands from the repository root. There is no thesis `pip install -e .` step:
the root `setup.py` packages the separate, older TimeEval framework.
The version file records the research environment used for local checks; it is
not a complete lockfile for all transitive dependencies or operating systems.

### Restore saved research on another computer

The code setup above is sufficient for a fresh start. To recover saved checkpoints
and recent experiment results, also download the attachments from the
[essential migration backup release](https://github.com/kamyarfaz/ESA-ADB/releases/tag/migration-backup-2026-09-17).
**Cloning the repository does not download release attachments.**

1. Download `esa-thesis-essential-20260917.tar.gz`, `manifest.json`, `RESTORE.md`,
   and `SHA256SUMS` into one separate download directory.
2. In that directory, run `sha256sum -c SHA256SUMS`.
3. Extract into your **newly cloned** project (replace both example paths):
   `tar -xzf /path/to/esa-thesis-essential-20260917.tar.gz -C /path/to/ESA-ADB`.
4. From the project root, run `python migration-backup/verify.py .`.
5. Download and prepare Mission 1 using Section 2 below, then run the data checks.

The 17 September base snapshot contains 1,583 files in approximately 1.28 GiB.
After restoring and verifying it, also download and apply the
[21 September completed-experiment update](https://github.com/kamyarfaz/ESA-ADB/releases/tag/migration-update-2026-09-21)
(about 40 MB). It contains the completed expanded-channel folds and recovery
checkpoint. Read that release's `RESTORE.md`: download each release to a separate
folder, verify the base before applying the update, then run
`python migration-update-20260921/verify.py .` from the restored project.
The update intentionally replaces older snapshot files, so their old base-manifest
hashes will differ afterward. **Both releases are needed for the full essential backup.**

Datasets, damaged CSV copies, and older score arrays outside the recent
development/scoring-comparison/audit directories remain excluded. Saved JSON files
may contain old absolute paths as provenance; use the new repository paths when
launching commands.

## 2. Data preparation

**Telemetry, preprocessed CSVs, trained weights, cached scores, and experiment
outputs are not distributed in this branch.** A fresh clone can run the software
checks above, but training requires the data below.

### Download the source dataset

Use the original **version 1.0** [ESA Anomaly Dataset record](https://zenodo.org/records/12528696)
for the preprocessing described here. Download `ESA-Mission1.zip` (about 3.8 GB).
The record also offers Missions 2 and 3; they are not needed for the thesis default.
A newer dataset version exists, so record any deliberate version change as a new experiment.

From the repository root:

```bash
mkdir -p data
curl --fail --location --continue-at - \
  --output data/ESA-Mission1.zip \
  'https://zenodo.org/records/12528696/files/ESA-Mission1.zip?download=1'
printf '%s\n' '80750189d171f5f398fb3d96c49df12b  data/ESA-Mission1.zip' | md5sum -c -
unzip data/ESA-Mission1.zip -d data
```

Verify that extraction produces this structure (move an extra enclosing archive
folder if necessary):

```text
data/ESA-Mission1/
  channels/             channel_*.zip telemetry files
  telecommands/         telecommand series
  labels.csv
  anomaly_types.csv
  telecommands.csv
```

The archive is only the compressed source. Preprocessing and training create much
larger files; reserve tens of GB beyond the download, with additional space for
multiple runs. The two 84-month prepared CSVs occupy about **6.9 GiB each**
(**13.8 GiB combined**); other generated splits require additional space.
The source dataset can be downloaded again and is intentionally excluded from
the migration backup. Do not place these artifacts in Git.

### Generate benchmark CSVs in a separate environment

The inherited preprocessing script imports the older TimeEval dependency stack.
Keep it separate from the research environment to avoid NumPy/statsmodels conflicts:

```bash
# Run from the repository root; this environment is named timeeval.
conda env create -f environment.yml
conda activate timeeval
PYTHONPATH=. python notebooks/data-prep/Mission1_semisupervised_prep_from_raw.py data/ESA-Mission1
conda activate esa-thesis
```

If `timeeval` already exists and has been modified, create a fresh named environment
with `conda env create -n esa-preprocess -f environment.yml` and activate that name.
The script prepares several training durations, not just 84 months, and may take
hours. It resamples telemetry at 30 seconds, transforms designated monotonic
channels, encodes telecommands, assigns channel labels, and writes metadata.
Its output location is relative to the repository, independently of `ESA_ADB_ROOT`.

The thesis expects:

```text
data/preprocessed/multivariate/ESA-Mission1-semi-supervised/
  84_months.train.csv
  84_months.test.csv
```

Input columns include `channel_N` features and **per-channel `is_anomaly_*` labels**.
A single global `is_anomaly` column is not a substitute. The inspected local files
have 175 columns including timestamps, features, and labels. The preprocessor
places training samples at/before 2007-01-01 and test samples after that boundary.

```bash
python -m esa_thesis doctor
python -m esa_thesis train --dry-run
```

`doctor` checks imports, GPU availability, expected paths, feature columns, and
label headers. It does not validate every row or certify the scientific evaluation.
`--dry-run` prints the configuration without training or creating results.

For complete structural validation of the prepared inputs (this reads every row):

```bash
python scripts/research/validate_mission1_csv.py \
  data/preprocessed/multivariate/ESA-Mission1-semi-supervised/84_months.train.csv \
  --split train --report data/repairs/train_validation.json
python scripts/research/validate_mission1_csv.py \
  data/preprocessed/multivariate/ESA-Mission1-semi-supervised/84_months.test.csv \
  --split test --report data/repairs/test_validation.json
```

These checks validate numeric fields, labels, timestamps, expected extent, and
record SHA256 hashes. They do not prove every value matches raw telemetry.
Historical repair scripts in the backup are records of the old machine's repair;
do not apply them automatically to newly generated data.

## 3. Launch an experiment

Start with one channel group and a new output directory:

```bash
WANDB_MODE=disabled python -m esa_thesis train \
  --runs fed_ch06_41_46 \
  --output-root results_longrun/first_experiment
```

Despite its historical name, the exact channel list is defined in
[`config.py`](esa_thesis/config.py). Inspect `RUN_SPECS` rather than inferring
channels from a run name.

- `python -m esa_thesis train` runs/resumes **all 31 configured experiments**.
- `--runs NAME ...` selects experiments; each may train both AE and MLP.
- `--output-root PATH` selects the sweep directory. Relative paths are resolved
  against the project root. A timestamped run directory is created inside it.
- Hyperparameters, channel groups, and `TRAIN_MLP` are in `esa_thesis/config.py`.
- Resume uses `latest_run.txt`, checkpoints, and cached scores. **Use a new output
  root after changing configuration**: incompatible/unversioned caches are rejected.
  Use a separate root for a subset run to avoid replacing an existing sweep summary.
- W&B defaults to offline logging. `WANDB_MODE=disabled` disables it; a W&B account
  is not required for local training.
- `ESA_ADB_ROOT=/absolute/path` changes the root used for research data and default
  output paths. External copies of old absolute `latest_run.txt` paths may need
  correction before resuming.

Outputs include model checkpoints, training histories, scalers, validation/test
score arrays, selected thresholds, JSON/CSV summaries, and plots. They stay local
under the chosen output root and are ignored by Git.

## 4. Technical overview

### Learning and scoring pipeline

1. **Load a channel subset.** Parse the prepared CSVs, fill missing feature values,
   and aggregate all positive `is_anomaly_*` labels into a binary event timeline.
2. **Split and scale.** Reserve the last three calendar months of the provided training series for
   validation. Fit per-channel median/IQR scaling on normal training points and
   clip normalized values to ±10.
3. **Construct windows.** AE input has shape `(batch, channels, 256, 1)`. Training
   selects normal windows with stride 16, capped at 250,000. Current code falls
   back to all windows if no normal windows exist; this needs care on new datasets.
4. **Train the Transformer AE.** Split each channel into 16-sample patches, project
   to 128 dimensions, and add patch-position and channel embeddings. Temporal and
   channel attention encode the representation; corresponding decoders reconstruct
   the input. The default uses 8 attention heads and 40 training epochs.
5. **Train the MLP forecaster.** A separate feed-forward model predicts the next
   32 samples from a 256-sample context. Absolute forecast error provides a
   complementary anomaly score. Selected experiments also inject pseudo-anomalies
   during AE training; this is an experimental variant.
6. **Score and aggregate.** AE reconstruction errors are averaged over each window
   and accumulated over overlapping positions; channel scores are reduced by max.
   MLP errors are assigned to forecast positions. The ensemble takes the maximum
   of separately min-max-normalized AE and MLP scores.
7. **Choose thresholds and postprocess.** Sweep thresholds, merge short gaps, remove
   short detections, and export event/point metrics and plots. Separate tools explore
   reselection, channel subsets, coverage ensembles, EVT, and DSPOT.

### Evaluation status and operational limitations

The active pipeline now computes corrected event-wise F0.5 with raw annotation IDs,
timestamps, and a multiplicative nominal-time false-alarm penalty. The fast evaluator
is checked against the pinned upstream `ESAScores` implementation. Validation alone
selects checkpoints and thresholds, and training uses the last three calendar months
for validation. See [implementation and baseline notes](docs/research/corrected-evaluation.md).

Historical scores used a different formula and must not be compared as if equivalent.
Historical evaluation commands require `--allow-legacy-metric`; use `recalibrate`
for a corrected diagnostic evaluation of saved scores. Legacy caches cannot be
resumed by new training. The default output root is `results_longrun/mission1_esa_ew_v1`.

The 0.85 target remains unverified. Full-series ensemble normalization, retrospective
reconstruction, future-based filling, and historical test exposure remain research
limitations. Point-label diagnostics still reflect the prepared binary labels;
annotation-ID event metrics are authoritative for the corrected event score.
Model architecture and score generation have not changed in this evaluation phase.

## 5. Code map

| Location | Purpose |
| --- | --- |
| `esa_thesis/config.py` | Hyperparameters, paths, and experiment definitions |
| `esa_thesis/data.py` | CSV loading, robust scaling, window datasets |
| `esa_thesis/models.py` | Transformer AE and MLP architecture |
| `esa_thesis/training.py` | Training and resume orchestration |
| `esa_thesis/scoring.py` | Reconstruction, forecast, and ensemble scores |
| `esa_thesis/metrics.py`, `thresholds.py` | Existing metric and threshold logic |
| `esa_thesis/checkpoints.py`, `plots.py`, `tracking.py` | Persistence and reporting |
| `esa_thesis/analysis/` | Correlations, importance, representative selection |
| `esa_thesis/evaluation/`, `calibration/` | Post-hoc evaluation experiments |
| `tests/thesis/` | Numerical refactor regression check |
| `docs/` | Research findings, migration map, benchmark documentation |
| `timeeval/`, `timeeval_experiments/`, `TimeEval-algorithms/` | Original benchmark stack |
| `notebooks/data-prep/`, `scripts/`, `examples/` | Benchmark preparation and utilities |
| `archive/` | Provenance notes and small regression source fixture |
| `data/`, `results/`, `results_longrun/`, `logs/`, `wandb/` | Local data/output areas; README guides only in Git |

For contributors and AI assistants: begin with this README and
[`docs/thesis-research-context.md`](docs/thesis-research-context.md). Active thesis
changes belong in `esa_thesis/`. Keep scientific corrections explicit and separately
validated; do not treat historical score files or archived code as authoritative.
The [migration map](docs/source-migration.json) connects old filenames to new modules.

## 6. Other commands

Run `python -m esa_thesis COMMAND --help` for arguments.

| Commands | Purpose |
| --- | --- |
| `doctor` | Software and dataset-header checks |
| `forecast-develop` | Causal Transformer, MLP, and persistence comparison; [instructions](docs/research/forecast-development.md) |
| `develop` | Chronological Transformer baseline; [instructions](docs/research/chronological-development.md) |
| `compare-scoring` | Same-weights window/per-timestep comparison; [instructions](docs/research/scoring-comparison.md) |
| `recalibrate` | Validation-only corrected evaluation of cached AE/MLP scores |
| `correlation`, `importance`, `representatives` | Channel analysis |
| `score`, `evaluate` | Earlier scoring utilities |
| `evaluate-ensemble`, `reselect` | AE/MLP ensembles and threshold reselection |
| `subset`, `subset-v1`, `coverage` | Channel-subset and coverage experiments |
| `evt`, `dspot` | Experimental threshold calibration |

Analysis commands often require cached artifacts from a completed run and may write
new summaries. Their defaults refer to the historical sweep; supply the appropriate
run path shown by `--help` for your own experiments. They are not fresh-clone demos.

## 7. Verification and troubleshooting

The refactor check compares 11 intentionally unchanged definitions with the source fixture,
loads identical model state dictionaries, compares CPU model outputs, and checks
legacy-metric/postprocessing parity. Separate tests check the corrected metric. No dataset is required. It preserves the unchanged model/scoring behavior. Run the corrected metric and
selection checks with:

```bash
PYTHONPATH=. python -m unittest discover -s tests/thesis -p 'test_*.py' -v
PYTHONPATH=. WANDB_MODE=disabled python tests/thesis/smoke_training.py
```

| Symptom | Action |
| --- | --- |
| `No module named esa_thesis` | Run from the cloned repository root |
| Missing train/test CSV | Complete Data preparation, then run `doctor` |
| NumPy `MachAr` or statsmodels error in preprocessing | Use the separate, fresh `environment.yml` environment |
| CUDA unavailable | Check `nvidia-smi` and the PyTorch build; CPU fallback is automatic |
| GPU out of memory | Reduce training/scoring batch sizes in `config.py`; use a new output root |
| Old results reused after an edit | Use a new output root; incompatible caches are rejected |
| No results for an analysis command | Train first and select your run directory with that tool's arguments |

Validation scope: the documented requirements were installed in a fresh Linux
Python 3.9 CPU environment. Dependency consistency, all module imports, 13 command
help pages, dry-run/error handling, and numerical regression passed against the
publication files without local datasets. Dataset headers were also checked in
the original workspace.
Full raw-data preprocessing, a fresh installation on every platform, and GPU
training were not rerun for this organization release.

## Attribution and license

Preserve the original [LICENSE](LICENSE) and [CITATION.cff](CITATION.cff).
See [CONTRIBUTIONS.md](CONTRIBUTIONS.md) for thesis provenance and
[original benchmark documentation](docs/benchmark-original.md) for ESA-ADB and
TimeEval citations. The dataset has its own source record and terms.

## Copy this complete working folder to another server

For external-drive transfer including local datasets and all experiment outputs,
follow [MIGRATION.md](MIGRATION.md). It includes SHA256 verification and environment setup.
