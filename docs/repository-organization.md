# Repository organization

The repository contains active thesis research alongside the original ESA
benchmark framework. Established dataset and output paths remain stable because
cached experiment metadata and external commands can refer to them.

| Location | Responsibility | Maintenance rule |
| --- | --- | --- |
| `esa_thesis/` | Active research implementation | New thesis code belongs here |
| `examples/` | Original benchmark launch examples | Run with repository on PYTHONPATH |
| `timeeval/` | Benchmark library | Preserve upstream structure |
| `timeeval_experiments/` | Benchmark algorithm adapters and generation | Preserve upstream structure |
| `TimeEval-algorithms/` | Algorithm implementations and Docker builds | Keep individual algorithm environments separate |
| `scripts/` | Original benchmark utilities | See its README for categories |
| `notebooks/` | Original exploratory notebooks | Preserve paths; treat stored outputs as historical |
| `tests/` | Benchmark tests and thesis regression checks | Thesis checks live in `tests/thesis/` |
| `docs/` | Research and benchmark documentation | Start with its README |
| `data/` | Source telemetry and preprocessed data | Keep separate from results |
| `results_longrun/` | Active saved research artifacts | Use a new output root for changed configurations |
| `results/` | Earlier saved research outputs | Retain as historical evidence |
| `logs/` | Loose historical console logs | Generated, ignored by Git |
| `wandb/` | Offline tracking records and symlinks | Keep each run intact |
| `archive/` | Source snapshots and earlier experiments | Reference only; do not import into active code |

## Files that intentionally remain at the root

- `README.md`, `LICENSE`, `CITATION.cff`: project entry point and attribution.
- `setup.py`, `setup.cfg`, `MANIFEST.in`, `environment.yml`, `requirements.dev`,
  `requirements.ci`: original framework packaging and environments.
- `requirements-thesis.txt`: observed research environment versions.
- `conftest.py`: pytest configuration shared across benchmark tests.
- `.gitignore`, `.codecov.yml`, `.readthedocs.yaml`: repository/tool configuration.
- `.git/`: version history; `.agents/` and `.codex/`: managed workspace metadata.
- `TimeEval.egg-info/`: local installation metadata, retained to avoid disturbing
  the current environment; ignored by Git.
- `__pycache__/`: generated Python caches, ignored by Git and safe to regenerate.

Framework packaging now explicitly selects `timeeval` and `timeeval_experiments`
packages. It does not accidentally include the thesis package, notebooks, or
archived Python directories. Run thesis commands from the checkout using its
research environment as documented in the root README.

## Changes in this pass

- Grouped the benchmark image and logo under `docs/assets/` and updated references.
- Fixed the Sphinx example inclusion after moving examples.
- Restored original mission runner examples from Git HEAD into `examples/`.
- Moved the generated root `build/` tree to `archive/generated/`.
- Added navigation documents to the data, results, scripts, notebooks, examples,
  logging, and documentation areas.

No datasets, checkpoints, saved scores, notebook outputs, or Git history were
removed in this pass. No training or benchmark rerun was performed. Scientific
metric and validation corrections remain separate work.
