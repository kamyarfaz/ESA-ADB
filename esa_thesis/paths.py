"""Project paths, independent of the process working directory."""
import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("ESA_ADB_ROOT", Path(__file__).resolve().parents[1])).resolve()

def project_path(relative):
    return PROJECT_ROOT / relative

def default_run_dir():
    root = project_path("results_longrun/mission1_reconstruction_ae_sweep_optimized")
    pointer = root / "latest_run.txt"
    if pointer.exists():
        value = Path(pointer.read_text().strip())
        local = root / value.name
        if local.is_dir():
            return local
        if value.is_absolute() and value.is_dir():
            return value
    return root / "20260519_155620"
