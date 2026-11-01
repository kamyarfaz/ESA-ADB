#!/bin/bash
set -euo pipefail
cd /home/k_faz/projects/ESA-ADB
export PYTHONPATH=/home/k_faz/projects/ESA-ADB
PYTHON=/home/k_faz/miniforge3/envs/esa-thesis/bin/python
"$PYTHON" -u scripts/research/run_target_boundary_20261101.py
"$PYTHON" -u scripts/research/verify_target_boundary_20261101.py
