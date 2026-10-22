#!/bin/bash
set -euo pipefail
cd /home/k_faz/projects/ESA-ADB
output=results_longrun/combinations_full_20261021_seed42_v1
resume=()
if [[ -f "$output/protocol.json" ]]; then resume=(--resume); fi
exec /home/k_faz/miniforge3/envs/esa-thesis/bin/python -u -m esa_thesis.full_combinations --output "$output" "${resume[@]}"
