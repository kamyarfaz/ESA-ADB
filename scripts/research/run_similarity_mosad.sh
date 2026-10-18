#!/bin/bash
set -euo pipefail
cd /home/k_faz/projects/ESA-ADB
output=results_longrun/similarity_mosad_full_20261018_seed42_v1
resume=()
if [[ -f "$output/protocol.json" ]]; then resume=(--resume); fi
exec /home/k_faz/miniforge3/envs/esa-thesis/bin/python -u -m esa_thesis.full_similarity_mosad --output "$output" "${resume[@]}"
