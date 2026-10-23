#!/bin/bash
set -euo pipefail
cd /home/k_faz/projects/ESA-ADB
output=results_longrun/short_no_roll_finetune_20261023_seed42_v1
if [[ -f "$output/complete.json" ]]; then
  echo 'Completed; verification and reporting only.'
  exit 0
fi
args=()
if [[ -f "$output/protocol.json" ]]; then args+=(--resume); fi
exec /home/k_faz/miniforge3/envs/esa-thesis/bin/python -u -m esa_thesis.short_no_roll_finetune --output "$output" "${args[@]}"
