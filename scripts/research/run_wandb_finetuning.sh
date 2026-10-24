#!/bin/bash
set -euo pipefail
cd /home/k_faz/projects/ESA-ADB
output=results_longrun/wandb_finetuning_20261024_seed42_v1
if [[ -f "$output/complete.json" ]]; then
  echo 'Completed; independent verification and reporting only.'
  exit 0
fi
args=()
if [[ -f "$output/protocol.json" ]]; then args+=(--resume); fi
exec /home/k_faz/miniforge3/envs/esa-thesis/bin/python -u -m esa_thesis.wandb_finetuning --output "$output" "${args[@]}"
