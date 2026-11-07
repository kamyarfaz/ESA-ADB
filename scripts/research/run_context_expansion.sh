#!/bin/bash
set -euo pipefail
cd /home/k_faz/projects/ESA-ADB
output=results_longrun/context_expansion_20261104_v1
if [[ -f "$output/complete.json" ]]; then
  echo 'Computation already complete; independent verification/delivery must use the saved artifacts.'
  exit 0
fi
resume=()
[[ -f "$output/protocol.json" ]] && resume=(--resume)
exec /home/k_faz/miniforge3/envs/esa-thesis/bin/python -u -m esa_thesis.context_expansion --output "$output" "${resume[@]}"
