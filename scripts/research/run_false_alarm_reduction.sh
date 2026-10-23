#!/bin/bash
set -euo pipefail
cd /home/k_faz/projects/ESA-ADB
output=results_longrun/false_alarm_reduction_20261023_seed42_v1
phase=diagnose
if [[ -f "$output/research_review.json" ]]; then phase=finish; fi
if [[ -f "$output/complete.json" ]]; then
  echo 'Run already complete; independent verification/delivery only.'
  exit 0
fi
exec /home/k_faz/miniforge3/envs/esa-thesis/bin/python -u -m esa_thesis.false_alarm_reduction --output "$output" --phase "$phase"
