#!/bin/bash
set -euo pipefail
cd /home/k_faz/projects/ESA-ADB
exec /home/k_faz/miniforge3/envs/esa-thesis/bin/python -u -m esa_thesis.cross_mission_study pipeline
