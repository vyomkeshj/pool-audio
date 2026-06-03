#!/usr/bin/env bash
# Launch the Pool-Audio desktop app (player + independent listener + spectrogram).
# Uses the conda env's python directly and ignores user-site/ESP-IDF PATH shims.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HOME/miniconda3/envs/pool-audio/bin/python"
# 1 BLAS thread keeps the live feature extraction snappy without oversubscribing
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONNOUSERSITE=1
exec "$PY" "$HERE/app_desktop.py" "$@"
