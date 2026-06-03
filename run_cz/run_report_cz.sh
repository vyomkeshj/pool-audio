#!/usr/bin/env bash
# Spuštění české výzkumné zprávy (Streamlit). Sdílí data/modely z ../run.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HOME/miniconda3/envs/pool-audio/bin/python"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONNOUSERSITE=1
exec "$PY" -m streamlit run "$HERE/report_app.py" "$@"
