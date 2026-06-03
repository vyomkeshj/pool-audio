#!/usr/bin/env bash
# Launch the Pool-Audio research report (Streamlit).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HOME/miniconda3/envs/pool-audio/bin/python"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONNOUSERSITE=1
exec "$PY" -m streamlit run "$HERE/report_app.py" "$@"
