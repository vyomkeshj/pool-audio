#!/usr/bin/env bash
# Launch the pool-pump diagnostics dashboard in the pool-audio conda env.
# First-time setup:  conda env create -f environment.yml
#
# NOTE: this machine has an ESP-IDF python env on PATH and a stale streamlit in
# ~/.local, both of which hijack `conda run`. So we call the env's python
# DIRECTLY and set PYTHONNOUSERSITE=1 to ignore ~/.local. Override the env
# location with POOL_AUDIO_ENV=/path/to/env if needed.
set -e
cd "$(dirname "$0")"

ENV="${POOL_AUDIO_ENV:-$(conda env list 2>/dev/null | awk '$1=="pool-audio"{print $NF}' | head -1)}"
[ -z "$ENV" ] && ENV="$HOME/miniconda3/envs/pool-audio"
PY="$ENV/bin/python"
[ -x "$PY" ] || { echo "pool-audio env not found at $ENV — run: conda env create -f environment.yml"; exit 1; }

echo "Starting dashboard at http://localhost:8501  (Ctrl-C to stop)"
exec env PYTHONNOUSERSITE=1 "$PY" -m streamlit run app.py "$@"
