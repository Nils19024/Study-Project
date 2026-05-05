#!/usr/bin/env bash
set -euo pipefail

export RL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export COPPELIASIM_ROOT="/home/nils/Documents/Study Project/Code/sim/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$COPPELIASIM_ROOT"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
export PYTHONPATH="$RL_ROOT:${PYTHONPATH:-}"

source "$RL_ROOT/.venv/bin/activate"
