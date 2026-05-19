#!/usr/bin/env bash
set -euo pipefail

export OWN_CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CODE_ROOT="$(cd "$OWN_CODE_ROOT/.." && pwd)"
export OWN_CODE_OUTPUTS="$OWN_CODE_ROOT/outputs"
export TAPAS_ROOT="$CODE_ROOT/TAPAS"
export COPPELIASIM_ROOT="$CODE_ROOT/sim/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$COPPELIASIM_ROOT"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
export PYTHONPATH="$TAPAS_ROOT:$CODE_ROOT/rlbench:${PYTHONPATH:-}"
if [[ -d "$CODE_ROOT/.venv/bin" ]]; then
  export PATH="$CODE_ROOT/.venv/bin:$PATH"
fi
