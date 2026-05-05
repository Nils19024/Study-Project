#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

cd "$COPPELIASIM_ROOT"
exec ./coppeliaSim "$SCRIPT_DIR/rlbench/task_design.ttt"
