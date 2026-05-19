#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"
COPPELIA_DIR="$PROJECT_ROOT/sim/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"

cd "$PROJECT_ROOT"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python 3.10 was not found as '$PYTHON_BIN'."
  echo "Install Python 3.10 or run with PYTHON_BIN=/path/to/python3.10 ./setup_env.sh"
  exit 1
fi

if [[ -f "$PROJECT_ROOT/.gitmodules" ]]; then
  git submodule update --init --recursive
elif [[ ! -d "$PROJECT_ROOT/riepybdlib/.git" ]]; then
  git clone https://github.com/vonHartz/riepybdlib.git "$PROJECT_ROOT/riepybdlib"
fi

if [[ ! -d "$COPPELIA_DIR" ]]; then
  echo "CoppeliaSim was not found at:"
  echo "  $COPPELIA_DIR"
  echo "Please add/install CoppeliaSim there before installing PyRep."
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

export COPPELIASIM_ROOT="$COPPELIA_DIR"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIA_DIR"
export LD_LIBRARY_PATH="$COPPELIA_DIR:${LD_LIBRARY_PATH:-}"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$PROJECT_ROOT/requirements.txt"
python -m pip install --no-build-isolation --force-reinstall --no-deps \
  "git+https://github.com/markusgrotz/PyRep.git#egg=pyrep"

echo "Environment setup complete."
