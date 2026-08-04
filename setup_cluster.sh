#!/usr/bin/env bash

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SIM_DIR="$ROOT/sim/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
mkdir -p "$ROOT/own_code/artifacts/datasets"

if ! command -v conda >/dev/null 2>&1; then
    echo "Conda was not found. Load the cluster's Conda/Miniconda module first."
    exit 1
fi

eval "$(conda shell.bash hook)"

if [[ ! -d "$ROOT/riepybdlib/.git" ]]; then
    git clone https://github.com/vonHartz/riepybdlib.git "$ROOT/riepybdlib"
fi
git -C "$ROOT/riepybdlib" checkout a56cbdb

if ! conda env list | awk '{print $1}' | grep -qx tapas_diffusion; then
    conda create -n tapas_diffusion python=3.10 -y
fi

conda run -n tapas_diffusion python -m pip install -r "$ROOT/TAPAS/requirements.txt"
if ! conda run -n tapas_diffusion python -c \
    "import torch; raise SystemExit(torch.version.cuda is None)"; then
    conda run -n tapas_diffusion python -m pip uninstall -y torch
    conda run -n tapas_diffusion python -m pip install torch==2.1.0 \
        --index-url https://download.pytorch.org/whl/cu118
fi
conda run -n tapas_diffusion python -m pip install \
    -e "$ROOT/riepybdlib" \
    -e "$ROOT/TAPAS[diffusion]" \
    ipykernel
conda run -n tapas_diffusion python -m ipykernel install --user \
    --name tapas_diffusion --display-name "Python (tapas_diffusion)"

mkdir -p "$ROOT/sim"
if [[ ! -d "$SIM_DIR" ]]; then
    ARCHIVE="$ROOT/sim/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz"
    if command -v wget >/dev/null 2>&1; then
        wget -O "$ARCHIVE" \
            https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
    else
        curl -L -o "$ARCHIVE" \
            https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
    fi
    tar -xf "$ARCHIVE" -C "$ROOT/sim"
    rm "$ARCHIVE"
fi

export COPPELIASIM_ROOT="$SIM_DIR"
export LD_LIBRARY_PATH="$SIM_DIR:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="$SIM_DIR"

TAPAS_PREFIX=$(conda run -n tapas_diffusion python -c 'import sys; print(sys.prefix)')
mkdir -p "$TAPAS_PREFIX/etc/conda/activate.d"
printf '%s\n' \
    "export COPPELIASIM_ROOT=\"$SIM_DIR\"" \
    "export LD_LIBRARY_PATH=\"$SIM_DIR:\${LD_LIBRARY_PATH:-}\"" \
    "export QT_QPA_PLATFORM_PLUGIN_PATH=\"$SIM_DIR\"" \
    > "$TAPAS_PREFIX/etc/conda/activate.d/coppeliasim.sh"

conda run -n tapas_diffusion python -m pip install \
    --no-deps --no-build-isolation git+https://github.com/markusgrotz/PyRep.git
conda run -n tapas_diffusion python -m pip install -e "$ROOT/rlbench"

if ! conda env list | awk '{print $1}' | grep -qx peract2; then
    conda create -n peract2 --clone tapas_diffusion -y
fi

PERACT_PREFIX=$(conda run -n peract2 python -c 'import sys; print(sys.prefix)')
mkdir -p "$PERACT_PREFIX/etc/conda/activate.d"
cp "$TAPAS_PREFIX/etc/conda/activate.d/coppeliasim.sh" \
    "$PERACT_PREFIX/etc/conda/activate.d/coppeliasim.sh"

conda run -n peract2 python -m pip install \
    --no-deps git+https://github.com/markusgrotz/YARR.git
conda run -n peract2 python -m pip install \
    moviepy pyrender==0.1.45 timeout-decorator
conda run -n peract2 python -m pip install \
    -e "$ROOT/rlbench" \
    -e "$ROOT/peract_bimanual"
PYTORCH3D_NO_EXTENSION=1 conda run -n peract2 python -m pip install \
    --no-build-isolation git+https://github.com/facebookresearch/pytorch3d.git@v0.7.5
conda run -n peract2 python -m ipykernel install --user \
    --name peract2 --display-name "Python (peract2)"

conda run -n tapas_diffusion python -c \
    "import torch, diffusers, tapas_gmm; print('TAPAS/Diffusion ready. CUDA build:', torch.version.cuda, 'GPU available:', torch.cuda.is_available())"
conda run -n peract2 python -c \
    "import torch, pyrep, rlbench, agents; print('PerAct2 ready. CUDA build:', torch.version.cuda, 'GPU available:', torch.cuda.is_available())"

echo "Setup complete. Available environments: tapas_diffusion and peract2"
