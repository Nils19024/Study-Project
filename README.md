## Cluster setup

```bash
git clone https://github.com/Nils19024/Study-Project.git
cd Study-Project
bash setup_cluster.sh
```

## Manual setup

### CoppeliaSim

```bash
mkdir -p sim
wget -P sim https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
tar -xf sim/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz -C sim
rm sim/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz

export COPPELIASIM_ROOT="$PWD/sim/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
export LD_LIBRARY_PATH="$COPPELIASIM_ROOT:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
```

### TAPAS and Diffusion

```bash
conda create -n tapas_diffusion python=3.10 -y
conda activate tapas_diffusion
git clone https://github.com/vonHartz/riepybdlib.git
git -C riepybdlib checkout a56cbdb
pip install -r TAPAS/requirements.txt
pip uninstall -y torch
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install --no-deps --no-build-isolation git+https://github.com/markusgrotz/PyRep.git
pip install -e riepybdlib -e rlbench -e "TAPAS[diffusion]"
pip install ipykernel
python -m ipykernel install --user --name tapas_diffusion --display-name "Python (tapas_diffusion)"
```

### PerAct2

```bash
conda create -n peract2 --clone tapas_diffusion
conda activate peract2
pip install --no-deps git+https://github.com/markusgrotz/YARR.git
pip install moviepy pyrender==0.1.45 timeout-decorator
pip install -e rlbench -e peract_bimanual
PYTORCH3D_NO_EXTENSION=1 pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.5"
python -m ipykernel install --user --name peract2 --display-name "Python (peract2)"
```
