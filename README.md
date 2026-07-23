## TAPAS and Diffusion

```bash
conda create -n tapas_diffusion python=3.10 -y
conda activate tapas_diffusion
git clone https://github.com/vonHartz/riepybdlib.git
git -C riepybdlib checkout a56cbdb
pip install -r TAPAS/requirements.txt
pip install -e riepybdlib -e rlbench -e "TAPAS[diffusion]"
python -m ipykernel install --user --name tapas_diffusion --display-name "Python (tapas_diffusion)"
```

## PerAct2

```bash
conda create -n peract2 --clone tapas_diffusion
conda activate peract2
pip install --no-deps git+https://github.com/markusgrotz/YARR.git
pip install --no-deps git+https://github.com/markusgrotz/PyRep.git
pip install -e rlbench -e peract_bimanual
PYTORCH3D_NO_EXTENSION=1 pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.5"
python -m ipykernel install --user --name peract2 --display-name "Python (peract2)"
```

Store the demos in `own_code/artifacts/datasets`, select the matching kernel, and run a notebook from `own_code/bimanual`.

## CoppeliaSim

```bash
export COPPELIASIM_ROOT="$PWD/sim/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$COPPELIASIM_ROOT"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
```
