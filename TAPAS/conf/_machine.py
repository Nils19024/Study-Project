from omegaconf import MISSING

from tapas_gmm.utils.misc import DataNamingConfig

data_naming_config = DataNamingConfig(
    feedback_type=MISSING, task=MISSING, data_root="data"
)


CODE_ROOT = "/home/nils/Documents/Study Project/Code"
CUDA_PATH = "/usr/local/cuda-11.1"
COPPELIASIM_ROOT = CODE_ROOT + "/sim/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
LD_LIBRARY_PATH = ":".join(
    [CUDA_PATH + "/lib64", COPPELIASIM_ROOT]
)
