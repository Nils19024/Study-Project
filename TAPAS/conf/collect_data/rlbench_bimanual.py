from conf._machine import data_naming_config
from pathlib import Path
from conf.dataset.scene.rlbench_bimanual import scene_dataset_config
from conf.env.rlbench.bimanual import rlbench_env_config
from tapas_gmm.collect_data_rlbench import Config
from tapas_gmm.env import Environment

PROJECT_ROOT = Path(__file__).resolve().parents[3]
data_naming_config.data_root = str(
    PROJECT_ROOT / "own_code" / "outputs" / "bimanual" / "tapas_observations"
)

config = Config(
    n_episodes=5,
    sequence_len=None,
    data_naming=data_naming_config,
    dataset_config=scene_dataset_config,
    env_type=Environment.RLBENCH,
    env_config=rlbench_env_config,
)
