from omegaconf import MISSING

from tapas_gmm.env.rlbench import RLBenchEnvironmentConfig

rlbench_env_config = RLBenchEnvironmentConfig(
    task=MISSING,
    cameras=("wrist_right", "wrist_left", "front"),
    camera_pose={
        "base": (0.2, 0.0, 0.2, 0, 0.194, 0.0, -0.981),
    },
    image_size=(256, 256),
    static=False,
    headless=False,
    scale_action=False,
    delay_gripper=False,
    gripper_plot=False,
    postprocess_actions=True,
    bimanual=True,
    robot_setup="dual_panda",
)
