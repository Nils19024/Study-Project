import os
import signal
import sys
import time

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig


def main():
    action_mode = MoveArmThenGripper(
        arm_action_mode=JointVelocity(),
        gripper_action_mode=Discrete(),
    )
    env = Environment(
        action_mode=action_mode,
        obs_config=ObservationConfig(joint_forces=False),
        headless=False,
    )

    def stop(_signum, _frame):
        env.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    try:
        print("Starting CoppeliaSim GUI...", flush=True)
        env.launch()
        print("CoppeliaSim GUI is running. Press Ctrl+C here to stop.", flush=True)
        demo_seconds = float(os.environ.get("RLBENCH_DEMO_SECONDS", "0"))
        if demo_seconds > 0:
            time.sleep(demo_seconds)
        else:
            while True:
                time.sleep(1)
    finally:
        env.shutdown()


if __name__ == "__main__":
    main()
