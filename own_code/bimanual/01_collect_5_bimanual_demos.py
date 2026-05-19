#!/usr/bin/env python3

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUTS_ROOT = ROOT / "own_code" / "outputs"
COPPELIA = ROOT / "sim" / "CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
DEFAULT_SAVE_ROOT = OUTPUTS_ROOT / "bimanual" / "bimanual_dual_push_buttons_demos"


def setup():
    os.environ["COPPELIASIM_ROOT"] = str(COPPELIA)
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(COPPELIA)

    old_ld = os.environ.get("LD_LIBRARY_PATH", "")
    if str(COPPELIA) not in old_ld.split(":"):
        os.environ["LD_LIBRARY_PATH"] = f"{old_ld}:{COPPELIA}" if old_ld else str(COPPELIA)
        if os.environ.get("BIMANUAL_COLLECT_REEXEC") != "1":
            os.environ["BIMANUAL_COLLECT_REEXEC"] = "1"
            os.execvpe(sys.executable, [sys.executable, *sys.argv], os.environ)

    for path in (ROOT / "rlbench", ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def save_demo(demo, path, variation, descriptions):
    from rlbench.backend.const import LOW_DIM_PICKLE, VARIATION_DESCRIPTIONS, VARIATION_NUMBER

    path.mkdir(parents=True, exist_ok=True)

    for name, data in (
        (LOW_DIM_PICKLE, demo),
        (VARIATION_NUMBER, variation),
        (VARIATION_DESCRIPTIONS, descriptions),
    ):
        with (path / name).open("wb") as f:
            pickle.dump(data, f)


def make_env():
    from rlbench.action_modes.action_mode import BimanualMoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import BimanualJointPosition
    from rlbench.action_modes.gripper_action_modes import BimanualDiscrete
    from rlbench.environment import Environment
    from rlbench.observation_config import ObservationConfig

    obs = ObservationConfig()
    obs.set_all_high_dim(False)
    obs.camera_configs = {}
    obs.set_all_low_dim(True)
    obs.gripper_matrix = True
    obs.task_low_dim_state = True

    return Environment(
        action_mode=BimanualMoveArmThenGripper(BimanualJointPosition(), BimanualDiscrete()),
        obs_config=obs,
        headless=False,
        robot_setup="dual_panda",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=30)
    parser.add_argument("--save-root", type=Path, default=DEFAULT_SAVE_ROOT)
    parser.add_argument("--variation", type=int)
    args = parser.parse_args()

    setup()

    from rlbench.backend.const import EPISODE_FOLDER, EPISODES_FOLDER, VARIATIONS_ALL_FOLDER
    from rlbench.bimanual_tasks.bimanual_dual_push_buttons import BimanualDualPushButtons

    env = make_env()
    save_root = args.save_root.resolve()
    episodes_dir = save_root / "bimanual_dual_push_buttons" / VARIATIONS_ALL_FOLDER / EPISODES_FOLDER
    episodes_dir.mkdir(parents=True, exist_ok=True)

    print("Launching visible CoppeliaSim/RLBench environment...")
    print(f"Saving demos to: {save_root}")

    env.launch()
    try:
        variation_count = env.get_task(BimanualDualPushButtons).variation_count()
        collected = 0

        for attempt in range(1, args.max_attempts + 1):
            if collected >= args.episodes:
                break

            variation = (
                np.random.randint(variation_count)
                if args.variation is None
                else args.variation % variation_count
            )

            try:
                task = env.get_task(BimanualDualPushButtons)
                task.set_variation(int(variation))
                descriptions, _ = task.reset()

                print(f"[{collected + 1}/{args.episodes}] Collecting variation {variation}: {descriptions[0]}")

                demo = task.get_demos(amount=1, live_demos=True)[0]
                path = episodes_dir / (EPISODE_FOLDER % collected)
                save_demo(demo, path, int(variation), descriptions)

                print(f"Saved {len(demo)} steps to {path}")
                collected += 1

            except Exception as exc:
                print(f"Attempt {attempt}/{args.max_attempts} failed for variation {variation}: {exc}")

        if collected < args.episodes:
            raise RuntimeError(f"Only collected {collected}/{args.episodes} demos.")
    finally:
        env.shutdown()


if __name__ == "__main__":
    main()
