#!/home/nils/tapas310/bin/python

import argparse
import os
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
COPPELIA = ROOT / "sim" / "CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
DEFAULT_DATA_ROOT = HERE / "bimanual_dual_push_buttons_demos"
DEFAULT_TASK = "bimanual_dual_push_buttons"

os.environ["COPPELIASIM_ROOT"] = str(COPPELIA)
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(COPPELIA)
old_ld = os.environ.get("LD_LIBRARY_PATH", "")
if str(COPPELIA) not in old_ld.split(":"):
    os.environ["LD_LIBRARY_PATH"] = f"{old_ld}:{COPPELIA}" if old_ld else str(COPPELIA)
    if os.environ.get("BIMANUAL_PLOT_REEXEC") != "1":
        os.environ["BIMANUAL_PLOT_REEXEC"] = "1"
        os.execvpe(sys.executable, [sys.executable, *sys.argv], os.environ)

for path in (ROOT / "rlbench", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def load_pickle(path):
    if not path.exists():
        return None
    with path.open("rb") as f:
        return pickle.load(f)


def episode_sort_key(path):
    try:
        return int(path.name.replace("episode", ""))
    except ValueError:
        return path.name


def find_episodes(data_root, task):
    episodes_dir = data_root / task / "all_variations" / "episodes"
    return sorted(episodes_dir.glob("episode*"), key=episode_sort_key)


def load_episode(path):
    demo = load_pickle(path / "low_dim_obs.pkl")
    if not demo:
        raise FileNotFoundError(f"Missing {path / 'low_dim_obs.pkl'}")

    data = {}
    add_low_dim_data(data, demo)
    variation = load_pickle(path / "variation_number.pkl")
    descriptions = load_pickle(path / "variation_descriptions.pkl") or []
    return data, variation, descriptions


def stack_side(demo, side, name):
    values = [getattr(getattr(obs, side), name) for obs in demo]
    if any(value is None for value in values):
        return None
    return np.asarray(values)


def stack_obs(demo, name):
    values = [getattr(obs, name) for obs in demo]
    if any(value is None for value in values):
        return None
    return np.asarray(values)


def add_low_dim_data(data, demo):
    for side in ("right", "left"):
        for name in (
            "joint_positions",
            "gripper_open",
            "gripper_pose",
            "gripper_matrix",
            "joint_velocities",
            "joint_forces",
            "gripper_joint_positions",
            "gripper_touch_forces",
            "ignore_collisions",
        ):
            values = stack_side(demo, side, name)
            if values is not None:
                data[f"{side}_{name}"] = values

    task_state = stack_obs(demo, "task_low_dim_state")
    if task_state is not None:
        data["task_low_dim_state"] = task_state


def time_axis(data):
    steps = len(data["right_gripper_open"])
    return np.linspace(0.0, 1.0, steps)


def label_for(episode_path, variation):
    label = episode_path.name
    if variation is not None:
        label += f" v{variation}"
    return label


def plot_gripper_positions(episodes, out_dir, show):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    names = [("x", 0), ("y", 1), ("z", 2)]

    for episode_path, data, variation, descriptions in episodes:
        t = time_axis(data)
        label = label_for(episode_path, variation)

        for col, (axis_name, axis_idx) in enumerate(names):
            axes[0, col].plot(t, data["right_gripper_pose"][:, axis_idx], label=label)
            axes[1, col].plot(t, data["left_gripper_pose"][:, axis_idx], label=label)
            axes[0, col].set_title(f"Right gripper position {axis_name}")
            axes[1, col].set_title(f"Left gripper position {axis_name}")
            axes[1, col].set_xlabel("Normalized time")

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
    axes[0, 0].set_ylabel("Position [m]")
    axes[1, 0].set_ylabel("Position [m]")
    axes[0, 2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "gripper_positions_xyz.png", dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def plot_gripper_3d(episodes, out_dir, show):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for episode_path, data, variation, descriptions in episodes:
        label = label_for(episode_path, variation)

        right = data["right_gripper_pose"][:, :3]
        left = data["left_gripper_pose"][:, :3]
        ax.plot(right[:, 0], right[:, 1], right[:, 2], label=f"{label} right")
        ax.plot(left[:, 0], left[:, 1], left[:, 2], linestyle="--", label=f"{label} left")
        ax.scatter(right[0, 0], right[0, 1], right[0, 2], marker="o", s=20)
        ax.scatter(right[-1, 0], right[-1, 1], right[-1, 2], marker="x", s=35)
        ax.scatter(left[0, 0], left[0, 1], left[0, 2], marker="o", s=20)
        ax.scatter(left[-1, 0], left[-1, 1], left[-1, 2], marker="x", s=35)

    ax.set_title("3D gripper trajectories")
    ax.set_xlabel("X position [m]")
    ax.set_ylabel("Y position [m]")
    ax.set_zlabel("Z position [m]")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "gripper_trajectories_3d.png", dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def plot_gripper_open(episodes, out_dir, show):
    fig, ax = plt.subplots(figsize=(11, 4))

    for episode_path, data, variation, descriptions in episodes:
        t = time_axis(data)
        label = label_for(episode_path, variation)
        ax.plot(t, data["right_gripper_open"], label=f"{label} right")
        ax.plot(t, data["left_gripper_open"], linestyle="--", label=f"{label} left")

    ax.set_title("Gripper open values")
    ax.set_xlabel("Normalized time")
    ax.set_ylabel("Open value (0=closed, 1=open)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "gripper_open.png", dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def plot_joint_positions(episodes, out_dir, show):
    fig, axes = plt.subplots(7, 2, figsize=(12, 14), sharex=True)

    for episode_path, data, variation, descriptions in episodes:
        t = time_axis(data)
        label = label_for(episode_path, variation)

        for joint_idx in range(7):
            axes[joint_idx, 0].plot(t, data["right_joint_positions"][:, joint_idx], label=label)
            axes[joint_idx, 1].plot(t, data["left_joint_positions"][:, joint_idx], label=label)
            axes[joint_idx, 0].set_ylabel(f"Joint {joint_idx + 1} [rad]")

    axes[0, 0].set_title("Right arm joint positions")
    axes[0, 1].set_title("Left arm joint positions")
    axes[-1, 0].set_xlabel("Normalized time")
    axes[-1, 1].set_xlabel("Normalized time")

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "joint_positions.png", dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def plot_gripper_orientation(episodes, out_dir, show):
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True)
    names = ["qx", "qy", "qz", "qw"]

    for episode_path, data, variation, descriptions in episodes:
        t = time_axis(data)
        label = label_for(episode_path, variation)
        for col, name in enumerate(names):
            axes[0, col].plot(t, data["right_gripper_pose"][:, col + 3], label=label)
            axes[1, col].plot(t, data["left_gripper_pose"][:, col + 3], label=label)
            axes[0, col].set_title(f"Right gripper orientation {name}")
            axes[1, col].set_title(f"Left gripper orientation {name}")
            axes[1, col].set_xlabel("Normalized time")

    for ax in axes.flat:
        ax.set_ylabel("Quaternion value")
        ax.grid(True, alpha=0.3)
    axes[0, 3].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "gripper_orientation_quaternion.png", dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def plot_joint_quantity(episodes, out_dir, show, key_suffix, title, ylabel, filename):
    if not all(f"right_{key_suffix}" in data and f"left_{key_suffix}" in data for _, data, _, _ in episodes):
        return

    fig, axes = plt.subplots(7, 2, figsize=(12, 14), sharex=True)
    for episode_path, data, variation, descriptions in episodes:
        t = time_axis(data)
        label = label_for(episode_path, variation)
        right = data[f"right_{key_suffix}"]
        left = data[f"left_{key_suffix}"]
        for joint_idx in range(7):
            axes[joint_idx, 0].plot(t, right[:, joint_idx], label=label)
            axes[joint_idx, 1].plot(t, left[:, joint_idx], label=label)
            axes[joint_idx, 0].set_ylabel(f"Joint {joint_idx + 1} {ylabel}")

    axes[0, 0].set_title(f"Right arm {title}")
    axes[0, 1].set_title(f"Left arm {title}")
    axes[-1, 0].set_xlabel("Normalized time")
    axes[-1, 1].set_xlabel("Normalized time")
    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def plot_gripper_joint_positions(episodes, out_dir, show):
    if not all("right_gripper_joint_positions" in data and "left_gripper_joint_positions" in data for _, data, _, _ in episodes):
        return

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for episode_path, data, variation, descriptions in episodes:
        t = time_axis(data)
        label = label_for(episode_path, variation)
        for joint_idx in range(2):
            axes[0, joint_idx].plot(t, data["right_gripper_joint_positions"][:, joint_idx], label=label)
            axes[1, joint_idx].plot(t, data["left_gripper_joint_positions"][:, joint_idx], label=label)
            axes[0, joint_idx].set_title(f"Right gripper finger joint {joint_idx + 1}")
            axes[1, joint_idx].set_title(f"Left gripper finger joint {joint_idx + 1}")
            axes[1, joint_idx].set_xlabel("Normalized time")

    for ax in axes.flat:
        ax.set_ylabel("Joint position [m or rad]")
        ax.grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "gripper_joint_positions.png", dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def plot_task_low_dim_state(episodes, out_dir, show):
    available = [(episode_path, data, variation) for episode_path, data, variation, _ in episodes if "task_low_dim_state" in data]
    if not available:
        return

    for episode_path, data, variation in available:
        fig, ax = plt.subplots(figsize=(13, 5))
        state = data["task_low_dim_state"].T
        image = ax.imshow(state, aspect="auto", interpolation="nearest", origin="lower")
        title = f"Task low-dimensional state heatmap ({label_for(episode_path, variation)})"
        ax.set_title(title)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Task state dimension")
        fig.colorbar(image, ax=ax, label="State value")
        fig.tight_layout()
        fig.savefig(out_dir / f"{episode_path.name}_task_low_dim_state_heatmap.png", dpi=160)
        if show:
            plt.show()
        plt.close(fig)


def example_episode(episodes):
    return episodes[0]


def plot_one_graph_summary(episodes, out_dir, show):
    episode_path, data, variation, descriptions = example_episode(episodes)
    t = time_axis(data)
    desc = descriptions[0] if descriptions else "no description"
    fig = plt.figure(figsize=(18, 16))
    grid = fig.add_gridspec(4, 3)

    fig.suptitle(
        f"Bimanual demo data overview: {label_for(episode_path, variation)} - {desc}",
        fontsize=14)

    ax = fig.add_subplot(grid[0, 0])
    ax.plot(t, data["right_gripper_pose"][:, 0], label="right x")
    ax.plot(t, data["left_gripper_pose"][:, 0], label="left x")
    ax.set_title("Gripper position example")
    ax.set_xlabel("Normalized time")
    ax.set_ylabel("X position [m]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(grid[0, 1], projection="3d")
    right = data["right_gripper_pose"][:, :3]
    left = data["left_gripper_pose"][:, :3]
    ax.plot(right[:, 0], right[:, 1], right[:, 2], label="right")
    ax.plot(left[:, 0], left[:, 1], left[:, 2], label="left")
    ax.set_title("3D gripper path")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(grid[0, 2])
    ax.plot(t, data["right_gripper_pose"][:, 6], label="right qw")
    ax.plot(t, data["left_gripper_pose"][:, 6], label="left qw")
    ax.set_title("Gripper orientation example")
    ax.set_xlabel("Normalized time")
    ax.set_ylabel("Quaternion qw")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(grid[1, 0])
    ax.plot(t, data["right_gripper_open"], label="right")
    ax.plot(t, data["left_gripper_open"], label="left")
    ax.set_title("Gripper open state")
    ax.set_xlabel("Normalized time")
    ax.set_ylabel("Open value (0=closed, 1=open)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(grid[1, 1])
    ax.plot(t, data["right_joint_positions"][:, 0], label="right joint 1")
    ax.plot(t, data["left_joint_positions"][:, 0], label="left joint 1")
    ax.set_title("Joint position example")
    ax.set_xlabel("Normalized time")
    ax.set_ylabel("Joint 1 position [rad]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(grid[1, 2])
    if "right_joint_velocities" in data and "left_joint_velocities" in data:
        ax.plot(t, data["right_joint_velocities"][:, 0], label="right joint 1")
        ax.plot(t, data["left_joint_velocities"][:, 0], label="left joint 1")
    ax.set_title("Joint velocity example")
    ax.set_xlabel("Normalized time")
    ax.set_ylabel("Joint 1 velocity [rad/s]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(grid[2, 0])
    if "right_joint_forces" in data and "left_joint_forces" in data:
        ax.plot(t, data["right_joint_forces"][:, 0], label="right joint 1")
        ax.plot(t, data["left_joint_forces"][:, 0], label="left joint 1")
    ax.set_title("Joint force example")
    ax.set_xlabel("Normalized time")
    ax.set_ylabel("Joint 1 force/torque [N or Nm]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(grid[2, 1])
    if "right_gripper_joint_positions" in data and "left_gripper_joint_positions" in data:
        ax.plot(t, data["right_gripper_joint_positions"][:, 0], label="right finger joint 1")
        ax.plot(t, data["left_gripper_joint_positions"][:, 0], label="left finger joint 1")
    ax.set_title("Gripper finger joint example")
    ax.set_xlabel("Normalized time")
    ax.set_ylabel("Finger joint position [m or rad]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(grid[2, 2])
    if "right_gripper_touch_forces" in data and "left_gripper_touch_forces" in data:
        ax.plot(t, data["right_gripper_touch_forces"][:, 0], label="right sensor 1")
        ax.plot(t, data["left_gripper_touch_forces"][:, 0], label="left sensor 1")
    ax.set_title("Gripper touch force example")
    ax.set_xlabel("Normalized time")
    ax.set_ylabel("Touch force value")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(grid[3, 0])
    if "right_ignore_collisions" in data and "left_ignore_collisions" in data:
        ax.step(t, data["right_ignore_collisions"], where="post", label="right")
        ax.step(t, data["left_ignore_collisions"], where="post", label="left")
    ax.set_title("Ignore collisions flag")
    ax.set_xlabel("Normalized time")
    ax.set_ylabel("Flag value (0=false, 1=true)")
    ax.set_ylim(-0.1, 1.1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(grid[3, 1:])
    if "task_low_dim_state" in data:
        image = ax.imshow(
            data["task_low_dim_state"].T,
            aspect="auto",
            interpolation="nearest",
            origin="lower")
        fig.colorbar(image, ax=ax, label="State value")
    ax.set_title("Task low-dimensional state")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Task state dimension")

    fig.tight_layout()
    out_file = out_dir / "bimanual_demo_overview.png"
    fig.savefig(out_file, dpi=170)
    if show:
        plt.show()
    plt.close(fig)
    return out_file


def add_line_plot(ax, t, values, title, ylabel, labels=None):
    values = np.asarray(values)
    if values.ndim == 1:
        ax.plot(t, values)
    elif values.ndim == 2:
        for i in range(values.shape[1]):
            label = labels[i] if labels and i < len(labels) else f"d{i}"
            ax.plot(t, values[:, i], label=label)
        if values.shape[1] <= 8:
            ax.legend(fontsize=6, ncol=2)
    else:
        flat = values.reshape(values.shape[0], -1)
        for i in range(flat.shape[1]):
            label = labels[i] if labels and i < len(labels) else f"d{i}"
            ax.plot(t, flat[:, i], label=label)
        if flat.shape[1] <= 8:
            ax.legend(fontsize=6, ncol=2)
    ax.set_title(title)
    ax.set_xlabel("Normalized time")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def add_heatmap(ax, values, title, ylabel, colorbar_label, fig):
    values = np.asarray(values)
    if values.ndim > 2:
        values = values.reshape(values.shape[0], -1)
    image = ax.imshow(values.T, aspect="auto", interpolation="nearest", origin="lower")
    ax.set_title(title)
    ax.set_xlabel("Timestep")
    ax.set_ylabel(ylabel)
    fig.colorbar(image, ax=ax, label=colorbar_label)


def plot_all_low_dim_variables(episodes, out_dir, show):
    episode_path, data, variation, descriptions = example_episode(episodes)
    t = time_axis(data)
    desc = descriptions[0] if descriptions else "no description"

    fig, axes = plt.subplots(6, 3, figsize=(22, 25))
    fig.suptitle(
        f"All low_dim_obs variables: {episode_path.name} - {desc}",
        fontsize=15)
    axes = axes.ravel()

    panels = [
        ("right_joint_positions", "Right joint positions", "Joint position [rad]", "line", [f"joint {i}" for i in range(1, 8)]),
        ("left_joint_positions", "Left joint positions", "Joint position [rad]", "line", [f"joint {i}" for i in range(1, 8)]),
        ("right_joint_velocities", "Right joint velocities", "Joint velocity [rad/s]", "line", [f"joint {i}" for i in range(1, 8)]),
        ("left_joint_velocities", "Left joint velocities", "Joint velocity [rad/s]", "line", [f"joint {i}" for i in range(1, 8)]),
        ("right_joint_forces", "Right joint forces", "Joint force/torque [N or Nm]", "line", [f"joint {i}" for i in range(1, 8)]),
        ("left_joint_forces", "Left joint forces", "Joint force/torque [N or Nm]", "line", [f"joint {i}" for i in range(1, 8)]),
        ("right_gripper_open", "Right gripper open", "Open value (0=closed, 1=open)", "line", None),
        ("left_gripper_open", "Left gripper open", "Open value (0=closed, 1=open)", "line", None),
        ("right_gripper_pose", "Right gripper pose", "Pose value", "line", ["x [m]", "y [m]", "z [m]", "qx", "qy", "qz", "qw"]),
        ("left_gripper_pose", "Left gripper pose", "Pose value", "line", ["x [m]", "y [m]", "z [m]", "qx", "qy", "qz", "qw"]),
        ("right_gripper_joint_positions", "Right gripper finger joints", "Finger joint position [m or rad]", "line", ["finger joint 1", "finger joint 2"]),
        ("left_gripper_joint_positions", "Left gripper finger joints", "Finger joint position [m or rad]", "line", ["finger joint 1", "finger joint 2"]),
        ("right_gripper_touch_forces", "Right gripper touch forces", "Touch force value", "line", [f"touch force {i}" for i in range(1, 7)]),
        ("left_gripper_touch_forces", "Left gripper touch forces", "Touch force value", "line", [f"touch force {i}" for i in range(1, 7)]),
        ("right_ignore_collisions", "Right ignore collisions", "Flag value (0=false, 1=true)", "line", None),
        ("left_ignore_collisions", "Left ignore collisions", "Flag value (0=false, 1=true)", "line", None),
    ]

    for ax, (key, title, ylabel, kind, labels) in zip(axes, panels):
        if key not in data:
            ax.set_title(f"{title} missing")
            ax.axis("off")
            continue
        if kind == "heatmap":
            add_heatmap(ax, data[key], title, ylabel, "Value", fig)
        else:
            add_line_plot(ax, t, data[key], title, ylabel, labels)

    for ax in axes[len(panels):]:
        ax.axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out_file = out_dir / "low_dim_obs_all_variables.png"
    fig.savefig(out_file, dpi=160)
    if show:
        plt.show()
    plt.close(fig)
    return out_file


def print_summary(episodes):
    for episode_path, data, variation, descriptions in episodes:
        desc = descriptions[0] if descriptions else "no description"
        steps = len(data["right_gripper_open"])
        fields = ", ".join(sorted(data.keys()))
        print(f"{episode_path.name}: variation={variation}, steps={steps}, description={desc}")
        print(f"  fields: {fields}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    out_dir = (args.out_dir or (data_root / "plots")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    episode_paths = find_episodes(data_root, args.task)
    episodes = []
    for path in episode_paths:
        try:
            episodes.append((path, *load_episode(path)))
        except FileNotFoundError as exc:
            print(f"Skipping {path.name}: {exc}")

    if not episodes:
        print(f"No episodes with low_dim_obs.pkl found under {data_root}")
        return

    print_summary(episodes)
    out_file = plot_all_low_dim_variables(episodes, out_dir, args.show)
    print(f"Saved all-variable plot to: {out_file}")


if __name__ == "__main__":
    main()
