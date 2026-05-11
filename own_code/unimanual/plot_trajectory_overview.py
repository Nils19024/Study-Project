#!/usr/bin/env python3
"""Create one overview image for a saved TAPAS/RLBench trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAJ = (
    CODE_ROOT
    / "TAPAS/data/PushButton/demos/trajectories/2026_05_04-09_08_15"
)


def numeric_files(folder: Path, suffix: str) -> list[Path]:
    return sorted(folder.glob(f"*{suffix}"), key=lambda p: int(p.stem))


def load_tensor_series(folder: Path) -> np.ndarray:
    values = [torch.load(path).detach().cpu().numpy() for path in numeric_files(folder, ".dat")]
    return np.stack(values)


def load_object_poses(folder: Path) -> dict[str, np.ndarray]:
    frames = [torch.load(path) for path in numeric_files(folder, ".dat")]
    keys = list(frames[0].keys())
    return {key: np.stack([frame[key].detach().cpu().numpy() for frame in frames]) for key in keys}


def load_rgb_stats(folder: Path) -> tuple[np.ndarray, Image.Image]:
    files = numeric_files(folder, ".png")
    means = []
    for path in files:
        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        means.append(image.mean(axis=(0, 1)))
    return np.stack(means), Image.open(files[0]).convert("RGB")


def plot_lines(ax, data: np.ndarray, title: str, ylabel: str, labels: list[str]) -> None:
    time = np.arange(len(data))
    for idx, label in enumerate(labels):
        ax.plot(time, data[:, idx], label=label, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Zeitschritt")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncols=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj", type=Path, default=DEFAULT_TRAJ)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    traj = args.traj.resolve()
    out = args.out or CODE_ROOT / "own_code" / f"{traj.name}_overview.png"

    metadata = json.loads((traj / "metadata.json").read_text())
    action = load_tensor_series(traj / "action")
    ee_pose = load_tensor_series(traj / "ee_pose")
    gripper_state = load_tensor_series(traj / "gripper_state")
    joint_pos = load_tensor_series(traj / "joint_pos")
    joint_vel = load_tensor_series(traj / "joint_vel")
    feedback = load_tensor_series(traj / "feedback")
    object_poses = load_object_poses(traj / "object_poses")

    cam_f_depth = load_tensor_series(traj / "cam_f_d")
    cam_w_depth = load_tensor_series(traj / "cam_w_d")
    cam_f_mask = load_tensor_series(traj / "cam_f_mask_gt")
    cam_w_mask = load_tensor_series(traj / "cam_w_mask_gt")
    cam_f_ext = load_tensor_series(traj / "cam_f_ext")
    cam_w_ext = load_tensor_series(traj / "cam_w_ext")
    cam_f_rgb_mean, cam_f_first = load_rgb_stats(traj / "cam_f_rgb")
    cam_w_rgb_mean, cam_w_first = load_rgb_stats(traj / "cam_w_rgb")
    cam_f_int = torch.load(traj / "cam_f_int.dat").detach().cpu().numpy()
    cam_w_int = torch.load(traj / "cam_w_int.dat").detach().cpu().numpy()

    fig, axes = plt.subplots(7, 3, figsize=(22, 28))
    fig.suptitle(f"TAPAS/RLBench Trajectory Overview\n{traj}\nlen={metadata['len']}", fontsize=14)
    axes = axes.ravel()

    plot_lines(axes[0], action, "action", "Delta / Gripper", ["dx", "dy", "dz", "d_rx", "d_ry", "d_rz", "gripper"])
    plot_lines(axes[1], ee_pose, "ee_pose", "Position / Quaternion", ["x", "y", "z", "qx", "qy", "qz", "qw"])
    plot_lines(axes[2], joint_pos, "joint_pos", "Gelenkwinkel", [f"j{i+1}" for i in range(7)])
    plot_lines(axes[3], joint_vel, "joint_vel", "Gelenkgeschwindigkeit", [f"j{i+1}" for i in range(7)])
    plot_lines(axes[4], gripper_state, "gripper_state", "Offen/Geschlossen", ["gripper"])
    plot_lines(axes[5], feedback, "feedback", "Expert-Feedback", ["feedback"])

    time = np.arange(metadata["len"])
    for key, values in object_poses.items():
        axes[6].plot(time, values[:, 0], label=f"{key}.x", linewidth=1)
        axes[7].plot(time, values[:, 1], label=f"{key}.y", linewidth=1)
        axes[8].plot(time, values[:, 2], label=f"{key}.z", linewidth=1)
    for ax, title, ylabel in zip(axes[6:9], ["object_poses x", "object_poses y", "object_poses z"], ["x", "y", "z"]):
        ax.set_title(title)
        ax.set_xlabel("Zeitschritt")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=6, ncols=2)

    depth_stats = [
        (cam_f_depth, "front depth"),
        (cam_w_depth, "wrist depth"),
    ]
    for ax, (data, title) in zip(axes[9:11], depth_stats):
        ax.plot(time, data.mean(axis=(1, 2)), label="mean")
        ax.plot(time, data.min(axis=(1, 2)), label="min")
        ax.plot(time, data.max(axis=(1, 2)), label="max")
        ax.set_title(title)
        ax.set_xlabel("Zeitschritt")
        ax.set_ylabel("Tiefe")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)

    for ax, data, title in [(axes[11], cam_f_mask, "front mask_gt"), (axes[12], cam_w_mask, "wrist mask_gt")]:
        ax.plot(time, data.mean(axis=(1, 2)), label="mean object id")
        ax.plot(time, [len(np.unique(frame)) for frame in data], label="unique ids")
        ax.set_title(title)
        ax.set_xlabel("Zeitschritt")
        ax.set_ylabel("Maskenwert / Anzahl IDs")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)

    plot_lines(axes[13], cam_f_rgb_mean, "front rgb mean", "mittlere Farbe", ["R", "G", "B"])
    plot_lines(axes[14], cam_w_rgb_mean, "wrist rgb mean", "mittlere Farbe", ["R", "G", "B"])
    plot_lines(axes[15], cam_f_ext[:, :3, 3], "front cam_f_ext translation", "Position", ["x", "y", "z"])
    plot_lines(axes[16], cam_w_ext[:, :3, 3], "wrist cam_w_ext translation", "Position", ["x", "y", "z"])

    axes[17].imshow(cam_f_first)
    axes[17].set_title("front rgb first frame")
    axes[17].set_xlabel("u Pixel")
    axes[17].set_ylabel("v Pixel")

    axes[18].imshow(cam_w_first)
    axes[18].set_title("wrist rgb first frame")
    axes[18].set_xlabel("u Pixel")
    axes[18].set_ylabel("v Pixel")

    for ax, mat, title in [(axes[19], cam_f_int, "cam_f_int"), (axes[20], cam_w_int, "cam_w_int")]:
        im = ax.imshow(mat)
        ax.set_title(title)
        ax.set_xlabel("Matrix-Spalte")
        ax.set_ylabel("Matrix-Zeile")
        for row in range(mat.shape[0]):
            for col in range(mat.shape[1]):
                ax.text(col, row, f"{mat[row, col]:.1f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=150)
    print(f"Saved overview image to: {out}")


if __name__ == "__main__":
    main()
