from functools import lru_cache
from typing import Any

import numpy as np
import torch
from loguru import logger

from tapas_gmm.dataset.demos import (
    Demos,
    DemosSegment,
    PartialFrameViewDemos,
    add_time_dimension,
    get_frame_transform_flat,
)
from tapas_gmm.utils.geometry_torch import (
    axis_angle_to_matrix,
    axis_angle_to_quaternion,
    identity_quaternions,
    invert_homogenous_transform,
    quaternion_to_matrix,
    standardize_quaternion,
)
from tapas_gmm.utils.observation import SceneObservation
from tapas_gmm.utils.torch import cat, to_numpy


def _pose7_to_transform(poses: torch.Tensor) -> torch.Tensor:
    pos = poses[..., :3].reshape(-1, 3)
    quat = standardize_quaternion(poses[..., 3:].reshape(-1, 4))
    rot = torch.Tensor(quaternion_to_matrix(quat))
    return get_frame_transform_flat(rot, pos, invert=False).reshape(
        *poses.shape[:-1], 4, 4
    )


def _project_arms_to_frames(
    transforms: torch.Tensor | tuple[torch.Tensor, ...],
    poses: torch.Tensor | tuple[torch.Tensor, ...],
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    if isinstance(transforms, tuple):
        return tuple(_project_arms_to_frames(t, p) for t, p in zip(transforms, poses))

    if transforms.ndim == 4:
        if transforms.shape[1] == 1:
            transforms = transforms.repeat(1, poses.shape[0], 1, 1)

        pos = poses[..., :3, 3]
        ones = torch.ones(*pos.shape[:-1], 1, dtype=pos.dtype, device=pos.device)
        pos_hom = torch.cat((pos, ones), dim=-1)

        projected = torch.einsum("ftij,taj->tfai", transforms, pos_hom)
        return projected[..., :3].reshape(poses.shape[0], transforms.shape[0], -1)

    if transforms.shape[2] == 1:
        transforms = transforms.repeat(1, 1, poses.shape[1], 1, 1)

    pos = poses[..., :3, 3]
    ones = torch.ones(*pos.shape[:-1], 1, dtype=pos.dtype, device=pos.device)
    pos_hom = torch.cat((pos, ones), dim=-1)

    projected = torch.einsum("nftij,ntaj->ntfai", transforms, pos_hom)
    return projected[..., :3].reshape(
        poses.shape[0], poses.shape[1], transforms.shape[1], -1
    )


def _merge_close_indices(indices: list[int], max_distance: int) -> tuple[int, ...]:
    if not indices:
        return tuple()

    clusters = [[indices[0]]]
    for idx in indices[1:]:
        if idx - clusters[-1][-1] <= max_distance:
            clusters[-1].append(idx)
        else:
            clusters.append([idx])

    return tuple(int(round(float(np.mean(cluster)))) for cluster in clusters)


def _either_arm_slow_indices(
    speeds: tuple[torch.Tensor, ...],
    min_len: int,
    velocity_threshold: float,
    min_end_distance: int,
    max_idx_dist: int,
) -> tuple[tuple[int, ...], ...]:
    segment_indices = []

    for traj in speeds:
        candidates = []
        traj_len = traj.shape[0]

        for arm in range(traj.shape[1]):
            slow = traj[:, arm] < velocity_threshold
            padded = torch.cat(
                (
                    torch.tensor([False], device=slow.device),
                    slow,
                    torch.tensor([False], device=slow.device),
                )
            )
            starts = torch.argwhere(~padded[:-1] & padded[1:]).flatten()
            stops = torch.argwhere(padded[:-1] & ~padded[1:]).flatten()

            for start, stop in zip(starts.tolist(), stops.tolist()):
                if stop - start < min_len:
                    continue
                center = int(round((start + stop - 1) / 2))
                if center <= min_end_distance or center >= traj_len - min_end_distance:
                    continue
                candidates.append(center)

        candidates = sorted(set(candidates))
        segment_indices.append(_merge_close_indices(candidates, max_idx_dist))

    min_count = min(len(indices) for indices in segment_indices)
    if any(len(indices) != min_count for indices in segment_indices):
        logger.warning("Got different bimanual segmentation counts. Truncating extras.")
        segment_indices = [indices[:min_count] for indices in segment_indices]

    return tuple(segment_indices)


class _BimanualDemosMixin:
    per_frame_position_count = 2
    gripper_action_dims = 2

    def partial_frame_view(self, frame_indices: list[int]):
        return BimanualPartialFrameView(self, frame_indices)

    @lru_cache
    def get_x_per_frame(
        self,
        subsampled: bool = True,
        fixed_frames: bool = False,
        flat: bool = False,
        pos_only: bool = False,
        as_quaternion: bool = True,
        skip_quat_dim: int | None = 0,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if not pos_only:
            raise NotImplementedError("Bimanual GMM training is position-only for now.")

        transforms = self._world2frames_fixed if fixed_frames else self.world2frames
        if subsampled and not fixed_frames:
            transforms = self._subsample(transforms)

        poses = self.stacked_ee_poses if subsampled else self.ee_poses
        obs = _project_arms_to_frames(transforms, poses)

        if flat:
            if subsampled:
                obs = obs.reshape(self.n_trajs, self.ss_len, -1)
            else:
                obs = tuple(o.reshape(o.shape[0], -1) for o in obs)

        return obs

    @lru_cache
    def get_per_frame_data(
        self,
        subsampled: bool = True,
        fixed_frames: bool = False,
        flat: bool = False,
        numpy: bool = False,
        pos_only: bool = False,
        as_quaternion: bool = True,
        skip_quat_dim: int | None = 0,
        add_time_dim: bool = False,
        add_action_dim: bool = False,
        action_as_orientation: bool = False,
        action_with_magnitude: bool = False,
        add_gripper_action: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | np.ndarray | tuple[np.ndarray, ...]:
        if add_action_dim or action_with_magnitude:
            raise NotImplementedError("Bimanual action components are not fitted yet.")
        if add_time_dim or add_gripper_action:
            assert flat, "Can't stack global vars on non-flat frame data."

        obs = self.get_x_per_frame(
            subsampled=subsampled,
            fixed_frames=fixed_frames,
            flat=flat,
            pos_only=pos_only,
            as_quaternion=as_quaternion,
            skip_quat_dim=skip_quat_dim,
        )

        if flat and add_time_dim:
            obs = add_time_dimension(
                obs, start=self.relative_start_time, stop=self.relative_stop_time
            )

        if flat and add_gripper_action:
            obs = cat(obs, self.get_gripper_action(subsampled=subsampled), dim=-1)

        return to_numpy(obs) if numpy else obs

    @lru_cache
    def get_world_data(
        self,
        add_time_dim: bool = False,
        add_action_dim: bool = False,
        position_only: bool = False,
        as_quaternion: bool = True,
        skip_quat_dim: int | None = None,
        action_as_orientation: bool = False,
        action_with_magnitude: bool = False,
        add_gripper_action: bool = False,
        numpy: bool = False,
    ) -> tuple[torch.Tensor, ...] | tuple[np.ndarray, ...]:
        if not position_only or add_action_dim or action_with_magnitude:
            raise NotImplementedError("Bimanual world data is position-only for now.")

        obs = tuple(p[..., :3, 3].reshape(p.shape[0], -1) for p in self.ee_poses)

        if add_time_dim:
            obs = add_time_dimension(
                obs, start=self.relative_start_time, stop=self.relative_stop_time
            )

        if add_gripper_action:
            obs = cat(obs, self.get_gripper_action(subsampled=False), dim=-1)

        return to_numpy(obs) if numpy else obs

    def get_gripper_action(self, subsampled: bool = False):
        return self.stacked_gripper_actions if subsampled else self.gripper_actions

    def get_action_magnitude(
        self,
        subsampled: bool = False,
        position_only: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        actions = self.stacked_actions_raw if subsampled else self.actions_raw
        if isinstance(actions, tuple):
            return tuple(self._speed_per_arm(action) for action in actions)
        return self._speed_per_arm(actions)

    @staticmethod
    def _speed_per_arm(action: torch.Tensor) -> torch.Tensor:
        right = torch.linalg.norm(action[..., :3], dim=-1)
        left = torch.linalg.norm(action[..., 7:10], dim=-1)
        return torch.stack((right, left), dim=-1)

    def segment(
        self,
        min_len: int,
        distance_based: bool,
        gripper_based: bool,
        velocity_based: bool,
        distance_threshold: float,
        repeat_first_step: int,
        repeat_final_step: int,
        fix_frames: bool,
        min_end_distance: int,
        velocity_threshold: float,
        gripper_threshold: float,
        max_idx_dist: int,
    ) -> tuple[Any, ...]:
        if not velocity_based or distance_based or gripper_based:
            raise NotImplementedError("Bimanual segmentation currently uses velocity.")

        speeds = self.get_action_magnitude(subsampled=False, position_only=True)
        indices_per_traj = _either_arm_slow_indices(
            speeds,
            min_len=min_len,
            velocity_threshold=velocity_threshold,
            min_end_distance=min_end_distance,
            max_idx_dist=max_idx_dist,
        )

        start_idcs = tuple((0,) + indices for indices in indices_per_traj)
        stop_idcs = tuple(
            indices + (stop,) for indices, stop in zip(indices_per_traj, self.traj_lens)
        )
        no_segments = len(start_idcs[0])

        return tuple(
            BimanualDemosSegment(
                self,
                tuple(t[i] for t in start_idcs),
                tuple(t[i] for t in stop_idcs),
                repeat_first_step=repeat_first_step,
                repeat_final_step=repeat_final_step,
                segment_no=i,
                segments_total=no_segments,
            )
            for i in range(no_segments)
        )


class BimanualDemos(_BimanualDemosMixin, Demos):
    def __init__(
        self,
        trajectories: list[SceneObservation],
        meta_data: dict | None = None,
        add_init_ee_pose_as_frame: bool = True,
        add_world_frame: bool = False,
        make_quats_continuous: bool = True,
    ):
        self.meta_data = {} if meta_data is None else meta_data
        self.meta_data["add_init_ee_pose_as_frame"] = add_init_ee_pose_as_frame
        self.meta_data["add_world_frame"] = add_world_frame
        self.meta_data["bimanual"] = True

        self.n_trajs = len(trajectories)
        self.actions_raw = tuple(obs.action for obs in trajectories)
        self.ee_poses_raw = tuple(obs.ee_pose for obs in trajectories)
        self.gripper_actions = tuple(obs.action[:, [6, 13]] for obs in trajectories)
        self.gripper_states = tuple(obs.gripper_state for obs in trajectories)

        self.ee_poses = []
        self.ee_poses_vel = []
        self.ee_quats = []
        self.ee_actions = []
        self.ee_actions_quats = []
        self.world2frames = []
        self.world2frames_velocities = []
        self.frames2world = []
        self.frames2world_velocities = []
        self.frame_quats = []

        frame_names = []

        for traj_idx, obs in enumerate(trajectories):
            right_pose = obs.ee_pose[:, :7]
            left_pose = obs.ee_pose[:, 7:14]
            ee_pose = torch.stack((right_pose, left_pose), dim=1)
            ee_tf = _pose7_to_transform(ee_pose)
            self.ee_poses.append(ee_tf)

            ee_vel = ee_tf.clone()
            ee_vel[..., :3, 3] = 0
            self.ee_poses_vel.append(ee_vel)

            ee_quat = standardize_quaternion(ee_pose[..., 3:])
            self.ee_quats.append(ee_quat)

            action = obs.action
            right_action = action[:, :7]
            left_action = action[:, 7:14]
            action_tf = torch.stack(
                (_action_to_transform(right_action), _action_to_transform(left_action)),
                dim=1,
            )
            self.ee_actions.append(action_tf)
            self.ee_actions_quats.append(
                torch.stack(
                    (
                        axis_angle_to_quaternion(right_action[:, 3:6]),
                        axis_angle_to_quaternion(left_action[:, 3:6]),
                    ),
                    dim=1,
                )
            )

            frames = []
            names = []
            if add_world_frame:
                identity = torch.zeros_like(right_pose)
                identity[:, 3:] = identity_quaternions((right_pose.shape[0],))
                frames.append(identity)
                names.append("world")
            if add_init_ee_pose_as_frame:
                frames.extend(
                    (
                        right_pose[0].unsqueeze(0).repeat(right_pose.shape[0], 1),
                        left_pose[0].unsqueeze(0).repeat(left_pose.shape[0], 1),
                    )
                )
                names.extend(("right_ee_init", "left_ee_init"))

            object_items = list(obs.object_poses.items())
            frames.extend(value for _, value in object_items)
            names.extend(key for key, _ in object_items)

            if traj_idx == 0:
                frame_names = names

            frame_poses = torch.stack(frames)
            frame_pos = frame_poses[..., :3].reshape(-1, 3)
            frame_quat = standardize_quaternion(frame_poses[..., 3:].reshape(-1, 4))
            frame_rot = torch.Tensor(quaternion_to_matrix(frame_quat))

            frame2world = get_frame_transform_flat(
                frame_rot, frame_pos, invert=False
            ).reshape(frame_poses.shape[0], frame_poses.shape[1], 4, 4)
            world2frame = invert_homogenous_transform(frame2world)

            frame2world_vel = frame2world.clone()
            frame2world_vel[..., :3, 3] = 0
            world2frame_vel = invert_homogenous_transform(frame2world_vel)

            self.frames2world.append(frame2world)
            self.world2frames.append(world2frame)
            self.frames2world_velocities.append(frame2world_vel)
            self.world2frames_velocities.append(world2frame_vel)
            self.frame_quats.append(frame_quat.reshape(frame_poses.shape[0], -1, 4))

        self.frame_names = tuple(frame_names)
        self.n_frames = len(self.frame_names)
        self._ee_frame_idx = 0 if add_init_ee_pose_as_frame else None

        self.traj_lens = tuple(t.shape[0] for t in self.ee_poses)
        self.min_traj_len = min(self.traj_lens)
        self.max_traj_len = max(self.traj_lens)
        self.mean_traj_len = int(np.mean(self.traj_lens))

        self.subsample_to_common_length()
        self.stacked_actions_raw = self._subsample(self.actions_raw, dim=0)

        self.relative_start_time = 0
        self.relative_stop_time = 1
        self.relative_duration = 1


def _action_to_transform(action: torch.Tensor) -> torch.Tensor:
    rot = torch.Tensor(axis_angle_to_matrix(action[:, 3:6]))
    return get_frame_transform_flat(rot, action[:, :3], invert=False)


class BimanualDemosSegment(_BimanualDemosMixin, DemosSegment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.actions_raw = self._get_indexed(self.full_demos.actions_raw, 0)
        self.stacked_actions_raw = self._subsample(self.actions_raw, dim=0)


class BimanualPartialFrameView(_BimanualDemosMixin, PartialFrameViewDemos):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.actions_raw = self.full_demos.actions_raw
        self.stacked_actions_raw = self.full_demos.stacked_actions_raw
