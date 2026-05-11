from typing import List
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.dummy import Dummy
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedSeveralCondition
from rlbench.backend.conditions import NothingGrasped
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.const import colors

MAX_STACKED_BLOCKS = 3
MIN_STACKED_BLOCKS = 3
DISTRACTORS = 4


class StackBlocks(Task):

    def init_task(self) -> None:
        self.blocks_stacked = 0
        self.target_blocks = [Shape('stack_blocks_target%d' % i)
                              for i in range(4)]
        self.distractors = [
            Shape('stack_blocks_distractor%d' % i)
            for i in range(DISTRACTORS)]

        self.boundaries = [Shape('stack_blocks_boundary%d' % i)
                           for i in range(4)]

        # Drop-off zone
        self.drop_off_zone = Shape('stack_blocks_target_plane')

        self.register_graspable_objects(self.target_blocks + self.distractors)

        self.register_waypoint_ability_start(0, self._move_above_next_target)
        self.register_waypoint_ability_start(3, self._move_above_drop_zone)
        self.register_waypoint_ability_start(5, self._is_last)
        self.register_waypoints_should_repeat(self._repeat)

    def init_episode(self, index: int) -> List[str]:
        # For each color, we want to have 2, 3 or 4 blocks stacked
        color_index = int(index / MAX_STACKED_BLOCKS)
        self.blocks_to_stack = MIN_STACKED_BLOCKS + index % MAX_STACKED_BLOCKS
        color_name, color_rgb = colors[color_index]
        for b in self.target_blocks:
            b.set_color(color_rgb)

        success_detector = ProximitySensor(
            'stack_blocks_success')
        self.register_success_conditions([DetectedSeveralCondition(
            self.target_blocks, success_detector, self.blocks_to_stack),
            NothingGrasped(self.robot.gripper)
        ])

        self.blocks_stacked = 0
        color_choices = np.random.choice(
            list(range(color_index)) + list(
                range(color_index + 1, len(colors))),
            size=2, replace=False)
        for i, ob in enumerate(self.distractors):
            name, rgb = colors[color_choices[int(i / 4)]]
            ob.set_color(rgb)
        b = SpawnBoundary(self.boundaries)
        for block in self.target_blocks + self.distractors:
            b.sample(block, min_distance=0.1)
            self._canonicalize_yaw_mod_90(block)

        return ['stack %d %s blocks' % (self.blocks_to_stack, color_name),
                'place %d of the %s cubes on top of each other'
                % (self.blocks_to_stack, color_name),
                'pick up and set down %d %s blocks on top of each other'
                % (self.blocks_to_stack, color_name),
                'build a tall tower out of %d %s cubes'
                % (self.blocks_to_stack, color_name),
                'arrange %d %s blocks in a vertical stack on the table top'
                % (self.blocks_to_stack, color_name),
                'set %d %s cubes on top of each other'
                % (self.blocks_to_stack, color_name)]

    def variation_count(self) -> int:
        return len(colors) * MAX_STACKED_BLOCKS

    def _move_above_next_target(self, _):
        if self.blocks_stacked >= self.blocks_to_stack:
            raise RuntimeError('Should not be here.')
        w2 = Dummy('waypoint1')
        x, y, z = self.target_blocks[self.blocks_stacked].get_position()
        _, _, oz = self.target_blocks[self.blocks_stacked].get_orientation()
        ox, oy, _ = w2.get_orientation()
        w2.set_position([x, y, z])
        w2.set_orientation([ox, oy, -oz])

    def _move_above_drop_zone(self, waypoint):
        x, y, z = self.drop_off_zone.get_position()
        waypoint.get_waypoint_object().set_position(
            [x, y, z + 0.08 + 0.06 * self.blocks_stacked])

    def _is_last(self, waypoint):
        last = self.blocks_stacked == self.blocks_to_stack - 1
        waypoint.skip = last

    def _repeat(self):
        self.blocks_stacked += 1
        return self.blocks_stacked < self.blocks_to_stack

    def _canonicalize_yaw_mod_90(self, shape):
        x_rot, y_rot, z_rot = shape.get_orientation()

        # modulo 90° (π/2)
        z_rot = np.mod(z_rot, np.pi / 2.0) + np.pi / 2.0

        # Set new orientation
        shape.set_orientation([x_rot, y_rot, z_rot])

    def get_low_dim_state(self) -> np.ndarray:
        shapes = self.target_blocks + self.distractors + [self.drop_off_zone]
        states = [s.get_pose() for s in shapes]
        return np.concatenate(states)
