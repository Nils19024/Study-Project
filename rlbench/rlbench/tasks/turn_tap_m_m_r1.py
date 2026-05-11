from typing import List

import numpy as np
from pyrep.objects.dummy import Dummy
from pyrep.objects.joint import Joint
from pyrep.objects.shape import Shape
from rlbench.backend.task import Task
from rlbench.backend.conditions import JointCondition, OrConditions

OPTIONS = ['left', 'right']


class TurnTapMMR1(Task):

    def init_task(self) -> None:
        self.left_start = Dummy('waypoint0')
        self.left_end = Dummy('waypoint1')
        self.right_start = Dummy('waypoint5')
        self.right_end = Dummy('waypoint6')
        self.left_joint = Joint('left_joint')
        self.right_joint = Joint('right_joint')

        self.tap = Shape('tap_main')

    def init_episode(self, index: int) -> List[str]:
        self.tap.set_orientation(
            # [0.0, 0.0, np.random.uniform(0.0, 1/2 * np.pi)]
            [0.0, 0.0,  np.pi]
        )
        self.tap.set_position(
            self.tap.get_position() + [0.25, 0.0, 0.0]
        )

        option = np.random.choice(OPTIONS)

        if option == 'right':
            self.left_start.set_position(self.right_start.get_position())
            self.left_start.set_orientation(self.right_start.get_orientation())
            self.left_end.set_position(self.right_end.get_position())
            self.left_end.set_orientation(self.right_end.get_orientation())

        joint_conditions = [
            JointCondition(self.right_joint, 1.57),
            JointCondition(self.left_joint, 1.57)]
        self.register_success_conditions(
            [OrConditions(joint_conditions)]
        )

        return ['turn %s tap' % option,
                'rotate the %s tap' % option,
                'grasp the %s tap and turn it' % option]

    def variation_count(self) -> int:
        return 2

    def get_low_dim_state(self) -> np.ndarray:
        shapes = [self.left_joint]
        states = [s.get_pose() for s in shapes]
        return np.concatenate(states)
