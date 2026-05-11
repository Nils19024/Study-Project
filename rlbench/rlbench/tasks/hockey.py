from typing import List
import numpy as np
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition, GraspedCondition
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor


class Hockey(Task):

    def init_task(self) -> None:
        self.stick = Shape('hockey_stick')
        self.ball = Shape('hockey_ball')
        self.goal = ProximitySensor('success')
        self.register_success_conditions([
            DetectedCondition(self.ball, self.goal),
            GraspedCondition(self.robot.gripper, self.stick)])
        self.register_graspable_objects([self.stick])

    def init_episode(self, index: int) -> List[str]:
        return ['hit the ball into the net',
                'use the stick to push the hockey ball into the goal',
                'pick up the hockey stick, then swing at the ball in the '
                'direction of the net',
                'score a hockey goal',
                'grasping one end of the hockey stick, swing it such that the '
                'other end collides with the ball such that the ball goes '
                'into the goal']

    def variation_count(self) -> int:
        return 1

    def get_low_dim_state(self):
        objects = [self.stick, self.ball, self.goal]
        poses = [o.get_pose() for o in objects]
        return np.concatenate(poses)