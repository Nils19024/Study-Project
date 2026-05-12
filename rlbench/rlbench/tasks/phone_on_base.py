from typing import List
import numpy as np

from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape

from rlbench.backend.conditions import DetectedCondition, NothingGrasped
from rlbench.backend.task import Task


class PhoneOnBase(Task):

    def init_task(self) -> None:
        self.phone = Shape('phone')
        self.phone_case = Shape('phone_case')
        self.register_graspable_objects([self.phone])
        self.register_success_conditions([
            DetectedCondition(self.phone, ProximitySensor('success')),
            NothingGrasped(self.robot.gripper)
        ])

    def init_episode(self, index: int) -> List[str]:
        return ['put the phone on the base',
                'put the phone on the stand',
                'put the hone on the hub',
                'grasp the phone and put it on the base',
                'place the phone on the base',
                'put the phone back on the base']

    def variation_count(self) -> int:
        return 1

    def get_low_dim_state(self) -> np.ndarray:
        # return ground truth phone pose for ground truth keypoints
        return np.array([self.phone_case.get_pose(),
                         self.phone.get_pose()])
