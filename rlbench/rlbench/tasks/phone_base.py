from typing import List
import numpy as np

from pyrep.objects.shape import Shape

from rlbench.backend.conditions import ImpossibleCondition
from rlbench.backend.task import Task


class PhoneBase(Task):

    def init_task(self) -> None:
        self.phone_case = Shape('phone_case')
        self.register_success_conditions([
            ImpossibleCondition()
        ])

    def init_episode(self, index: int) -> List[str]:
        return ['play around with the base']

    def variation_count(self) -> int:
        return 1

    def get_low_dim_state(self) -> np.ndarray:
        # return ground truth phone pose for ground truth keypoints
        return np.array([self.phone_case.get_pose()])
