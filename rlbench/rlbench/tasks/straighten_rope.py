from typing import List
import numpy as np
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from rlbench.backend.conditions import DetectedCondition
from rlbench.backend.task import Task


class StraightenRope(Task):

    def init_task(self) -> None:
        self._head = Shape('head')
        self._tail = Shape('tail')
        self._waypoint0 = Dummy('waypoint0')
        self._waypoint4 = Dummy('waypoint4')
        self._success_head = ProximitySensor('success_head')
        self._success_tail = ProximitySensor('success_tail')
        self.register_success_conditions(
            [DetectedCondition(self._head, self._success_head),
             DetectedCondition(self._tail, self._success_tail)])

    def init_episode(self, index: int) -> List[str]:
        return ['straighten rope',
                'pull the rope straight',
                'grasping each end of the rope in turn, leave the rope straight'
                ' on the table',
                'pull each end of the rope until is is straight',
                'tighten the rope',
                'pull the rope tight']

    def variation_count(self) -> int:
        return 1
    
    def get_low_dim_state(self) -> np.ndarray:
        shapes = [self._waypoint0, self._waypoint4, self._success_head, self._success_tail]
        states = [s.get_pose() for s in shapes]
        return np.concatenate(states)