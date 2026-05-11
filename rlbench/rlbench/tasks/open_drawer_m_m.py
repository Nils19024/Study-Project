from typing import List, Tuple
import numpy as np
from pyrep.objects.dummy import Dummy
from pyrep.objects.joint import Joint
from rlbench.backend.conditions import JointCondition, OrConditions
from rlbench.backend.task import Task


class OpenDrawerMM(Task):

    def init_task(self) -> None:
        self._options = ['bottom', 'middle', 'top']
        self._anchors = [Dummy('waypoint_anchor_%s' % opt)
                         for opt in self._options]
        self._joints = [Joint('drawer_joint_%s' % opt)
                        for opt in self._options]
        self._waypoint1 = Dummy('waypoint1')

    def init_episode(self, index: int) -> List[str]:
        # index = np.random.randint(0, 3)
        # self.variation_index = index
        self._current_index = index
        option = self._options[index]
        self._waypoint1.set_position(self._anchors[index].get_position())
        self.joint_conditions = [JointCondition(self._joints[i], 0.15) for i in range(3)]
        self.register_success_conditions(
            [OrConditions(self.joint_conditions)]
        )
        return ['open %s drawer' % option,
                'grip the %s handle and pull the %s drawer open' % (
                    option, option),
                'slide the %s drawer open' % option]
    
    def get_mode_if_applicable(self):
        conditions_met = [c.condition_met()[0] for c in self.joint_conditions]
        # only one condition should be met at a time
        if sum(conditions_met) > 1:
            print(f"\n WARNING: More than one condition met: {conditions_met} \n")
        return conditions_met.index(True) if any(conditions_met) else None

    def variation_count(self) -> int:
        return 3

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0, 0, - np.pi / 8], [0, 0, np.pi / 8]

    def get_low_dim_state(self) -> np.ndarray:
        shapes = [self._joints[0]]
        states = [s.get_pose() for s in shapes]
        return np.concatenate(states)
