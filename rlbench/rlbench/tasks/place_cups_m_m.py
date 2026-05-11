from typing import List, Tuple
import numpy as np
from pyrep.objects.dummy import Dummy
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.conditions import DetectedCondition, NothingGrasped, \
    OrConditions
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.backend.task import Task


class PlaceCupsMM(Task):

    def init_task(self) -> None:
        self._cups = [Shape('mug%d' % i) for i in range(3)]
        self._spokes = [Shape('place_cups_holder_spoke%d' % i) for i in
                        range(3)]
        self._cups_boundary = Shape('mug_boundary')
        self._w1 = Dummy('waypoint1')
        self._w4 = Dummy('waypoint4')
        success_detectors = [
            ProximitySensor('success_detector%d' % i) for i in range(3)]
        # self._on_peg_conditions = [OrConditions([
        #     DetectedCondition(self._cups[ci], success_detectors[sdi]) for sdi in
        #     range(3)]) for ci in range(3)]
        # The original code was checking whether each cup was on any spoke.
        # But we want the multimodality wrt the spokes, so instead we check
        # whether any cup is on the spoke corresponding to the current index.
        self._on_peg_conditions = [OrConditions([
            DetectedCondition(self._cups[ci], success_detectors[sdi]) for ci in range(3)
        ]) for sdi in range(3)]
        self.register_graspable_objects(self._cups)
        self._initial_relative_cup = self._w1.get_pose(self._cups[0])
        self._initial_relative_spoke = self._w4.get_pose(self._spokes[0])

    def init_episode(self, index: int) -> List[str]:
        self._free_spokes = list(range(3))
        self._cups_placed = 0
        self._next_spoke = index
        # print('next spoke:', self._next_spoke)
        self._index = 0  # index is used for the number of cups to place
        b = SpawnBoundary([self._cups_boundary])
        [b.sample(c, min_distance=0.14) for c in self._cups]
        # success_conditions = [NothingGrasped(self.robot.gripper)
        #                       ] + self._on_peg_conditions[:index + 1]
        success_conditions = [
            NothingGrasped(self.robot.gripper),
            OrConditions(self._on_peg_conditions)] 
        self.register_success_conditions(success_conditions)
        self.register_waypoint_ability_start(
            0, self._move_above_next_target)
        self.register_waypoints_should_repeat(self._repeat)

        if index == 0:
            return ['place 1 cup on the cup holder',
                    'pick up one cup and put it on the mug tree',
                    'move 1 mug from the table to the cup holder',
                    'pick up one cup and slide its handle onto a spoke on the '
                    'mug holder']
        else:
            return ['place %d cups on the cup holder' % (index + 1),
                    'pick up %d cups and place them on the holder'
                    % (index + 1),
                    'move %d cups from the table to the mug tree'
                    % (index + 1),
                    'pick up %d mugs and slide their handles onto the cup '
                    'holder spokes' % (index + 1)]

    def variation_count(self) -> int:
        return 3

    def get_mode_if_applicable(self):
        conditions_met = [c.condition_met()[0] for c in
                          self._on_peg_conditions]
        # only one condition should be met at a time
        if sum(conditions_met) > 1:
            print(f"\n WARNING: More than one condition met: {conditions_met} \n")
        return conditions_met.index(True) if any(conditions_met) else None


    def _move_above_next_target(self, waypoint):
        self._w1.set_parent(self._cups[self._cups_placed])
        self._w4.set_pose(
            self._initial_relative_spoke,
            relative_to=self._spokes[self._next_spoke])
        self._w1.set_pose(
            self._initial_relative_cup,
            relative_to=self._cups[self._cups_placed])
        self._cups_placed += 1
        self._free_spokes.remove(self._next_spoke)
        self._next_spoke = np.random.choice(self._free_spokes)

    def _repeat(self):
        return self._cups_placed < self._index + 1

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0.0, 0.0, -np.pi / 2], [0.0, 0.0, np.pi / 2]

    def get_low_dim_state(self) -> np.ndarray:
        shapes = self._cups + [self._spokes[0]]
        states = [s.get_pose() for s in shapes]
        return np.concatenate(states)

