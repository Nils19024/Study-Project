from typing import List
import numpy as np
from pyrep.const import PrimitiveShape
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.objects.cartesian_path import CartesianPath
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from rlbench.backend.conditions import EmptyCondition
from rlbench.backend.spawn_boundary import SpawnBoundary

DIRT_POINTS = 50


class WipeDeskMM(Task):

    def init_task(self) -> None:
        self.dirt_spots = []
        self.sponge = Shape('sponge')
        self.sensor = ProximitySensor('sponge_sensor')
        self.register_graspable_objects([self.sponge])

        boundaries = [Shape('dirt_boundary')]
        _, _, self.z_boundary = boundaries[0].get_position()
        self.b = SpawnBoundary(boundaries)

    def init_episode(self, index: int) -> List[str]:
        # mode = np.random.randint(0, 2)
        mode = index
        if mode == 1:
            wp2 = Dummy('waypoint2')
            wp4 = Dummy('waypoint4')
            pose2 = wp2.get_position()
            pose4 = wp4.get_position()
            wp2.set_position(pose4)
            wp4.set_position(pose2)
            wp3 = CartesianPath('waypoint3')

            reversed_path = CartesianPath.create(path_color=[0, 0, 1], automatic_orientation=False)
            reversed_path.set_name('waypoint3overwrite')
            reversed_path.set_parent(wp3.get_parent())
            reversed_path.set_orientation(wp3.get_orientation())

            num_samples = 50
            sampled_poses = []
            for i in range(num_samples + 1):
                rel_dist = i / num_samples
                pos, ori = wp3.get_pose_on_path(rel_dist)
                sampled_poses.append(pos + ori)
            reversed_poses = sampled_poses[::-1]
            reversed_path.insert_control_points(reversed_poses)

            self.reversed_path = reversed_path
        else:
            self.reversed_path = None

            
        self._place_dirt()
        self.register_success_conditions([EmptyCondition(self.dirt_spots)])
        return ['wipe dirt off the desk',
                'use the sponge to clean up the desk',
                'remove the dirt from the desk',
                'grip the sponge and wipe it back and forth over any dirt you '
                'see',
                'clean up the mess',
                'wipe the dirt up']

    def variation_count(self) -> int:
        return 1

    def step(self) -> None:
        for d in self.dirt_spots:
            if self.sensor.is_detected(d):
                self.dirt_spots.remove(d)
                d.remove()

    def cleanup(self) -> None:
        for d in self.dirt_spots:
            d.remove()
        self.dirt_spots = []

        if self.reversed_path:
            self.reversed_path.remove()

    def _place_dirt(self):
        for i in range(DIRT_POINTS):
            spot = Shape.create(type=PrimitiveShape.CUBOID,
                                size=[.005, .005, .001],
                                mass=0, static=True, respondable=False,
                                renderable=True,
                                color=[0.58, 0.29, 0.0])
            spot.set_parent(self.get_base())
            spot.set_position([-1, -1, self.z_boundary + 0.001])
            self.b.sample(spot, min_distance=0.00,
                          min_rotation=(0.00, 0.00, 0.00),
                          max_rotation=(0.00, 0.00, 0.00))
            self.dirt_spots.append(spot)
        self.b.clear()

    def get_low_dim_state(self) -> np.ndarray:
        shapes = [self.sponge]  # + self.dirt_spots
        states = [s.get_pose() for s in shapes]
        return np.concatenate(states)