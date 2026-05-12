from typing import List
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.conditions import DetectedCondition
from rlbench.backend.task import Task


class ArmScan(Task):

    def init_task(self) -> None:
        success_sensor = ProximitySensor('success')
        self.register_success_conditions(
            [DetectedCondition(self.robot.arm.get_tip(), success_sensor)])

    def init_episode(self, index: int) -> List[str]:
        return ['just move the arm around to collect some scans']

    def variation_count(self) -> int:
        return 1
