#!/usr/bin/env python3
from pathlib import Path

from tapas_gmm.dataset.demos import Demos
from tapas_gmm.policy.models.tpgmm import (
    AutoTPGMM,
    AutoTPGMMConfig,
    DemoSegmentationConfig,
    FittingStage,
    TPGMMConfig,
)
from tapas_gmm.utils.misc import DataNamingConfig, load_scene_data


def main() -> None:
    task = "PushButton"
    out_path = Path("../own_code") / "pushbutton_gmm_policy.pt"

    scene_data = load_scene_data(
        DataNamingConfig(
            feedback_type="demos",
            task=task,
            data_root="data",
        )
    )
    observations = scene_data.get_demos()

    demos = Demos(trajectories=observations)

    model = AutoTPGMM(
        AutoTPGMMConfig(
            tpgmm=TPGMMConfig(add_time_component=True),
            demos_segmentation=DemoSegmentationConfig(
                distance_based=False,
                velocity_based=True,
            )
        )
    )

    model.fit_trajectories(
        demos,
        fitting_actions=(FittingStage.INIT, FittingStage.EM_HMM),
    )
    model.to_disk(out_path)


if __name__ == "__main__":
    main()
