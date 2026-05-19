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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_ROOT = PROJECT_ROOT / "own_code" / "outputs"


def main() -> None:
    task = "PushButton"
    out_path = OUTPUTS_ROOT / "unimanual" / "pushbutton_gmm_policy.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    scene_data = load_scene_data(
        DataNamingConfig(
            feedback_type="demos",
            task=task,
            data_root=str(OUTPUTS_ROOT / "unimanual" / "tapas_observations"),
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
