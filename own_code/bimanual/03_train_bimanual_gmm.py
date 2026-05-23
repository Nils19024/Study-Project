#!/usr/bin/env python3
from pathlib import Path

from tapas_gmm.dataset.bimanual_demos import BimanualDemos
from tapas_gmm.policy.models.tpgmm import (
    AutoTPGMM,
    AutoTPGMMConfig,
    CascadeConfig,
    DemoSegmentationConfig,
    FittingStage,
    FrameSelectionConfig,
    InitStrategy,
    ModelType,
    TPGMMConfig,
)
from tapas_gmm.utils.misc import DataNamingConfig, load_scene_data

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_ROOT = PROJECT_ROOT / "own_code" / "outputs"


def load_bimanual_demos() -> BimanualDemos:
    task = "BimanualDualPushButtons"
    data_root = OUTPUTS_ROOT / "bimanual" / "tapas_observations"

    scene_data = load_scene_data(
        DataNamingConfig(
            feedback_type="demos",
            task=task,
            data_root=str(data_root),
        )
    )

    # Bimanual demos use left/right wrist cameras, but GMM training only needs low-dim.
    observations = [scene_data._get_bc_traj(i, cams=()) for i in range(len(scene_data))]

    return BimanualDemos(
        trajectories=observations,
        meta_data={"data_root": str(data_root), "task": task},
        add_init_ee_pose_as_frame=True,
        add_world_frame=False,
    )


def make_model() -> AutoTPGMM:
    tpgmm_config = TPGMMConfig(
        n_components=10,
        model_type=ModelType.HMM,
        use_riemann=True,
        add_time_component=True,
        add_action_component=False,
        position_only=True,
        add_gripper_action=True,
        reg_shrink=1e-2,
        reg_diag=2e-4,
        reg_diag_gripper=2e-2,
        reg_em_finish_shrink=1e-2,
        reg_em_finish_diag=2e-4,
        reg_em_finish_diag_gripper=2e-2,
        trans_cov_mask_t_pos_corr=False,
        em_steps=1,
        fix_first_component=True,
        fix_last_component=True,
        reg_init_diag=5e-4,
        heal_time_variance=False,
    )

    frame_selection_config = FrameSelectionConfig(
        init_strategy=InitStrategy.TIME_BASED,
        fitting_actions=(FittingStage.INIT,),
        rel_score_threshold=0.5,
        use_bic=False,
        drop_redundant_frames=False,
    )

    demos_segmentation_config = DemoSegmentationConfig(
        gripper_based=False,
        distance_based=False,
        velocity_based=True,
        repeat_final_step=0,
        repeat_first_step=0,
        components_prop_to_len=True,
        min_n_components=3,
    )

    cascade_config = CascadeConfig(
        kl_keep_time_dim=True,
        kl_keep_rotation_dim=False,
    )

    return AutoTPGMM(
        AutoTPGMMConfig(
            tpgmm=tpgmm_config,
            frame_selection=frame_selection_config,
            demos_segmentation=demos_segmentation_config,
            cascade=cascade_config,
        )
    )


def main() -> None:
    out_path = OUTPUTS_ROOT / "bimanual" / "bimanual_gmm_policy.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    demos = load_bimanual_demos()
    model = make_model()

    model.fit_trajectories(
        demos,
        fix_frames=True,
        init_strategy=InitStrategy.TIME_BASED,
        fitting_actions=(FittingStage.INIT,),
    )
    model.fit_trajectories(
        demos,
        fix_frames=True,
        fitting_actions=(FittingStage.EM_HMM,),
    )
    model.to_disk(out_path)

    print(f"Saved bimanual GMM policy to: {out_path}")


if __name__ == "__main__":
    main()
