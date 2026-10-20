# Bimanual Imitation Learning: TAPAS and Diffusion Policies

This is my study project on learning two-arm robot manipulation from demonstrations. I extended TAPAS for bimanual tasks and compared it with joint and leader-follower diffusion policies.

The experiments use four tasks in RLBench: pushing buttons, handing over an object, putting a bottle in a fridge and taking a tray out of an oven. I compare success rates, training time and robustness, and also look at segmentation and replanning.

## Main folders

- `own_code/demos/` – demonstrations for the four tasks.
- `own_code/training/` – notebooks for training the models.
- `own_code/evaluation/` – notebooks for running the evaluations.
- `own_code/results/` – saved evaluation results.
- `TAPAS/` – TAPAS code with the bimanual extensions and diffusion policies.
- `rlbench/` – simulation environment and task code.

The trained models are available in the [Evaluation models release](https://github.com/Nils19024/Study-Project/releases/tag/evaluation-models).

## Based on

- `TAPAS/` is based on [TAPAS](https://github.com/robot-learning-freiburg/TAPAS), which I extended for bimanual manipulation.
- `rlbench/` is based on [Markus Grotz's RLBench fork](https://github.com/markusgrotz/RLBench), which adds bimanual support to [RLBench](https://github.com/stepjam/RLBench). I made further changes for my experiments.
