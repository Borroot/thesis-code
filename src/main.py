import os
from pprint import pprint

import grid2op
import numpy as np
import ray
from args import commandline_arguments
from env import Env
from grid2op.Chronics import MultifolderWithCache
from grid2op.Reward import LinesCapacityReward
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.examples.rl_modules.classes.action_masking_rlm import (
    ActionMaskingTorchRLModule,
)

if __name__ == "__main__":
    args = commandline_arguments()

    # Setup storage directories

    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_path, "models")
    env_path = os.path.join(base_path, "envs")

    os.makedirs(model_path, exist_ok=True)
    os.makedirs(env_path, exist_ok=True)

    # TODO download the env once in $HOME and move it to the cluster nodes
    grid2op.change_local_dir(env_path)  # change where to store the environment

    # Train a mask model

    env_config = {
        # "env_name": "l2rpn_icaps_2021_small",  # causes RAM shortage
        "env_name": "l2rpn_case14_sandbox",
        "reward_class": LinesCapacityReward,
        "chronics_class": MultifolderWithCache,
        "mask_model": None,
    }
    env = Env(config=env_config)
    action_space_size = env.action_space.n
    env_config["mask_model"] = lambda _obs: np.random.choice(
        [0.0, 1.0], size=action_space_size
    ).astype(
        np.float32
    )  # TODO implement a real mask model

    # Train and evaluate the agent

    ray.init(include_dashboard=False)

    algo = (
        PPOConfig()
        .env_runners(
            num_env_runners=args.num_env_runners,
            num_envs_per_env_runner=args.num_envs_per_env_runner,
            num_cpus_per_env_runner=args.num_cpus_per_env_runner,
            num_gpus_per_env_runner=0,
        )
        .learners(
            # 0 means training takes place on a local learner on main process
            # CPUs or 1 GPU determined by num_gpus_per_learner
            num_learners=args.num_learners,
            num_cpus_per_learner=0,
            num_gpus_per_learner=args.num_gpus_per_learner,
        )
        .environment(
            env=Env,
            env_config=env_config,
            action_mask_key=None,
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=ActionMaskingTorchRLModule,
                model_config={
                    "head_fcnet_hiddens": [300, 300, 300],
                    "head_fcnet_activation": "relu",
                },
            )
        )
        .training(
            lr=args.learning_rate,
            gamma=args.gamma,
            clip_param=args.epsilon,
            num_epochs=args.num_epochs,
            minibatch_size=args.minibatch_size,
        )
        .evaluation(
            evaluation_interval=args.evaluation_interval,
            evaluation_duration=args.evaluation_duration,
            evaluation_parallel_to_training=args.evaluation_parallel_to_training,
        )
    ).build_algo()

    max_iter = 3
    for i in range(max_iter):
        pprint(algo.train())
        algo.save_to_path(path=os.path.join(model_path, f"{i:03d}"))

    # algo = Algorithm.from_checkpoint(path=os.path.join(model_path, f"{1 - 1:03d}"))
    # pprint(algo.train())
