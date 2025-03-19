import argparse
import os
from pprint import pprint

import grid2op
import numpy as np
import ray
from grid2op.Chronics import MultifolderWithCache
from grid2op.Reward import LinesCapacityReward
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.examples.rl_modules.classes.action_masking_rlm import (
    ActionMaskingTorchRLModule,
)

from env import Env

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ArgumentParser Examples")

    # # 1. Positional argument
    # parser.add_argument('name', type=str, help='Your name')

    # # 2. Optional argument with default value
    # parser.add_argument('--age', type=int, default=25, help='Your age (default: 25)')

    # # 3. Flag argument (store_true or store_false)
    # parser.add_argument('--verbose', action='store_true', help='Increase output verbosity')

    # # 4. List of values (nargs='*' allows 0 or more values)
    # parser.add_argument('--colors', nargs='*', help='List of favorite colors')

    # # 5. Argument with a specific range (choices)
    # parser.add_argument('--size', choices=['small', 'medium', 'large'], help='Choose a size')

    # # 6. Multiple values for a single argument (nargs='+')
    # parser.add_argument('--numbers', nargs='+', type=int, help='List of numbers')

    # # 7. File argument (type=str)
    # parser.add_argument('--output', type=str, help='Output file name')

    # # 8. Argument that stores a constant value (store_const)
    # parser.add_argument('--debug', dest='debug', action='store_const', const=True, help='Enable debugging')

    args = parser.parse_args()

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
            num_env_runners=16,  # number of CPUs, NOT number of threads
            num_envs_per_env_runner=1,
            num_cpus_per_env_runner=1,
            num_gpus_per_env_runner=0,
        )
        .learners(
            # 0 means training takes place on a local learner on main process
            # CPUs or 1 GPU determined by num_gpus_per_learner
            num_learners=0,
            num_cpus_per_learner=0,
            num_gpus_per_learner=1,  # can be fractional
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
            lr=3e-6, gamma=0.999, clip_param=0.2, num_epochs=10, minibatch_size=16
        )
        .evaluation(
            evaluation_interval=1,
            evaluation_duration=10,
            evaluation_parallel_to_training=False,
        )
    ).build_algo()

    max_iter = 3
    for i in range(max_iter):
        pprint(algo.train())
        algo.save_to_path(path=os.path.join(model_path, f"{i:03d}"))

    # algo = Algorithm.from_checkpoint(path=os.path.join(model_path, f"{1 - 1:03d}"))
    # pprint(algo.train())
