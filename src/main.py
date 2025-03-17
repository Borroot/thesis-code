import os
from pprint import pprint

import grid2op
from grid2op.Chronics import MultifolderWithCache
from grid2op.Reward import LinesCapacityReward
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig

from env import Env

# Setup storage directories

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_path, "models")
env_path = os.path.join(base_path, "envs")

os.makedirs(model_path, exist_ok=True)
os.makedirs(env_path, exist_ok=True)

# TODO download the env once in $HOME and move it to the cluster nodes
grid2op.change_local_dir(env_path)  # change where to store the environment

# Train and evaluate the agent

config = (
    PPOConfig()
    .env_runners(num_env_runners=8)
    .environment(
        env=Env,
        env_config={
            "env_name": "l2rpn_case14_sandbox",
            "reward_class": LinesCapacityReward,
            "chronics_class": MultifolderWithCache,
        },
        action_mask_key=None,
    )
    .training(lr=0.9999, num_epochs=1, minibatch_size=32)
    .evaluation(
        evaluation_interval=1,
        evaluation_duration=10,
        evaluation_parallel_to_training=False,
    )
)
algo = config.build_algo()

max_iter = 2
for i in range(max_iter):
    pprint(algo.train())
    algo.save_to_path(path=os.path.join(model_path, f"{i:03d}"))

# algo = Algorithm.from_checkpoint(path=os.path.join(model_path, f"{1 - 1:03d}"))
# pprint(algo.train())
