import copy

import grid2op
import gymnasium as gym
import numpy as np
from grid2op.gym_compat import BoxGymObsSpace, DiscreteActSpaceGymnasium, GymEnv
from grid2op.gym_compat.box_gym_obsspace import ALL_ATTR_OBS
from gymnasium.spaces import Box, Dict
from lightsim2grid import LightSimBackend

from utils import remove_invalid_actions

ALL_ATTR_ACT_DISCRETE = (
    "set_line_status",
    "change_line_status",
    "set_bus",
    "change_bus",
)


class Env(gym.Env):
    """Environment usable from rllib, mapping a grid2op environment."""

    def __init__(self, config=None):
        backend_class = config.pop("backend_class", LightSimBackend)
        backend_kwargs = config.pop("backend_kwargs", {})
        backend = backend_class(**backend_kwargs)
        env_name = config.pop("env_name")
        obs_tokeep = config.pop("obs_tokeep", copy.deepcopy(ALL_ATTR_OBS))
        act_tokeep = config.pop("act_tokeep", copy.deepcopy(ALL_ATTR_ACT_DISCRETE))

        # 1. create the grid2op environment
        self.g2p_env = grid2op.make(env_name, backend=backend, **config)
        self.g2p_env.chronics_handler.real_data.reset()
        act_tokeep = remove_invalid_actions(self.g2p_env, act_tokeep)

        # 2. create the gym environment
        self.gym_env = GymEnv(self.g2p_env)

        # 3. customize action space and observation space
        # action space
        self.gym_env.action_space.close()
        self.gym_env.action_space = DiscreteActSpaceGymnasium(
            self.g2p_env.action_space, attr_to_keep=act_tokeep
        )
        # observation space
        self.gym_env.observation_space.close()
        self.gym_env.observation_space = BoxGymObsSpace(
            self.g2p_env.observation_space, attr_to_keep=obs_tokeep
        )

        # 4. set the observation space and action space
        self.action_space = self.gym_env.action_space
        self.observation_space = Dict(
            {
                "action_mask": Box(0.0, 1.0, shape=(self.action_space.n,)),
                "observations": self.gym_env.observation_space,
            }
        )

        # TODO add action space masking
        # https://github.com/ray-project/ray/blob/master/rllib/examples/envs/classes/action_mask_env.py
        # https://github.com/ray-project/ray/blob/84d17cad835665631c2f68f6fb332e2973028fab/rllib/examples/action_masking.py

    def reset(self, *, seed=None, options=None):
        obs, info = self.gym_env.reset(seed=seed, options=options)

        action_mask = np.ones(self.action_space.n)
        action_mask[::2] = 0.0

        # Create a dictionary with the observation and action mask
        obs = {"observations": obs, "action_mask": action_mask}
        return obs, info

    def step(self, action):
        if action % 2 == 0:
            print("invalid", action)
            raise ValueError("Invalid action")

        print("valid", action)
        obs, reward, terminated, truncated, info = self.gym_env.step(action)

        action_mask = np.ones(self.action_space.n)
        action_mask[::2] = 0.0

        # Create a dictionary with the observation and action mask
        obs = {"observations": obs, "action_mask": action_mask}
        return obs, reward, terminated, truncated, info
