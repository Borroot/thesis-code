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
        config = copy.deepcopy(config)
        backend_class = config.pop("backend_class", LightSimBackend)
        backend_kwargs = config.pop("backend_kwargs", {})
        backend = backend_class(**backend_kwargs)
        env_name = config.pop("env_name")
        obs_tokeep = config.pop("obs_tokeep", copy.deepcopy(ALL_ATTR_OBS))
        act_tokeep = config.pop("act_tokeep", copy.deepcopy(ALL_ATTR_ACT_DISCRETE))
        self.mask_checkpoint = config.pop("mask_model", None)

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

        # 5. initialize a mask checkpoint always returning 1.0 if none provided
        if self.mask_checkpoint is None:
            self.mask_checkpoint = lambda _obs: np.ones(
                self.action_space.n, dtype=np.float32
            )

    def reset(self, *, seed=None, options=None):
        obs, info = self.gym_env.reset(seed=seed, options=options)
        self.action_mask = self.mask_checkpoint(obs)
        obs = {"observations": obs, "action_mask": self.action_mask}
        return obs, info

    def step(self, action):
        if not self.action_mask[action]:
            raise ValueError("Invalid action (masked)")

        obs, reward, terminated, truncated, info = self.gym_env.step(action)
        self.action_mask = self.mask_checkpoint(obs)
        obs = {"observations": obs, "action_mask": self.action_mask}
        return obs, reward, terminated, truncated, info
