import copy

import grid2op
import gymnasium as gym
import numpy as np
from grid2op.gym_compat import BoxGymObsSpace, DiscreteActSpaceGymnasium, GymEnv
from grid2op.gym_compat.box_gym_obsspace import ALL_ATTR_OBS
from gymnasium.spaces import Box, Dict
from lightsim2grid import LightSimBackend

ALL_ATTR_ACT_DISCRETE = (
    "set_line_status",
    "change_line_status",
    "set_bus",
    "change_bus",
)


class Env(gym.Env):
    """Environment usable from rllib, mapping a grid2op environment."""

    # TODO create a child class EnvMasked which applies a mask model

    def __init__(self, config=None):
        config = copy.deepcopy(config)

        # Configure the grid2op backend
        backend_class = config.pop("backend_class", LightSimBackend)
        backend_kwargs = config.pop("backend_kwargs", {})
        backend = backend_class(**backend_kwargs)

        # Configure the environment
        env_name = config.pop("env_name")
        self.obs_tokeep = config.pop("obs_tokeep", copy.deepcopy(ALL_ATTR_OBS))
        self.act_tokeep = config.pop("act_tokeep", copy.deepcopy(ALL_ATTR_ACT_DISCRETE))

        # Configure the mask model, the default model masks no actions
        default_mask_model = lambda _obs: np.ones(self.action_space.n, dtype=np.float32)
        self.mask_model = config.pop("mask_model", default_mask_model)
        self.mask_model = (
            default_mask_model if self.mask_model is None else self.mask_model
        )

        # Create the grid2op environment
        self.g2p_env = grid2op.make(env_name, backend=backend, **config)
        self.g2p_env.chronics_handler.reset()
        self.act_tokeep = self._remove_invalid_actions(self.g2p_env, self.act_tokeep)

        # Create the gym environment
        self._setup_gym_env()

        # Set the observation space and action space
        self.action_space = self.gym_env.action_space
        self.observation_space = Dict(
            {
                "action_mask": Box(0.0, 1.0, shape=(self.action_space.n,)),
                "observations": self.gym_env.observation_space,
            }
        )

    def _remove_invalid_actions(self, g2p_env, act_tokeep):
        """Filter out actions which are not usable in the environment."""
        return [act for act in act_tokeep if g2p_env.action_space.supports_type(act)]

    def _setup_gym_env(self):
        # Create the gym environment
        self.gym_env = GymEnv(self.g2p_env)

        # Setup observation space
        self.gym_env.observation_space.close()
        self.gym_env.observation_space = BoxGymObsSpace(
            self.g2p_env.observation_space, attr_to_keep=self.obs_tokeep
        )
        # Setup action space
        self.gym_env.action_space.close()
        self.gym_env.action_space = DiscreteActSpaceGymnasium(
            self.g2p_env.action_space, attr_to_keep=self.act_tokeep
        )

    def reset(self, *, seed=None, options=None):
        # Reset the environment and compute the initial action mask
        obs, info = self.gym_env.reset(seed=seed, options=options)
        self.action_mask = self.mask_model(obs)
        obs = {"observations": obs, "action_mask": self.action_mask}
        return obs, info

    def step(self, action):
        # The step function should not be called with a masked action, instead
        # this should be filtered out using the action_mask provided in the
        # observation returned in the previous step/reset call.
        if not self.action_mask[action]:
            raise ValueError("Invalid action (masked)")

        # Step the environment and compute the new action mask
        obs, reward, terminated, truncated, info = self.gym_env.step(action)
        self.action_mask = self.mask_model(obs)
        obs = {"observations": obs, "action_mask": self.action_mask}
        return obs, reward, terminated, truncated, info
