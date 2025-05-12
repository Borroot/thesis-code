import copy
import os
import sys
import time

import grid2op
import numpy as np
from env import Env
from grid2op.Chronics import MultifolderWithCache
from grid2op.Reward import LinesCapacityReward
from lightsim2grid import LightSimBackend
from mask import MaskModel


def time_obs_batch(env, reps=1):
    """Time mask model observation batch generation."""
    env.reset()
    times = []

    # Time the duration to get one observation batch
    for _ in range(reps):
        start = time.time()
        for action in range(env.act_dim):
            env_copy = copy.deepcopy(env)
            env_copy.step(action)
        times.append(time.time() - start)

    times = np.array(times)
    print(f"obs_batch real: {times.mean():.6f} ±{times.std():.6f} (N={len(times)})")
    return times


def time_action(env, max_actions=np.inf):
    """Time performing an action in the environment."""
    times = []

    # Time the duration to perform one action
    for action in range(min(env.act_dim, max_actions)):
        env.reset()
        start = time.time()
        env.step(action)
        times.append(time.time() - start)

    times = np.array(times)
    print(f"action: {times.mean():.6f} ±{times.std():.6f} (N={len(times)})")
    return times


def time_env_copy(env, reps=1):
    """Time deep copying the environment."""
    times = []

    # Time the duration to copy the environment
    for _ in range(reps):
        env.reset()
        start = time.time()
        copy.deepcopy(env)
        times.append(time.time() - start)

    times = np.array(times)
    print(f"env_copy: {times.mean():.6f} ±{times.std():.6f} (N={len(times)})")
    return times


def time_mask_model(env, reps=1):
    mask_model = MaskModel(env.obs_dim, env.act_dim)
    mask_model.train(env, num_episodes=reps, fake_batch=True)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        grid2op.change_local_dir(sys.argv[1])

    env_names = [
        "l2rpn_case14_sandbox",
        "l2rpn_neurips_2020_track1_small",
        # "l2rpn_icaps_2021_small",
        "l2rpn_wcci_2022",
        # "l2rpn_idf_2023",
        # "l2rpn_neurips_2020_track2_small", # error
    ]

    for env_name in env_names:
        print(env_name)
        env = Env(
            {
                "env_name": env_name,
                "backend_class": LightSimBackend,
                "reward_class": LinesCapacityReward,
                "chronics_class": MultifolderWithCache,
            }
        )

        print("action space:", env.act_dim)
        print("observation space:", env.obs_dim)

        times_action = time_action(env, max_actions=10000)
        times_env_copy = time_env_copy(env, reps=30)

        obs_batch_estimate = env.act_dim * (times_action.mean() + times_env_copy.mean())
        print(f"obs_batch estimate: {obs_batch_estimate:.6f}")
        if obs_batch_estimate < 60:
            times_obs_batch = time_obs_batch(env, reps=30)
            time_mask_model(env, reps=100)
        else:
            time_mask_model(env, reps=10)

        print()
