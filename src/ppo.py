import json
import os
from pprint import pprint

import grid2op
import jsonpickle
import ray
from grid2op.Chronics import MultifolderWithCache
from grid2op.Reward import LinesCapacityReward
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.examples.rl_modules.classes.action_masking_rlm import (
    ActionMaskingTorchRLModule,
)

from args import commandline_arguments
from env import Env


def create_algo(args):
    """Create a new algorithm given the commandline arguments."""

    env_config = {
        "env_name": args.env_name,
        "reward_class": LinesCapacityReward,
        "chronics_class": MultifolderWithCache,
        "mask_model": None,
    }

    # TODO Implement a real mask model
    # Create a dummy mask model
    # env = Env(config=env_config)
    # action_space_size = env.action_space.n
    # env_config["mask_model"] = lambda _obs: np.random.choice(
    #     [0.0, 1.0], size=action_space_size
    # ).astype(np.float32)

    algo = (
        PPOConfig()
        .env_runners(
            num_env_runners=args.num_env_runners,
            num_envs_per_env_runner=args.num_envs_per_env_runner,
            num_cpus_per_env_runner=args.num_cpus_per_env_runner,
            num_gpus_per_env_runner=0,
            sample_timeout_s=None,  # TODO set a sensible value
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

    # save the settings to a JSON file
    with open(os.path.join(args.path_checkpoint_save, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    return algo


def load_algo(args):
    """Load an algorithm from a checkpoint."""
    algo = Algorithm.from_checkpoint(path=args.path_checkpoint_load)
    return algo


def save_checkpoint(algo, args):
    """Save the current state of the algorithm to a checkpoint."""
    algo.save_to_path(
        path=os.path.join(args.path_checkpoint_save, f"{algo.iteration - 1:06d}")
    )


def save_results(results, args):
    """Save the results of the training iteration to a JSON file."""
    path_results = os.path.join(
        args.path_checkpoint_save, f"{results['training_iteration'] - 1:06d}.json"
    )
    with open(path_results, "w") as f:
        f.write(jsonpickle.encode(results, indent=4))


if __name__ == "__main__":
    args = commandline_arguments()

    # Setup directories
    os.makedirs(args.path_checkpoint_save, exist_ok=True)
    os.makedirs(args.path_env, exist_ok=True)
    grid2op.change_local_dir(args.path_env)

    # Create or load an algorithm
    ray.init(include_dashboard=False)
    if args.path_checkpoint_load is None:
        algo = create_algo(args)
    else:
        algo = load_algo(args)

    # Train the RL agent
    while algo.iteration < args.num_iterations:
        results = algo.train()
        save_results(results, args)
        pprint(results)

        if algo.iteration % args.checkpoint_interval == 0:
            save_checkpoint(algo, args)

    save_checkpoint(algo, args)
