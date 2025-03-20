import argparse
import json
import os


def process_paths(args):
    """
    Process the paths into absolute paths. Any relative path is assumed to be
    relative to the project root.
    """
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for key in ["path_model_save", "path_model_load", "path_env"]:
        if getattr(args, key) is not None and not os.path.isabs(getattr(args, key)):
            setattr(args, key, os.path.join(base_path, getattr(args, key)))
    return args


def load_json_defaults(defaults_path):
    with open(defaults_path, "r") as f:
        defaults = json.load(f)
    return {key.replace("-", "_"): value for key, value in defaults.items()}


def commandline_arguments():
    parser = argparse.ArgumentParser(description="Train PPO on a Grid2Op environment")

    # Config parameters
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.json",
        help="Path to the default configuration JSON file",
    )

    # Environment parameters
    parser.add_argument(
        "--env-name",
        type=str,
        choices=[
            "l2rpn_case14_sandbox",
            "l2rpn_icaps_2021_small",
            "l2rpn_icaps_2021_large",
            "l2rpn_wcci_2020",
            "l2rpn_wcci_2022",
            "l2rpn_idf_2023",
        ],
        help="Name of the environment",
    )
    parser.add_argument(
        "--path-env", type=str, help="Path to save and load the environment"
    )

    # Model parameters
    parser.add_argument(
        "--path-model-save", type=str, help="Path to save the model checkpoints to"
    )
    parser.add_argument(
        "--path-model-load",
        type=str,
        help="Path to load a specific model checkpoint from",
    )

    # Environment runner parameters
    parser.add_argument(
        "--num-env-runners", type=int, help="Number of environment runners"
    )
    parser.add_argument(
        "--num-envs-per-env-runner",
        type=int,
        help="Number of environments per environment runner",
    )
    parser.add_argument(
        "--num-cpus-per-env-runner",
        type=int,
        help="Number of CPUs per environment runner",
    )

    # Learner parameters
    parser.add_argument(
        "--num-learners",
        type=int,
        help="Number of learners (0 means one local learner)",
    )
    parser.add_argument(
        "--num-gpus-per-learner",
        type=float,
        help="Number of GPUs per learner, can be fractional",
    )

    # Training parameters
    parser.add_argument("--epsilon", type=float, help="Clip parameter for PPO")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--gamma", type=float, help="Discount factor")
    parser.add_argument("--num-epochs", type=int, help="Number of epochs")
    parser.add_argument("--minibatch-size", type=int, help="Minibatch size")

    # Evaluation parameters
    parser.add_argument(
        "--evaluation-interval",
        type=int,
        help="Evaluation interval in iterations",
    )
    parser.add_argument(
        "--evaluation-duration",
        type=int,
        help="Evaluation duration in episodes",
    )
    parser.add_argument(
        "--evaluation-parallel-to-training",
        action="store_true",
        help="Run evaluation parallel to training",
    )

    # Load and set the defaults from the config file
    defaults = load_json_defaults(parser.get_default("config"))
    parser.set_defaults(**defaults)
    args = parser.parse_args()
    args = process_paths(args)
    return args
