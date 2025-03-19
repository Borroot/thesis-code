import argparse
import json
import os


def load_json_defaults(defaults_path):
    with open(defaults_path, 'r') as f:
        return json.load(f)


def commandline_arguments():
    parser = argparse.ArgumentParser(description="ArgumentParser Examples")

    parser.add_argument(
        "--config", type=str, default="default_config.json", help="Path to the default configuration JSON file"
    )
    parser.add_argument(
        "--override-config", type=str, help="Path to an alternative configuration JSON file"
    )

    args, unknown = parser.parse_known_args()

    defaults = load_json_defaults(args.config)

    if args.override_config:
        overrides = load_json_defaults(args.override_config)
        defaults.update(overrides)

    parser.set_defaults(**defaults)

    parser.add_argument(
        "--path-model-save", type=str, help="Path to save the model checkpoints to"
    )
    parser.add_argument(
        "--path-model-load",
        type=str,
        help="Path to load a specific model checkpoint from",
    )
    parser.add_argument(
        "--path-env", type=str, help="Path to save and load the environment"
    )
    parser.add_argument(
        "env-name",
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

    parser.add_argument(
        "--num-learners",
        type=int,
        help="Number of learners (0 means one local learner)",
    )
    parser.add_argument(
        "--num-gpus-per-learner",
        type=int,
        help="Number of GPUs per learner, can be fractional",
    )

    # Training parameters
    parser.add_argument(
        "--epsilon", type=float, help="Clip parameter for PPO"
    )
    parser.add_argument(
        "--learning-rate", type=float, help="Learning rate"
    )
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

    args = parser.parse_args()

    # Write the defaults to another file
    with open('used_config.json', 'w') as f:
        json.dump(vars(args), f, indent=4)

    return args
