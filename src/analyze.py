import os
from pprint import pprint

import jsonpickle
import matplotlib.pyplot as plt


def load_checkpoints(path_folder):
    # Load iteration data
    iterations = []
    for file_name in sorted(os.listdir(path_folder)):
        if file_name.startswith("0") and file_name.endswith(".json"):
            path_file = os.path.join(path_folder, file_name)
            with open(path_file, "r", encoding="utf-8") as f:
                iterations.append(jsonpickle.decode(f.read()))

    # Load config file
    path_config = os.path.join(path_folder, "config.json")
    with open(path_config, "r", encoding="utf-8") as f:
        config = jsonpickle.decode(f.read())

    return config, iterations


def plot_metric(data, key_path):
    """Plot a specified metric over time."""
    plt.figure(figsize=(10, 6))

    values = []
    for entry in data:
        value = entry
        for key in key_path:
            value = value.get(key, None)
        values.append(value)

    plt.plot(range(len(values)), values, marker="o")

    plt.xlabel("iteration")
    plt.ylabel(key_path[-1])

    plt.grid(True)
    plt.show()


def plot_metrics(data, key_paths):
    """Plot a specified metric over time."""
    plt.figure(figsize=(10, 6))

    for key_path in key_paths:
        values = []
        for entry in data:
            value = entry
            for key in key_path:
                value = value.get(key, None)
            values.append(value)

        plt.plot(range(len(values)), values, marker="o", label=key_path[-1])

    plt.xlabel("iteration")
    plt.legend()

    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    path_folder = "checkpoints/"
    config, iterations = load_checkpoints(path_folder)

    key_paths = [
        ["learners", "default_policy", "policy_loss"],
        ["learners", "default_policy", "vf_loss"],
        # ["learners", "default_policy", "total_loss"],
        # ["learners", "default_policy", "mean_kl_loss"],
    ]
    key_path = [
        "evaluation",
        "env_runners",
        "agent_episode_returns_mean",
        "default_agent",
    ]
    pprint(iterations[0])

    # plot_metric(iterations, key_path)
    # plot_metrics(iterations, key_paths)
