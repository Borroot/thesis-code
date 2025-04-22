import json
import os

import matplotlib.pyplot as plt


def load_checkpoints(path_folder):
    # Load iteration data
    iterations = []
    for file_name in sorted(os.listdir(path_folder)):
        if file_name.startswith("0") and file_name.endswith(".json"):
            path_file = os.path.join(path_folder, file_name)
            with open(path_file, "r", encoding="utf-8") as f:
                iterations.append(json.load(f))

    # Load config file
    path_config = os.path.join(path_folder, "config.json")
    with open(path_config, "r", encoding="utf-8") as f:
        config = json.load(f)

    return config, iterations


def plot_metric(data, key_path):
    """Plot a specified metric over time."""
    values = []
    for entry in data:
        value = entry
        for key in key_path:
            value = value.get(key, None)
        values.append(value)

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(values)), values, marker="o")

    plt.xlabel("iteration")
    plt.ylabel(key_path[-1])
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    path_folder = "checkpoints/"
    config, iterations = load_checkpoints(path_folder)
    plot_metric(iterations, ["learners", "default_policy", "total_loss"])
