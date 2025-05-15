import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


def mask_model_training_time(
    iterations, batch_gen_time, label_gen_time, num_threads, model_update_time
):
    return iterations * (
        ((batch_gen_time + label_gen_time) / num_threads) + model_update_time
    )


def ppo_training_time(
    iterations, sampling_time, num_threads, learner_update_time, evaluation_time
):
    return iterations * (
        (sampling_time / num_threads) + learner_update_time + evaluation_time
    )


def plot_env_comparison(
    laptop_env_stats, cluster_env_stats, threads_list_dict, iterations_range
):
    color_cycle = plt.colormaps["tab10"]

    # Ensure the plots directory exists
    plots_dir = os.path.join(os.path.dirname(__file__), "..", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    for env in laptop_env_stats.keys():
        # Select threads_list based on environment
        threads_list = threads_list_dict[env]
        thread_to_color = {
            n_threads: color_cycle(i) for i, n_threads in enumerate(threads_list)
        }

        plt.figure(figsize=(20, 16))  # Increase height
        plt.rcParams.update(
            {
                "font.size": 28,
                "axes.titlesize": 32,
                "axes.labelsize": 30,
                "xtick.labelsize": 26,
                "ytick.labelsize": 26,
                "legend.fontsize": 24,
                "legend.title_fontsize": 26,
            }
        )
        lines = []
        labels = []
        if env in ["neurips_2020_track1", "wcci_2022"]:
            env_iterations_range = np.arange(0, 100_001, 5_000)
        else:
            env_iterations_range = iterations_range
        for n_threads in threads_list:
            for system_name, env_stats, marker in [
                ("Laptop", laptop_env_stats, "o"),
                ("Cluster", cluster_env_stats, "^"),
            ]:
                stats = env_stats[env]
                times = [
                    mask_model_training_time(
                        iters,
                        batch_gen_time=stats["batch_gen_time"],
                        label_gen_time=stats["label_gen_time"],
                        model_update_time=stats["model_update_time"],
                        num_threads=n_threads,
                    )
                    / 3600
                    for iters in env_iterations_range
                ]
                # Fix: For log plots, set the first value to np.nan to avoid log(0)
                if env in ["neurips_2020_track1", "wcci_2022"]:
                    if len(times) > 0:
                        times[0] = np.nan
                (line,) = plt.plot(
                    env_iterations_range,
                    times,
                    marker=marker,
                    color=thread_to_color[n_threads],
                    label=f"{system_name} - {n_threads} Threads",
                )
                lines.append(line)
                labels.append(f"{system_name} - {n_threads} Threads")
        plt.xlabel("Number of Iterations")

        ax = plt.gca()
        # Set log scale for neurips_2020_track1 and wcci_2022, but set ticks/labels after plotting
        if env in ["neurips_2020_track1", "wcci_2022"]:
            ax.set_yscale("log")
            plt.ylabel("Total Training Time (hours)")

            # Manually set four log ticks (adjust values as needed for your data)
            yticks = [1e2, 1e3, 1e4, 1e5]
            ylabels = [
                f"{int(yticks[0]):,}\n({yticks[0]/24:.1f}d)",
                f"{int(yticks[1]):,}\n({yticks[1]/168:.1f}w)",
                f"{int(yticks[2]):,}\n({yticks[2]/168:.1f}w)",
                f"{int(yticks[3]):,}\n({yticks[3]/8760:.2f}y)",
            ]
            ax.set_yticks(yticks)
            ax.set_yticklabels(ylabels)

            # Add more horizontal grid lines (minor grid)
            ax.yaxis.set_minor_locator(
                mticker.LogLocator(base=10.0, subs=np.arange(1, 10), numticks=100)
            )
            ax.grid(which="minor", axis="y", linestyle=":", linewidth=0.7, alpha=0.6)
            ax.grid(which="major", axis="y", linestyle="-", linewidth=1.2)
        else:
            # Only show hours, no weeks for sandbox
            hours_ticks = ax.get_yticks()
            hours_ticks = [tick for tick in hours_ticks if tick >= 0]
            hour_labels = [
                f"{int(tick):,}" if tick > 0 else "0" for tick in hours_ticks
            ]
            ax.set_yticks(hours_ticks)
            ax.set_yticklabels(hour_labels)
            plt.ylabel("Total Training Time (hours)")

        plt.title(f"{env}: Laptop vs Cluster Mask Model Training Time")
        plt.legend(
            lines,
            labels,
            ncol=len(threads_list),
            loc="lower center",
            bbox_to_anchor=(0.5, 1.07),
            borderaxespad=0.2,
            frameon=True,
        )
        plt.grid(True)
        plt.gca().xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
        plt.tight_layout(rect=[0, 0, 1, 0.98])  # Leave more space for legend

        # Save the figure with a descriptive filename (no "laptop_vs_cluster")
        save_path = os.path.join(plots_dir, f"{env}_mask_model_training.png")
        plt.savefig(save_path)
        plt.show()


def plot_ppo_env_comparison(threads_list):
    # Only use the 1 environment setup
    systems = [
        ("Laptop", "o", "#1f77b4"),
        ("Cluster", "^", "#ff7f0e"),
    ]
    env_stats = {
        "Laptop": {
            1: {
                "sampling_time": 100.28,
                "learner_update_time": 19.049,
                "evaluation_time": 0.237,
            },
        },
        "Cluster": {
            1: {
                "sampling_time": 133.128,
                "learner_update_time": 29.393,
                "evaluation_time": 0.334,
            },
        },
    }
    iterations_range = np.arange(0, 1_000_001, 50_000)  # Only up to 1 million

    color_cycle = plt.colormaps["tab10"]
    thread_to_color = {
        n_threads: color_cycle(i) for i, n_threads in enumerate(threads_list)
    }

    plt.figure(figsize=(20, 16))  # Increase height
    plt.rcParams.update(
        {
            "font.size": 28,
            "axes.titlesize": 32,
            "axes.labelsize": 30,
            "xtick.labelsize": 26,
            "ytick.labelsize": 26,
            "legend.fontsize": 24,
            "legend.title_fontsize": 26,
        }
    )

    lines = []
    labels = []
    for n_threads in threads_list:
        for system_name, marker, _ in systems:
            stats = env_stats[system_name][1]
            times = [
                ppo_training_time(
                    iters,
                    sampling_time=stats["sampling_time"],
                    num_threads=n_threads,
                    learner_update_time=stats["learner_update_time"],
                    evaluation_time=stats["evaluation_time"],
                )
                / 3600  # convert to hours
                for iters in iterations_range
            ]
            (line,) = plt.plot(
                iterations_range,
                times,
                marker=marker,
                color=thread_to_color[n_threads],
                label=f"{system_name} - {n_threads} Threads",
            )
            lines.append(line)
            labels.append(f"{system_name} - {n_threads} Threads")

    plt.xlabel("Number of Iterations")

    # Add weeks to y-axis labels for sandbox PPO
    ax = plt.gca()
    hours_ticks = ax.get_yticks()
    hours_ticks = [tick for tick in hours_ticks if tick >= 0]
    week_labels = [
        f"{int(tick):,} ({tick/168:.2f}w)" if tick > 0 else "0" for tick in hours_ticks
    ]
    ax.set_yticks(hours_ticks)
    ax.set_yticklabels(week_labels)
    plt.ylabel("Total Training Time (hours, weeks)")

    plt.title("case14_sandbox: Laptop vs Cluster PPO Training Time (1 Env)")
    plt.legend(
        lines,
        labels,
        ncol=len(threads_list),
        loc="lower center",
        bbox_to_anchor=(0.5, 1.07),
        borderaxespad=0.2,
        frameon=True,
    )
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    plt.tight_layout(rect=[0, 0, 1, 0.98])  # Leave more space for legend

    plots_dir = os.path.join(os.path.dirname(__file__), "..", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    save_path = os.path.join(plots_dir, "case14_sandbox_ppo_training_1env_threads.png")
    plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    laptop_env_stats = {
        "case14_sandbox": {
            "batch_gen_time": 9.79,
            "label_gen_time": 0.0052,
            "model_update_time": 0.00083,
        },
        "neurips_2020_track1": {
            "batch_gen_time": 177760,
            "label_gen_time": 257.98,
            "model_update_time": 0.00142,
        },
        "wcci_2022": {
            "batch_gen_time": 228217,
            "label_gen_time": 964.00,
            "model_update_time": 0.00140,
        },
    }

    cluster_env_stats = {
        "case14_sandbox": {
            "batch_gen_time": 15.83,
            "label_gen_time": 0.0090,
            "model_update_time": 0.00129,
        },
        "neurips_2020_track1": {
            "batch_gen_time": 259252,
            "label_gen_time": 335.82,
            "model_update_time": 0.00181,
        },
        "wcci_2022": {
            "batch_gen_time": 328844,
            "label_gen_time": 1269.0,
            "model_update_time": 0.00186,
        },
    }

    # Define threads_list for each environment
    threads_list_dict = {
        "case14_sandbox": [16, 72, 2 * 72],
        "neurips_2020_track1": [16, 72 * 2, 72 * 2 * 72],
        "wcci_2022": [16, 72 * 2, 72 * 2 * 72],
    }
    iterations_range = np.arange(0, 1_000_001, 50_000)

    plot_env_comparison(
        laptop_env_stats, cluster_env_stats, threads_list_dict, iterations_range
    )
    # For PPO, use the same threads as for neurips/wcci
    plot_ppo_env_comparison([16, 72 * 2, 72 * 2 * 72])
