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
    iterations,
    sampling_time,
    num_threads,
    learner_update_time,
    evaluation_time,
    num_gpus,
):
    return iterations * (
        (sampling_time / num_threads)
        + (learner_update_time / num_gpus)
        + evaluation_time
    )


def set_log_yscale_and_labels(ax, env):
    # Set log scale and custom y-ticks/labels for specific environments
    ax.set_yscale("log")
    plt.ylabel("Total Training Time (hours)")
    yticks = [1e2, 1e3, 1e4, 1e5]
    ylabels = [
        f"{int(yticks[0]):,}h\n({yticks[0]/24:.1f}d)",
        f"{int(yticks[1]):,}h\n({yticks[1]/168:.1f}w)",
        f"{int(yticks[2]):,}h\n({yticks[2]/168:.1f}w)",
        f"{int(yticks[3]):,}h\n({yticks[3]/8760:.2f}y)",
    ]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.yaxis.set_minor_locator(
        mticker.LogLocator(base=10.0, subs=np.arange(1, 10), numticks=100)
    )
    ax.grid(which="minor", axis="y", linestyle=":", linewidth=0.7, alpha=0.6)
    ax.grid(which="major", axis="y", linestyle="-", linewidth=1.2)


def set_linear_yscale_and_labels(ax):
    # Set linear scale and y-ticks/labels for sandbox
    hours_ticks = ax.get_yticks()
    hours_ticks = [tick for tick in hours_ticks if tick >= 0]
    hour_labels = [f"{int(tick):,}" if tick > 0 else "0" for tick in hours_ticks]
    ax.set_yticks(hours_ticks)
    ax.set_yticklabels(hour_labels)
    plt.ylabel("Total Training Time (hours)")


def plot_mask_model_lines(
    ax,
    env,
    env_iterations_range,
    threads_list,
    thread_to_color,
    laptop_env_stats,
    cluster_env_stats,
):
    lines = []
    labels = []
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
            (line,) = ax.plot(
                env_iterations_range,
                times,
                marker=marker,
                color=thread_to_color[n_threads],
                label=f"{system_name} - {n_threads} Threads",
            )
            lines.append(line)
            labels.append(f"{system_name} - {n_threads} Threads")
    return lines, labels


def mask_setup_plot(figsize=(20, 16), font_sizes=None):
    """Setup plot for mask model plots."""
    if font_sizes is None:
        font_sizes = {
            "font.size": 28,
            "axes.titlesize": 32,
            "axes.labelsize": 30,
            "xtick.labelsize": 26,
            "ytick.labelsize": 26,
            "legend.fontsize": 24,
            "legend.title_fontsize": 26,
        }
    plt.figure(figsize=figsize)
    plt.rcParams.update(font_sizes)


def mask_save_and_show_plot(plots_dir, filename):
    os.makedirs(plots_dir, exist_ok=True)
    save_path = os.path.join(plots_dir, filename)
    plt.savefig(save_path)
    plt.show()


def mask_get_thread_to_color(threads_list, color_map="tab10"):
    color_cycle = plt.colormaps[color_map]
    return {n_threads: color_cycle(i) for i, n_threads in enumerate(threads_list)}


def mask_plot_env_comparison(
    laptop_env_stats, cluster_env_stats, threads_list_dict, iterations_range
):
    plots_dir = os.path.join(os.path.dirname(__file__), "..", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    for env in laptop_env_stats.keys():
        threads_list = threads_list_dict[env]
        thread_to_color = mask_get_thread_to_color(threads_list)

        mask_setup_plot()
        if env in ["neurips_2020_track1", "wcci_2022"]:
            env_iterations_range = np.arange(0, 100_001, 5_000)
        else:
            env_iterations_range = iterations_range

        ax = plt.gca()
        lines, labels = plot_mask_model_lines(
            ax,
            env,
            env_iterations_range,
            threads_list,
            thread_to_color,
            laptop_env_stats,
            cluster_env_stats,
        )
        plt.xlabel("Number of Iterations")

        if env in ["neurips_2020_track1", "wcci_2022"]:
            set_log_yscale_and_labels(ax, env)
        else:
            set_linear_yscale_and_labels(ax)

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
        plt.tight_layout(rect=[0, 0, 1, 0.98])

        mask_save_and_show_plot(plots_dir, f"{env}_mask_model_training.png")


def ppo_setup_plot(figsize=(20, 16), font_sizes=None):
    """Setup plot for PPO plots."""
    if font_sizes is None:
        font_sizes = {
            "font.size": 28,
            "axes.titlesize": 32,
            "axes.labelsize": 30,
            "xtick.labelsize": 26,
            "ytick.labelsize": 26,
            "legend.fontsize": 22,
            "legend.title_fontsize": 26,
        }
    plt.figure(figsize=figsize)
    plt.rcParams.update(font_sizes)


def ppo_save_and_show_plot(plots_dir, filename):
    os.makedirs(plots_dir, exist_ok=True)
    save_path = os.path.join(plots_dir, filename)
    plt.savefig(save_path)
    plt.show()


def ppo_get_thread_to_color(threads_list, color_map="tab10"):
    color_cycle = plt.colormaps[color_map]
    return {n_threads: color_cycle(i) for i, n_threads in enumerate(threads_list)}


def plot_ppo_lines(
    ax, threads_list, thread_to_color, env_stats, iterations_range, systems
):
    lines, labels = [], []
    for n_threads in threads_list:
        num_gpus = max(1, int(np.ceil(n_threads / 36)))
        for system_name, marker, _ in systems:
            stats = env_stats[system_name][1]
            times = [
                ppo_training_time(
                    iters,
                    sampling_time=stats["sampling_time"],
                    num_threads=n_threads,
                    learner_update_time=stats["learner_update_time"],
                    evaluation_time=stats["evaluation_time"],
                    num_gpus=num_gpus,
                )
                / 3600
                for iters in iterations_range
            ]
            if times:
                times[0] = np.nan
            (line,) = ax.plot(
                iterations_range,
                times,
                marker=marker,
                color=thread_to_color[n_threads],
                label=f"{system_name} - {n_threads} Threads, {num_gpus} GPUs",
            )
            lines.append(line)
            labels.append(f"{system_name} - {n_threads} Threads, {num_gpus} GPUs")
    return lines, labels


def ppo_set_log_yscale_and_labels(ax):
    ax.set_yscale("log")
    yticks = [1e1, 1e2, 1e3, 1e4]
    ylabels = [
        f"{int(yticks[0]):,}h",
        f"{int(yticks[1]):,}h\n({yticks[1]/24:.2f}d)",
        f"{int(yticks[2]):,}h\n({yticks[2]/168:.2f}w)",
        f"{int(yticks[3]):,}h\n({yticks[3]/8760:.2f}y)",
    ]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.yaxis.set_minor_locator(
        mticker.LogLocator(base=10.0, subs=np.arange(1, 10), numticks=100)
    )
    ax.grid(which="minor", axis="y", linestyle=":", linewidth=0.7, alpha=0.6)
    ax.grid(which="major", axis="y", linestyle="-", linewidth=1.2)
    plt.ylabel("Total Training Time (hours)")


def ppo_plot_env_comparison(threads_list, env_stats):
    systems = [
        ("Laptop", "o", "#1f77b4"),
        ("Cluster", "^", "#ff7f0e"),
    ]
    iterations_range = np.arange(0, 1_000_001, 50_000)
    thread_to_color = ppo_get_thread_to_color(threads_list)

    ppo_setup_plot()
    ax = plt.gca()
    lines, labels = plot_ppo_lines(
        ax, threads_list, thread_to_color, env_stats, iterations_range, systems
    )
    plt.xlabel("Number of Iterations")
    ppo_set_log_yscale_and_labels(ax)
    plt.title("case14_sandbox: Laptop vs Cluster PPO Training Time")
    plt.legend(
        lines,
        labels,
        ncol=len(threads_list),
        loc="lower center",
        bbox_to_anchor=(0.45, 1.07),
        borderaxespad=0.2,
        frameon=True,
    )
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plots_dir = os.path.join(os.path.dirname(__file__), "..", "plots")
    ppo_save_and_show_plot(plots_dir, "case14_sandbox_ppo_training.png")


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

    threads_list_dict = {
        "case14_sandbox": [16, 72, 2 * 72],
        "neurips_2020_track1": [16, 72 * 2, 72 * 2 * 72],
        "wcci_2022": [16, 72 * 2, 72 * 2 * 72],
    }
    iterations_range = np.arange(0, 1_000_001, 50_000)

    mask_plot_env_comparison(
        laptop_env_stats, cluster_env_stats, threads_list_dict, iterations_range
    )

    ppo_env_stats = {
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
    ppo_plot_env_comparison([16, 72 * 2, 72 * 2 * 72], ppo_env_stats)
