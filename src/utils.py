import json
import os

from grid2op.Episode import EpisodeReplay


def remove_invalid_actions(g2p_env, act_tokeep):
    """Filter out actions which are not usable in the environment."""
    return [act for act in act_tokeep if g2p_env.action_space.supports_type(act)]


def load_used_attributes(path_full, obs_tokeep, act_tokeep):
    """Load the obs_tokeep and act_tokeep dicts from the serialized json files."""
    path_obs = os.path.join(path_full, "obs_tokeep.json")
    with open(path_obs, encoding="utf-8", mode="r") as f:
        obs_tokeep = json.load(fp=f)

    path_act = os.path.join(path_full, "act_tokeep.json")
    with open(path_act, encoding="utf-8", mode="r") as f:
        act_tokeep = json.load(fp=f)

    return obs_tokeep, act_tokeep


def save_used_attributes(path_save, name, obs_tokeep, act_tokeep):
    """Serialize, as json, the obs_tokeep and act_tokeep dicts."""
    path_full = os.path.join(path_save, name)
    os.makedirs(path_full, exist_ok=True)

    file_obs = os.path.join(path_full, "obs_tokeep.json")
    with open(file_obs, encoding="utf-8", mode="w") as f:
        json.dump(fp=f, obj=obs_tokeep)

    file_act = os.path.join(path_full, "act_tokeep.json")
    with open(file_act, encoding="utf-8", mode="w") as f:
        json.dump(fp=f, obj=act_tokeep)


def save_log_gif(path_log, results, gif_name=None):
    """
    Output a gif named (by default "episode.gif") that is the replay of the
    episode in a gif format, for each episode in the input.

    Parameters
    ----------
    path_log: ``str``
        Path where the log of the agents are saved.

    res: ``list``
        List resulting from the call to `runner.run`

    gif_name: ``str``
        Name of the gif that will be used.
    """

    episode_replay = EpisodeReplay(path_log)
    for _, chron_name, _, _, _ in results:
        print("Creating {}.gif".format(gif_name))
        episode_replay.replay_episode(
            episode_id=chron_name, gif_name=gif_name + chron_name, display=False
        )
        gif_path = os.path.join(path_log, chron_name, gif_name)
        print("Wrote {}.gif".format(gif_path))
