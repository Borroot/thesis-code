import os
import re

import grid2op
import ray
from grid2op.Chronics import MultifolderWithCache
from grid2op.Reward import LinesCapacityReward
from grid2op.Runner import Runner
from l2rpn_baselines.PPO_RLLIB import evaluate, train
from lightsim2grid import LightSimBackend

model_path = os.path.dirname(os.path.abspath(__file__)) + "/../models"
env_path = os.path.dirname(os.path.abspath(__file__)) + "/../envs"

if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(env_path):
    os.makedirs(env_path)

grid2op.change_local_dir(env_path)  # change where to store the environment

# Train the agent

env_name = "l2rpn_case14_sandbox"
env = grid2op.make(env_name, backend=LightSimBackend())
# env.chronics_handler.real_data.reset()

ray.init()
try:
    trained_aget = train(
        env,
        iterations=1,
        save_path=model_path,
        name="test",
        net_arch=[100, 100, 100],
        save_every_xxx_steps=1,
        env_kwargs={
            "reward_class": LinesCapacityReward,
            # "chronics_class": MultifolderWithCache,
            # "data_feeding_kwargs": {
            #     # use one over 100 chronics to train (for speed)
            #     "filter_func": lambda x: re.match(".*00$", x)
            #     is not None
            # },
        },
        verbose=True,
    )
finally:
    env.close()
    ray.shutdown()

# Test the agent

nb_episode = 7
nb_process = 1
verbose = True

env_name = "l2rpn_case14_sandbox"
env = grid2op.make(
    env_name, reward_class=LinesCapacityReward, backend=LightSimBackend()
)

try:
    trained_agent = evaluate(
        env,
        nb_episode=nb_episode,
        load_path=model_path,
        name="test",
        nb_process=1,
        verbose=verbose,
    )

    runner_params = env.get_params_for_runner()
    runner = Runner(**runner_params)
    res = runner.run(nb_episode=nb_episode, nb_process=nb_process)

    if verbose:
        print("Evaluation summary for DN:")
        for _, chron_name, cum_reward, nb_time_step, max_ts in res:
            msg_tmp = "chronics at: {}".format(chron_name)
            msg_tmp += "\ttotal score: {:.6f}".format(cum_reward)
            msg_tmp += "\ttime steps: {:.0f}/{:.0f}".format(nb_time_step, max_ts)
            print(msg_tmp)
finally:
    env.close()
