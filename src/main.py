# import os
# import re

# import grid2op
# import ray
# from eval import evaluate
from grid2op.Chronics import MultifolderWithCache
from grid2op.Reward import LinesCapacityReward
from ray.rllib.algorithms.ppo import PPOConfig

# from grid2op.Runner import Runner
# from lightsim2grid import LightSimBackend
# from train import train
from env import Env

# base_path = os.path.dirname(os.path.abspath(__file__))
# model_path = os.path.join(base_path, "models")
# env_path = os.path.join(base_path, "envs")

# os.makedirs(model_path, exist_ok=True)
# os.makedirs(env_path, exist_ok=True)

# grid2op.change_local_dir(env_path)  # change where to store the environment

config = PPOConfig().environment(
    Env,
    env_config={
        "env_name": "l2rpn_case14_sandbox",
        "reward_class": LinesCapacityReward,
        "chronics_class": MultifolderWithCache,
    },
)
algo = config.build_algo()
print(algo.train())

# # Train the agent

# env_name = "l2rpn_case14_sandbox"
# env = grid2op.make(
#     env_name, reward_class=LinesCapacityReward, backend=LightSimBackend()
# )

# ray.init()
# try:
#     trained_agent = train(
#         env,
#         iterations=1,
#         save_path=model_path,
#         name="test",
#         net_arch=[100, 100, 100],
#         save_every_xxx_steps=1,
#         env_kwargs={
#             "chronics_class": MultifolderWithCache,
#         },
#         verbose=True,
#     )
# finally:
#     env.close()
#     ray.shutdown()

# # # Test the agent

# nb_episode = 7
# nb_process = 1

# env = grid2op.make(
#     env_name, reward_class=LinesCapacityReward, backend=LightSimBackend()
# )
# try:
#     trained_agent = evaluate(
#         env,
#         nb_episode=nb_episode,
#         load_path=model_path,
#         name="test",
#         nb_process=nb_process,
#         verbose=True,
#     )

#     runner_params = env.get_params_for_runner()
#     runner = Runner(**runner_params)
#     res = runner.run(nb_episode=nb_episode, nb_process=nb_process)

#     print("Evaluation summary for DN:")
#     for _, chron_name, cum_reward, nb_time_step, max_ts in res:
#         msg_tmp = "chronics at: {}".format(chron_name)
#         msg_tmp += "\ttotal score: {:.6f}".format(cum_reward)
#         msg_tmp += "\ttime steps: {:.0f}/{:.0f}".format(nb_time_step, max_ts)
#         print(msg_tmp)
# finally:
#     env.close()
