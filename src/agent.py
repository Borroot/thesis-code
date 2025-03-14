import copy
import json
import os

import jsonpickle
import numpy as np
import torch
from grid2op.Agent import BaseAgent
from grid2op.gym_compat import BoxGymActSpace, BoxGymObsSpace, GymEnv
from grid2op.gym_compat.box_gym_obsspace import ALL_ATTR_OBS
from grid2op.gym_compat.utils import ATTR_DISCRETE
from grid2op.Runner import Runner
from ray.rllib.algorithms.ppo import PPO
from ray.tune.logger import pretty_print

from agent import GymAgent
from env import Env
from utils import (
    load_used_attributes,
    remove_invalid_actions,
    save_log_gif,
    save_used_attributes,
)


class GymAgent(BaseAgent):
    """
    This class maps a neural network (trained using ray/rllib) to a grid2op
    agent. It can then be used as a "regular" grid2op agent, in a runner,
    grid2viz, grid2game etc.
    """

    def __init__(
        self,
        g2p_act_space,
        gym_act_space,
        gym_obs_space,
        *,
        nn_path=None,
        nn_kwargs=None,
    ):
        if nn_path is None and nn_kwargs is None:
            raise RuntimeError(
                "Impossible to build a GymAgent without providing at least "
                "`nn_path` (to load an agent from the disk) or "
                "`nn_kwargs` (to create a new agent)."
            )

        if nn_path is not None and nn_kwargs is not None:
            raise RuntimeError(
                "Impossible to build a GymAgent by providing both "
                "`nn_path` (to load an agent from the disk) and "
                "`nn_kwargs` (to create a new agent)."
            )

        self._nn_config = nn_kwargs
        nn_kwargs = {"env": Env, "config": nn_kwargs}

        super().__init__(g2p_act_space)
        self._gym_act_space = gym_act_space
        self._gym_obs_space = gym_obs_space

        self._nn_kwargs = copy.deepcopy(nn_kwargs)
        self._nn_path = nn_path

        self.nn_model = None
        if nn_path is not None:
            self.load()
        else:
            self.build()

    def get_act(self, gym_obs, reward, done):
        """Retrieve the action from the NN model"""
        model_outputs = self.nn_model.forward_inference(
            {"obs": torch.from_numpy(gym_obs)}
        )
        action_dist_params = model_outputs["action_dist_inputs"][0].numpy()
        return np.argmax(action_dist_params)

    def load(self):
        """Load an existing NN model."""
        # TODO
        self.build()
        self.nn_model.restore(checkpoint_path=self._nn_path)

    def save(self):
        """Save the current NN model."""
        # TODO
        self.nn_model.save(checkpoint_dir=self._nn_path)

    def build(self):
        """Build a new NN model"""
        # TODO
        self.nn_model = PPO(**self._nn_kwargs)

    def act(self, g2p_obs, reward, done):
        """Take an action based on the observation and the reward."""
        gym_obs = self._gym_obs_space.to_gym(g2p_obs)
        gym_act = self.get_act(gym_obs, reward, done)
        return self._gym_act_space.from_gym(gym_act)

    def train(
        self,
        env,
        name="ppo_rllib",
        iterations=1,
        path_save=None,
        load_path=None,
        net_arch=None,
        learning_rate=3e-4,
        verbose=False,
        save_every_xxx_steps=None,
        obs_tokeep=copy.deepcopy(ALL_ATTR_OBS),
        act_tokeep=copy.deepcopy(ATTR_DISCRETE),
        env_kwargs=None,
        **kwargs,
    ):
        """
        .. warning::
            The environment used by RLLIB is copied and remade. This class does
            not work if you over specialize the environment !
            For example, opponent is not taken into account (yet), nor the chronics class
            etc.

            If you want such level of control, please use the `env_kwargs` parameters !

        env_kwargs: Optional[dict]
            Extra key word arguments passed to the building of the
            grid2op environment.

        kwargs:
            extra parameters passed to the trainer from rllib

        """

        if path_save is not None:
            path_expe = os.path.join(path_save, name)
            os.makedirs(path_expe, exist_ok=True)

        # save the attributes kept
        act_tokeep = remove_invalid_actions(env, act_tokeep)
        save_used_attributes(path_save, name, obs_tokeep, act_tokeep)
        need_saving = save_every_xxx_steps is not None

        if env_kwargs is None:
            env_kwargs = {}

        env_config = {
            "backend_class": env.get_kwargs()["_raw_backend_class"],
            "env_name": env.env_name,
            "obs_tokeep": obs_tokeep,
            "act_tokeep": act_tokeep,
            **env_kwargs,
        }

        model_dict = {}
        if net_arch is not None:
            model_dict["fcnet_hiddens"] = net_arch
        env_config_ppo = {
            # config to pass to env class
            "env_config": env_config,
            # neural network config
            "lr": learning_rate,
            "model": model_dict,
            **kwargs,
        }

        # store it
        encoded = jsonpickle.encode(env_config_ppo)
        with open(
            os.path.join(path_expe, "env_config.json"), "w", encoding="utf-8"
        ) as f:
            f.write(encoded)

        # define the gym environment from the grid2op env
        env_gym = GymEnv(env)
        env_gym.observation_space.close()
        env_gym.observation_space = BoxGymObsSpace(
            env.observation_space, attr_to_keep=obs_tokeep
        )
        env_gym.action_space.close()
        env_gym.action_space = BoxGymActSpace(env.action_space, attr_to_keep=act_tokeep)

        # then define a "trainer"
        agent = GymAgent(
            g2op_action_space=env.action_space,
            gym_act_space=env_gym.action_space,
            gym_obs_space=env_gym.observation_space,
            nn_config=env_config_ppo,
            nn_path=load_path,
        )

        for i in range(iterations):
            result = agent.nn_model.train()
            if verbose:
                print(pretty_print(result))

            if i % save_every_xxx_steps == 0:
                agent.save()

        agent.save()

        return agent

    def evaluate(
        env,
        name="ppo_rllib",
        path_load=".",
        path_logs=None,
        nb_episode=1,
        nb_process=1,
        max_steps=-1,
        verbose=False,
        save_gif=False,
        **kwargs,
    ):
        """
        logs_dir: ``str``
            Where to store the tensorboard generated logs during the training. ``None`` if you don't want to log them.

        nb_episode: ``str``
            How many episodes to run during the assessment of the performances

        nb_process: ``int``
            On how many process the assessment will be made. (setting this > 1 can lead to some speed ups but can be
            unstable on some plaform)

        max_steps: ``int``
            How many steps at maximum your agent will be assessed

        kwargs:
            extra parameters passed to the PPO from stable baselines 3

        Returns
        -------

        baseline:
            The loaded baseline as a stable baselines PPO element.

        """

        # load the attributes kept
        path_full = os.path.join(path_load, name)
        obs_tokeep, act_tokeep = load_used_attributes(path_full, obs_tokeep, act_tokeep)

        # create the action and observation space
        gym_obs_space = BoxGymObsSpace(env.observation_space, attr_to_keep=obs_tokeep)
        gym_act_space = BoxGymActSpace(env.action_space, attr_to_keep=act_tokeep)

        # retrieve the env config (for rllib)
        with open(
            os.path.join(path_full, "env_config.json"), "r", encoding="utf-8"
        ) as f:
            str_ = f.read()
        env_config_ppo = jsonpickle.decode(str_)

        # create a grid2gop agent based on that (this will reload the save weights)
        full_path = os.path.join(path_load, name)
        agent = GymAgent(
            env.action_space,
            gym_act_space,
            gym_obs_space,
            nn_config=env_config_ppo,
            nn_path=os.path.join(full_path),
        )

        # Build runner
        runner_params = env.get_params_for_runner()
        runner_params["verbose"] = verbose
        runner = Runner(**runner_params, agentClass=None, agentInstance=agent)

        # Run the agent on the scenarios
        if path_logs is not None:
            os.makedirs(path_logs, exist_ok=True)

        res = runner.run(
            path_save=path_logs,
            nb_episode=nb_episode,
            nb_process=nb_process,
            max_iter=max_steps,
            pbar=verbose,
            **kwargs,
        )

        # Print summary
        if verbose:
            print("Evaluation summary:")
            for _, chron_name, cum_reward, nb_time_step, max_ts in res:
                msg_tmp = "chronics at: {}".format(chron_name)
                msg_tmp += "\ttotal score: {:.6f}".format(cum_reward)
                msg_tmp += "\ttime steps: {:.0f}/{:.0f}".format(nb_time_step, max_ts)
                print(msg_tmp)

        if save_gif:
            save_log_gif(path_logs, res)

        return agent, res
