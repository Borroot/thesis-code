import copy

import numpy as np
import torch
import torch.nn as nn
from grid2op.Chronics import MultifolderWithCache
from grid2op.Reward import LinesCapacityReward
from lightsim2grid import LightSimBackend
from sklearn.cluster import DBSCAN

from env import Env


class MaskModel(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim),
        )
        self.obs_dim = obs_dim
        self.act_dim = act_dim

    def forward(self, x):
        """Compute a forward pass and give the normalized log probabilities."""
        probs = torch.sigmoid(self.fc(x))
        logprobs = torch.log(probs + 1e-9)
        return logprobs

    def get_mask(self, x):
        """Compute the mask for the given state."""
        with torch.no_grad():
            probs = torch.sigmoid(self.fc(x))
            mask = torch.bernoulli(probs)
            return mask

    def _generate_obs_batch(self, env):
        """Execute each action in the current state of the environment."""
        obs_batch = np.zeros((self.act_dim, self.obs_dim), dtype=np.float32)
        for action in range(env.action_space.n):
            print(f"{action}/{range(env.action_space.n)}")
            env_copy = copy.deepcopy(env)
            obs_next, _, _, _, _ = env_copy.step(action)
            obs_batch[action] = obs_next["observations"]
        return obs_batch

    def _generate_labels(self, obs_batch):
        """Generate the label for each action based on clustering results."""
        # Cluster the observations
        dbscan = DBSCAN(eps=0.1, min_samples=1)
        clusters = dbscan.fit_predict(obs_batch)

        # Generate cluster labels
        labels = torch.zeros(*clusters.shape, dtype=torch.float32, device=self.device)
        seen = set()
        for index, c in enumerate(clusters):
            if c not in seen:
                seen.add(c)
                labels[index] = 1
        loglabels = torch.log(labels + 1e-9)
        return loglabels

    def train(self, env, num_episodes=10):
        """Train the mask model on the given environment."""
        # Move neural network to the gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        obs, _ = env.reset()

        for episode in range(num_episodes):
            # Generate the observations, the true labels and the predictions
            obs_batch = self._generate_obs_batch(env)
            loglabels = self._generate_labels(obs_batch)
            logprobs = self.forward(
                torch.tensor(obs["observations"], device=self.device)
            )

            # Compute loss and update model
            loss = (loglabels - logprobs).pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Episode {episode + 1}/{num_episodes}, Loss: {loss.item()}")

            # Perform a random action to update the environment
            obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated or truncated:
                obs, _ = env.reset()


if __name__ == "__main__":
    # TODO use a BaseMultiProcessEnv to utilize all cpu cores available
    # TODO write the obs_batches to a file for reuse
    # TODO decouple generation of data, labels and mask model training
    # TODO construct an environment which is fast and has a big action space
    # TODO add commandline argument parsing
    # TODO write training config and results to a file

    env = Env(
        {
            "env_name": "l2rpn_case14_sandbox",
            "backend_class": LightSimBackend,
            "reward_class": LinesCapacityReward,
            "chronics_class": MultifolderWithCache,
            # "n_busbar": 3,
        }
    )

    # mask_model = MaskModel(
    #     *env.observation_space["observations"].shape, env.action_space.n
    # )
    # mask_model.train(env, num_episodes=1)
