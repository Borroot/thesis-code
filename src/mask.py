import copy

import torch
import torch.nn as nn
from grid2op.Chronics import MultifolderWithCache
from grid2op.Reward import LinesCapacityReward
from sklearn.cluster import DBSCAN
from torch.distributions.categorical import Categorical

from env import Env


class MaskModel(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 64), nn.ReLU(), nn.Linear(64, act_dim * 2)
        )
        self.action_num = act_dim

    def forward(self, x):
        x = self.fc(x)
        return x

    def get_mask(self, x):
        logits = self.forward(x).view(-1, self.action_num, 2)
        categoricals = Categorical(logits=torch.clamp(logits, 0.0, 1.0))
        mask = categoricals.sample()
        logprob = categoricals.log_prob(mask)
        return mask, logprob

    def train(self, env, num_episodes=10):
        """Train the mask model on the given environment."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        # Initialize the environment
        obs, _ = env.reset()

        for episode in range(num_episodes):

            obs_batch = torch.zeros(
                (env.action_space.n, *obs["observations"].shape),
                dtype=torch.float32,
                device=device,
            )

            # Execute each action in the environment
            for action in range(env.action_space.n):
                env_copy = copy.deepcopy(env)
                obs, _, _, _, _ = env_copy.step(action)
                obs_batch[action] = obs["observations"]

            print(obs_batch.shape)
            print(obs_batch)

            return

            # Compute state differences and cluster
            dbscan = DBSCAN(eps=0.1, min_samples=1)
            clusters = dbscan.fit_predict(_.cpu().numpy())

            # Generate cluster labels
            labels = []
            tmp = []
            for c in clusters:
                tag = 0 if c in tmp else 1
                tmp.append(c)
                labels.append(tag)
            labels = torch.tensor(labels, dtype=torch.float32, device=device)

            # Generate mask and probabilities
            probs, mask = self.get_mask(obs)
            prob_labels = 1 - torch.abs(mask - labels).to(device)

            # Compute loss and update model
            loss = (probs.exp() - prob_labels).pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Episode {episode + 1}/{num_episodes}, Loss: {loss.item()}")

            # Perform a random action
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated or truncated:
                env.reset()

    # for idx, state in enumerate(obs[inds]):
    #     state_batch = torch.from_numpy(state).repeat(args.action_num, 1, 1, 1).numpy()
    #     action_batch =  torch.from_numpy(actions[inds][idx]).repeat(args.action_num, 1).to(device)
    #     motion = action_batch.clone()
    #     motion[:, 0] = motion_1
    #     with torch.no_grad():
    #         encode_s_batch = vf.encode(state_batch).detach()
    #         pred_s_= tr.pred_s(encode_s_batch, motion).detach()
    #         s_diff = pred_s_ - encode_s_batch
    #         dbscan = DBSCAN(eps=0.1, min_samples=1)
    #         clusters = dbscan.fit_predict(s_diff.cpu().numpy())
    #
    # encode_obs = vf.encode(obs[inds]).detach()
    # probs, mask = mk.get_space1_mask(encode_obs, invalid_action_masks[inds][:, 0:env.action_space.nvec[0]].to(device))


if __name__ == "__main__":
    env_config = {
        "env_name": "l2rpn_case14_sandbox",
        "reward_class": LinesCapacityReward,
        "chronics_class": MultifolderWithCache,
        # "mask_model": None,
    }
    env = Env(config=env_config)

    mask_model = MaskModel(
        env.observation_space["observations"].shape[0], env.action_space.n
    )
    mask_model.train(env, num_episodes=10)
