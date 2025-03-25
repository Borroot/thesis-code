import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


class MaskModel(nn.Module):
    def __init__(self, obs_dim, action_dim, load_model):
        super(MaskModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(), nn.Linear(64, action_dim * 2)
        )
        self.action_num = action_dim
        self.load_model = load_model

    def forward(self, x):
        x = self.fc(x)
        return x

    def get_mask(self, x, mask=None):
        logits = self.forward(x).view(-1, self.action_num, 2)
        categoricals = Categorical(logits=torch.clamp(logits, 0.0, 1.0))
        if mask is None:
            mask = categoricals.sample()
        logprob = categoricals.log_prob(mask)
        return mask, logprob

    # np.random.shuffle(inds)
    # motion_1 = torch.tensor(np.arange(args.action_num), dtype=torch.float32, device=device)
    # motion_2 = torch.tensor(np.arange(args.action_num), dtype=torch.float32, device=device)
    # labels = []
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
    #     label = []
    #     tmp = []
    #     for c in clusters:
    #         tag = 0 if c in tmp else 1
    #         tmp.append(c)
    #         label.append(tag)
    #     labels.append(label)
    # labels = torch.tensor(np.array(labels), dtype=torch.float32, device=device)
    #
    # encode_obs = vf.encode(obs[inds]).detach()
    # probs, mask = mk.get_space1_mask(encode_obs, invalid_action_masks[inds][:, 0:env.action_space.nvec[0]].to(device))
    # prob_labels = 1 - torch.abs(mask - labels).to(device)
    #
    # loss1 = (probs.exp() - prob_labels).pow(2).mean()
    # mk_optimizer.zero_grad()
    # loss1.backward()
    # mk_optimizer.step()
