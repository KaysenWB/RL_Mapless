import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torchvision import models
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


class AgentPosi_backup(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args.device
        self.state_fests = args.state_fests
        self.num_actions = args.num_actions
        self.hidden_feats = args.hidden_feats

        self.obs_enc = nn.Sequential(
            nn.Linear(self.state_fests, self.hidden_feats),
            nn.ReLU(),
            nn.Linear(self.hidden_feats, self.hidden_feats),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(self.hidden_feats, self.hidden_feats)
        self.actor = nn.Linear(self.hidden_feats, self.num_actions)
        self.critic = nn.Linear(self.hidden_feats, 1)

    def init_hidden(self, batch_size = 1):
        h0 = torch.zeros(batch_size, 1, self.hidden_feats, device=self.device)
        c0 = torch.zeros(batch_size, 1, self.hidden_feats, device=self.device)
        return (h0, c0)

    def forward(self, obs, hidden):
        # obs: [T, B, obs_dim]
        x = self.obs_enc(obs).unsqueeze(1)
        out, hidden_next = self.lstm(x, hidden)
        logits = self.actor(out).squeeze()
        values = self.critic(out).squeeze()
        return logits, values, hidden_next




class AgentPosi(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args.device
        self.state_fests = args.state_fests
        self.num_actions = args.num_actions
        self.hidden_feats = args.hidden_feats

        self.obs_enc = nn.Sequential(
            nn.Linear(20, self.hidden_feats),
            nn.Tanh(),
            nn.Linear(self.hidden_feats, self.hidden_feats),
        )
        self.ten = nn.Tanh()

        """
            self.pred_obs = nn.Sequential(
            nn.Linear(self.hidden_feats, self.hidden_feats),
            nn.Tanh(),
            nn.Linear(self.hidden_feats, 256),
        )
        self.pred_goal = nn.Sequential(
            nn.Linear(self.hidden_feats, self.hidden_feats),
            nn.Tanh(),
            nn.Linear(self.hidden_feats, 256),
        )"""

        self.lstm = nn.LSTM(self.hidden_feats, self.hidden_feats)
        self.actor = nn.Linear(self.hidden_feats, self.num_actions)
        self.critic = nn.Linear(self.hidden_feats, 1)
        self.pred_obs = nn.Linear(self.hidden_feats, 2)
        self.pred_goal = nn.Linear(self.hidden_feats, 2)
        self.pred_sp = nn.Linear(self.hidden_feats, 16)

        self.cnn =nn.Linear(1000, self.hidden_feats)
        self.fusion = nn.Linear(self.hidden_feats * 2, self.hidden_feats)



    def init_hidden(self, batch_size=1):
        h0 = torch.zeros(1, batch_size, self.hidden_feats, device=self.device)
        c0 = torch.zeros(1, batch_size, self.hidden_feats, device=self.device)
        return (h0, c0)

    def loss_mse(self, da, da1):
        return torch.sum((da - da1) ** 2, dim=-1).mean()

    def loss_en(self, logits, labels):
        return F.cross_entropy(logits, labels, reduction='sum')

    def forward(self, obs, hidden, views = 0):

        bins_x, bins_y = obs[:, 0].long(), obs[:, 1].long()
        grids_x, grids_y = obs[:, 2:4], obs[:, 4:6]
        lonlat_x, lonlat_y = obs[:, 6:8], obs[:, 8:10]

        tra_x, tra_y = lonlat_x, lonlat_y
        tar_sp = obs[:, 11].long()

        obs_ = obs[:, 32:]
        #obs_ = torch.cat((lonlat_x, grids_y), dim=-1)

        x = self.obs_enc(obs_).unsqueeze(1)
        if isinstance(views, torch.Tensor):
            view_enc = self.cnn(views).unsqueeze(1)
            x = self.fusion(torch.cat((x, view_enc),dim=-1))

        out, hidden_next = self.lstm(x, hidden)
        out = self.ten(out)
        logits = self.actor(out).squeeze()
        values = self.critic(out).squeeze()


        sp_pred = self.pred_sp(out).squeeze(1)
        loss_sp = self.loss_en(sp_pred, tar_sp)

        loss = 0
        obs_pred = self.pred_obs(out).squeeze(1)
        goal_pred = self.pred_goal(out).squeeze(1)
        loss = self.loss_mse(obs_pred, tra_x) + self.loss_mse(goal_pred, tra_y)

        return logits, values, hidden_next, loss


# min 场景
#  pics + marker: 0.45,
#  pics + marker + pred_lonlat: 0.67-0.75
#  pics + marker + pred_sp: 0.65-0.72
#  pics + marker + pred_emb: 0.5
#  pics + marker + pred_bins: 0.2


# mid 场景
# lonlat + lonlat: 0.5, training: above 5
# marker + marker: -, training: -
# pics + lonlat: 0.15, training: 1.5
# pics + marker: 0.15, training: 1

# lonlat + lonlat + pred_lonlat & sp: 0.20, training: 2.5
# lonlat + marker + pred_lonlat & sp: 0.20, training: 2
# pic + lonlat + pred_lonlat & sp: -, training: -
# pic + marker + pred_lonlat & sp: -, training: -


# 8e5 times
# pics + lonlat + pred_lonlat & sp: 0.30, training: 3.5