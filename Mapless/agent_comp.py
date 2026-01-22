import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torchvision import models
import ssl
ssl._create_default_https_context = ssl._create_unverified_context





class Base(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args.device
        self.state_fests = args.state_fests
        self.num_actions = args.num_actions
        self.hidden_feats = args.hidden_feats
        self.lstm = nn.LSTM(self.hidden_feats, self.hidden_feats)
        self.actor = nn.Linear(self.hidden_feats, self.num_actions)
        self.critic = nn.Linear(self.hidden_feats, 1)
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
        return obs



class lonlat_lonlat(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args.device
        self.state_fests = args.state_fests
        self.num_actions = args.num_actions
        self.hidden_feats = args.hidden_feats
        self.lstm = nn.LSTM(self.hidden_feats, self.hidden_feats)
        self.actor = nn.Linear(self.hidden_feats, self.num_actions)
        self.critic = nn.Linear(self.hidden_feats, 1)
        self.cnn = nn.Linear(1000, self.hidden_feats)
        self.fusion = nn.Linear(self.hidden_feats * 2, self.hidden_feats)
        self.obs_enc = nn.Sequential(
            nn.Linear(4, self.hidden_feats),
            nn.ReLU(),
            nn.Linear(self.hidden_feats, self.hidden_feats),
            nn.ReLU(),
        )
    def init_hidden(self, batch_size=1):
        h0 = torch.zeros(1, batch_size, self.hidden_feats, device=self.device)
        c0 = torch.zeros(1, batch_size, self.hidden_feats, device=self.device)
        return (h0, c0)

    def loss_mse(self, da, da1):
        return torch.sum((da - da1) ** 2, dim=-1).mean()

    def loss_en(self, logits, labels):
        return F.cross_entropy(logits, labels, reduction='sum')


    def forward(self, obs, hidden, views = 0):
        obs_ = obs[:, 6:10]

        x = self.obs_enc(obs_).unsqueeze(1)
        if isinstance(views, torch.Tensor):
            view_enc = self.cnn(views).unsqueeze(1)
            x = self.fusion(torch.cat((x, view_enc),dim=-1))

        out, hidden_next = self.lstm(x, hidden)
        logits = self.actor(out).squeeze()
        values = self.critic(out).squeeze()
        loss = 0
        return logits, values, hidden_next, loss


agents_name = ["lonlat_lonlat", "lonlat_marker", "pics_lonlat", "pics_marker",
          "pics_marker_conti", "pics_marker_Rconti", "pics_marker_disc"]

class lonlat_marker(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args.device
        self.state_fests = args.state_fests
        self.num_actions = args.num_actions
        self.hidden_feats = args.hidden_feats
        self.lstm = nn.LSTM(self.hidden_feats, self.hidden_feats)
        self.actor = nn.Linear(self.hidden_feats, self.num_actions)
        self.critic = nn.Linear(self.hidden_feats, 1)
        self.cnn = nn.Linear(1000, self.hidden_feats)
        self.fusion = nn.Linear(self.hidden_feats * 2, self.hidden_feats)
        self.obs_enc = nn.Sequential(
            nn.Linear(200, self.hidden_feats),
            nn.ReLU(),
            nn.Linear(self.hidden_feats, self.hidden_feats),
            nn.ReLU(),
        )

    def init_hidden(self, batch_size=1):
        h0 = torch.zeros(1, batch_size, self.hidden_feats, device=self.device)
        c0 = torch.zeros(1, batch_size, self.hidden_feats, device=self.device)
        return (h0, c0)

    def loss_mse(self, da, da1):
        return torch.sum((da - da1) ** 2, dim=-1).mean()

    def loss_en(self, logits, labels):
        return F.cross_entropy(logits, labels, reduction='sum')

    def forward(self, obs, hidden, views = 0):
        obs_ = obs[:, 10:]

        x = self.obs_enc(obs_).unsqueeze(1)
        if isinstance(views, torch.Tensor):
            view_enc = self.cnn(views).unsqueeze(1)
            x = self.fusion(torch.cat((x, view_enc),dim=-1))

        out, hidden_next = self.lstm(x, hidden)
        logits = self.actor(out).squeeze()
        values = self.critic(out).squeeze()
        loss = 0
        return logits, values, hidden_next, loss

agents_name = ["lonlat_lonlat", "lonlat_marker", "pics_lonlat", "pics_marker",
          "pics_marker_conti", "pics_marker_Rconti", "pics_marker_disc"]


class pics_lonlat(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args.device
        self.state_fests = args.state_fests
        self.num_actions = args.num_actions
        self.hidden_feats = args.hidden_feats
        self.lstm = nn.LSTM(self.hidden_feats, self.hidden_feats)
        self.actor = nn.Linear(self.hidden_feats, self.num_actions)
        self.critic = nn.Linear(self.hidden_feats, 1)
        self.cnn = nn.Linear(1000, self.hidden_feats)
        self.fusion = nn.Linear(self.hidden_feats * 2, self.hidden_feats)
        self.obs_enc = nn.Sequential(
            nn.Linear(2, self.hidden_feats),
            nn.ReLU(),
            nn.Linear(self.hidden_feats, self.hidden_feats),
            nn.ReLU(),
        )

    def init_hidden(self, batch_size=1):
        h0 = torch.zeros(1, batch_size, self.hidden_feats, device=self.device)
        c0 = torch.zeros(1, batch_size, self.hidden_feats, device=self.device)
        return (h0, c0)

    def loss_mse(self, da, da1):
        return torch.sum((da - da1) ** 2, dim=-1).mean()

    def loss_en(self, logits, labels):
        return F.cross_entropy(logits, labels, reduction='sum')

    def forward(self, obs, hidden, views=0):
        obs_ = obs[:, 8:10]

        x = self.obs_enc(obs_).unsqueeze(1)
        if isinstance(views, torch.Tensor):
            view_enc = self.cnn(views).unsqueeze(1)
            x = self.fusion(torch.cat((x, view_enc), dim=-1))

        out, hidden_next = self.lstm(x, hidden)
        logits = self.actor(out).squeeze()
        values = self.critic(out).squeeze()
        loss = 0
        return logits, values, hidden_next, loss

agents_name = ["lonlat_lonlat", "lonlat_marker", "pics_lonlat", "pics_marker",
          "pics_marker_conti", "pics_marker_Rconti", "pics_marker_disc"]

class pics_marker(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args.device
        self.state_fests = args.state_fests
        self.num_actions = args.num_actions
        self.hidden_feats = args.hidden_feats
        self.lstm = nn.LSTM(self.hidden_feats, self.hidden_feats)
        self.actor = nn.Linear(self.hidden_feats, self.num_actions)
        self.critic = nn.Linear(self.hidden_feats, 1)
        self.cnn = nn.Linear(1000, self.hidden_feats)
        self.fusion = nn.Linear(self.hidden_feats * 2, self.hidden_feats)
        self.obs_enc = nn.Sequential(
            nn.Linear(100, self.hidden_feats),
            nn.ReLU(),
            nn.Linear(self.hidden_feats, self.hidden_feats),
            nn.ReLU(),
        )

    def init_hidden(self, batch_size=1):
        h0 = torch.zeros(1, batch_size, self.hidden_feats, device=self.device)
        c0 = torch.zeros(1, batch_size, self.hidden_feats, device=self.device)
        return (h0, c0)

    def loss_mse(self, da, da1):
        return torch.sum((da - da1) ** 2, dim=-1).mean()

    def loss_en(self, logits, labels):
        return F.cross_entropy(logits, labels, reduction='sum')

    def forward(self, obs, hidden, views=0):
        obs_ = obs[:, 110:]

        x = self.obs_enc(obs_).unsqueeze(1)
        if isinstance(views, torch.Tensor):
            view_enc = self.cnn(views).unsqueeze(1)
            x = self.fusion(torch.cat((x, view_enc), dim=-1))

        out, hidden_next = self.lstm(x, hidden)
        logits = self.actor(out).squeeze()
        values = self.critic(out).squeeze()
        loss = 0
        return logits, values, hidden_next, loss



class pics_marker_conti(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args.device
        self.state_fests = args.state_fests
        self.num_actions = args.num_actions
        self.hidden_feats = args.hidden_feats
        self.lstm = nn.LSTM(self.hidden_feats, self.hidden_feats)
        self.actor = nn.Linear(self.hidden_feats, self.num_actions)
        self.critic = nn.Linear(self.hidden_feats, 1)
        self.cnn = nn.Linear(1000, self.hidden_feats)
        self.fusion = nn.Linear(self.hidden_feats * 2, self.hidden_feats)
        self.obs_enc = nn.Sequential(
            nn.Linear(100, self.hidden_feats),
            nn.ReLU(),
            nn.Linear(self.hidden_feats, self.hidden_feats),
            nn.ReLU(),
        )
        self.pred_obs = nn.Linear(self.hidden_feats, 2)
        self.pred_goal = nn.Linear(self.hidden_feats, 2)

    def init_hidden(self, batch_size=1):
        h0 = torch.zeros(1, batch_size, self.hidden_feats, device=self.device)
        c0 = torch.zeros(1, batch_size, self.hidden_feats, device=self.device)
        return (h0, c0)

    def loss_mse(self, da, da1):
        return torch.sum((da - da1) ** 2, dim=-1).mean()

    def loss_en(self, logits, labels):
        return F.cross_entropy(logits, labels, reduction='sum')

    def forward(self, obs, hidden, views = 0):

        tra_x, tra_y = obs[:, 6:8], obs[:, 8:10]
        obs_ = obs[:, 110:]

        x = self.obs_enc(obs_).unsqueeze(1)
        if isinstance(views, torch.Tensor):
            view_enc = self.cnn(views).unsqueeze(1)
            x = self.fusion(torch.cat((x, view_enc),dim=-1))

        out, hidden_next = self.lstm(x, hidden)
        logits = self.actor(out).squeeze()
        values = self.critic(out).squeeze()

        obs_pred = self.pred_obs(out).squeeze(1)
        goal_pred = self.pred_goal(out).squeeze(1)
        loss_obs = self.loss_mse(obs_pred, tra_x)
        loss_goal = self.loss_mse(goal_pred, tra_y)
        loss = loss_obs + loss_goal
        return logits, values, hidden_next, loss



agents_name = ["lonlat_lonlat", "lonlat_marker", "pics_lonlat", "pics_marker",
               "pics_marker_conti", "pics_marker_Rconti", "pics_marker_disc"]

class pics_marker_Rconti(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args.device
        self.state_fests = args.state_fests
        self.num_actions = args.num_actions
        self.hidden_feats = args.hidden_feats
        self.lstm = nn.LSTM(self.hidden_feats, self.hidden_feats)
        self.actor = nn.Linear(self.hidden_feats, self.num_actions)
        self.critic = nn.Linear(self.hidden_feats, 1)
        self.cnn = nn.Linear(1000, self.hidden_feats)
        self.fusion = nn.Linear(self.hidden_feats * 2, self.hidden_feats)
        self.obs_enc = nn.Sequential(
            nn.Linear(100, self.hidden_feats),
            nn.ReLU(),
            nn.Linear(self.hidden_feats, self.hidden_feats),
            nn.ReLU(),
        )
        self.pred_sp = nn.Linear(self.hidden_feats, 2)

    def init_hidden(self, batch_size=1):
        h0 = torch.zeros(1, batch_size, self.hidden_feats, device=self.device)
        c0 = torch.zeros(1, batch_size, self.hidden_feats, device=self.device)
        return (h0, c0)

    def loss_mse(self, da, da1):
        return torch.sum((da - da1) ** 2, dim=-1).mean()

    def loss_en(self, logits, labels):
        return F.cross_entropy(logits, labels, reduction='sum')

    def forward(self, obs, hidden, views = 0):

        lonlat_x, lonlat_y = obs[:, 6:8], obs[:, 8:10]
        tar_sp = lonlat_y - lonlat_x

        obs_ = obs[:, 110:]

        x = self.obs_enc(obs_).unsqueeze(1)
        if isinstance(views, torch.Tensor):
            view_enc = self.cnn(views).unsqueeze(1)
            x = self.fusion(torch.cat((x, view_enc),dim=-1))

        out, hidden_next = self.lstm(x, hidden)
        logits = self.actor(out).squeeze()
        values = self.critic(out).squeeze()

        sp_pred = self.pred_sp(out).squeeze(1)
        loss_sp = self.loss_mse(sp_pred, tar_sp)

        return logits, values, hidden_next, loss_sp



class pics_marker_disc(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args.device
        self.state_fests = args.state_fests
        self.num_actions = args.num_actions
        self.hidden_feats = args.hidden_feats
        self.lstm = nn.LSTM(self.hidden_feats, self.hidden_feats)
        self.actor = nn.Linear(self.hidden_feats, self.num_actions)
        self.critic = nn.Linear(self.hidden_feats, 1)
        self.cnn = nn.Linear(1000, self.hidden_feats)
        self.fusion = nn.Linear(self.hidden_feats * 2, self.hidden_feats)
        self.obs_enc = nn.Sequential(
            nn.Linear(100, self.hidden_feats),
            nn.ReLU(),
            nn.Linear(self.hidden_feats, self.hidden_feats),
            nn.ReLU(),
        )
        self.pred_obs = nn.Linear(self.hidden_feats, 256)
        self.pred_goal = nn.Linear(self.hidden_feats, 256)


    def init_hidden(self, batch_size=1):
        h0 = torch.zeros(1, batch_size, self.hidden_feats, device=self.device)
        c0 = torch.zeros(1, batch_size, self.hidden_feats, device=self.device)
        return (h0, c0)

    def loss_mse(self, da, da1):
        return torch.sum((da - da1) ** 2, dim=-1).mean()

    def loss_en(self, logits, labels):
        return F.cross_entropy(logits, labels, reduction='sum')

    def forward(self, obs, hidden, views = 0):

        tra_x, tra_y =  obs[:, 0].long(), obs[:, 1].long()
        obs_ = obs[:, 110:]

        x = self.obs_enc(obs_).unsqueeze(1)
        if isinstance(views, torch.Tensor):
            view_enc = self.cnn(views).unsqueeze(1)
            x = self.fusion(torch.cat((x, view_enc),dim=-1))

        out, hidden_next = self.lstm(x, hidden)
        logits = self.actor(out).squeeze()
        values = self.critic(out).squeeze()

        obs_pred = self.pred_obs(out).squeeze(1)
        goal_pred = self.pred_goal(out).squeeze(1)
        loss_obs = self.loss_en(obs_pred, tra_x)
        loss_goal = self.loss_en(goal_pred, tra_y)
        loss = loss_obs + loss_goal

        return logits, values, hidden_next, loss