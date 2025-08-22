import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
#from torchvision import models
#import ssl
#ssl._create_default_https_context = ssl._create_unverified_context


class AgentPics(nn.Module):

    def __init__(self,hidden_feats=256, num_actions = 5, dropout = 0.5):
        super().__init__()
        self.num_actions = num_actions
        self.hidden_feats = hidden_feats
        self.dropout = dropout

        self.cnn = models.resnet18(weights='IMAGENET1K_V1')
        self.cnn.fc = nn.Linear(512, hidden_feats)

        self.pro1 = nn.Linear(2, hidden_feats // 8)
        self.pro2 = nn.Linear(2, hidden_feats // 8)

        self.lstm = nn.LSTM(input_size=hidden_feats + hidden_feats // 4, hidden_size=hidden_feats)

        self.policy_logits = nn.Linear(hidden_feats, num_actions)
        self.baseline = nn.Linear(hidden_feats, 1)


    def initial_state(self, batch_size):
        device = next(self.parameters()).device
        h = torch.zeros(1, batch_size, self.hidden_feats, device=device)
        c = torch.zeros(1, batch_size, self.hidden_feats, device=device)
        return (h, c)

    def forward(self, env_state, core_state):

        view, goal, posi = env_state
        goal = self.pro1(goal)
        posi = self.pro1(posi)
        emb_ = self.cnn(view)
        emb = torch.cat([emb_, goal, posi], dim=1).unsqueeze(0)

        core_out, _core_state = self.lstm(emb, core_state)
        core_out = F.dropout(core_out, p=self.dropout, training=self.training)

        policy_logits = self.policy_logits(core_out).squeeze()
        baseline = self.baseline(core_out).squeeze()
        action_dist = Categorical(logits=policy_logits)
        new_action = action_dist.sample()

        return new_action, policy_logits, baseline, _core_state





class AgentPosi(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.num_actions = args.num_actions
        self.hidden_feats =  args.hidden_feats
        self.state_fests = args.state_fests
        self.batch_size = args.batch_size
        self.device = args.device


        self.lstm = nn.LSTM(input_size=self. hidden_feats, hidden_size=self.hidden_feats)

        self.embeding = nn.Sequential(
            nn.Linear(self.state_fests, self.hidden_feats//2),
            nn.ReLU(),
            nn.Linear(self.hidden_feats//2, self.hidden_feats),
        )
        self.policy_logits = nn.Sequential(
            nn.Linear(self.hidden_feats, self.hidden_feats//2),
            nn.ReLU(),
            nn.Linear(self.hidden_feats//2, self.num_actions),
        )
        self.baseline = nn.Sequential(
            nn.Linear(self.hidden_feats, self.hidden_feats//2),
            nn.ReLU(),
            nn.Linear(self.hidden_feats//2, 1),
        )


    def initial_state(self, batch_):

        h = torch.zeros(1, batch_, self.hidden_feats, device=self.device)
        c = torch.zeros(1, batch_, self.hidden_feats, device=self.device)
        return (h, c)

    def forward(self, env_state, agent_state, iftrain=None):
        if iftrain ==None:
            embed = self.embeding(env_state).unsqueeze(0)
        else:
            embed = self.embeding(env_state)
        out, next_state = self.lstm(embed, agent_state)

        policy_logits = self.policy_logits(out)#.squeeze()
        baseline = self.baseline(out)#.squeeze()

        return policy_logits.squeeze(), baseline.squeeze(), next_state

class AgentMlp(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.num_actions = args.num_actions
        self.hidden_feats =  args.hidden_feats
        self.state_fests = args.state_fests
        self.batch_size = args.batch_size
        self.device = args.device

        self.embeding = nn.Sequential(
            nn.Linear(self.state_fests, self.hidden_feats//2),
            nn.ReLU(),
            nn.Linear(self.hidden_feats//2, self.hidden_feats),
        )
        self.policy_logits = nn.Sequential(
            nn.Linear(self.hidden_feats, self.hidden_feats//2),
            nn.ReLU(),
            nn.Linear(self.hidden_feats//2, self.num_actions),
        )
        self.baseline = nn.Sequential(
            nn.Linear(self.hidden_feats, self.hidden_feats//2),
            nn.ReLU(),
            nn.Linear(self.hidden_feats//2, 1),
        )


    def forward(self, env_state):

        embed = self.embeding(env_state).unsqueeze(0)
        policy_logits = self.policy_logits(embed)#.squeeze()
        baseline = self.baseline(embed)#.squeeze()

        return policy_logits.squeeze(), baseline.squeeze()


if __name__ == "__main__":
    # test pos
    import argparse

    parser = argparse.ArgumentParser(description='Mapless_Navigation')
    parser.add_argument('--num_actions', default=4, type=int)
    parser.add_argument('--hidden_feats', default=128, type=int)
    parser.add_argument('--state_fests', default=4, type=int)
    parser.add_argument('--batch_size', default=32, type=int)

    args = parser.parse_args()

    agent = AgentPosi(args)
    _state = agent.initial_state()
    env_state = torch.randn((32, 4))
    _traj = {
        "actions": torch.empty((50, args.batch_size, 1)),
        "logits": torch.empty(50, args.batch_size,  4),
        "baselines": torch.empty(50, args.batch_size,  1)
    }
    for i in range(50):
        action, policy_logits, baseline, next_state = agent(env_state, _state)
        _traj["actions"][i ,:,  :] = action
        _traj["logits"][i,:, :] = policy_logits
        _traj["baselines"][i, :, :] = baseline
    print(';')






    # test pano
    dim = 256
    batch_size = 32
    episodes = 15

    view_image = torch.randn((batch_size, 3, 225, 225))
    goal = torch.randn((batch_size, 2))
    posi = torch.randn((batch_size, 2))
    env_state = (view_image, goal, posi)

    agent = AgentPics()
    core_state = agent.initial_state(batch_size)
    tra = {
        "actions":[],
        "logits": [],
        "baseline": [],
    }
    for _ in range(episodes):
        action, logits, baseline, _core_state = agent(env_state, core_state)
        tra["actions"].append(action)
        tra["logits"].append(logits)
        tra["baseline"].append(baseline)

        env_state = (view_image, goal, posi)
        core_state = _core_state
        print(_)
    print(';')

