import argparse
import os.path
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt

import torch


class GridWorld:
    def __init__(self, args):
        self.size = args.grids_size
        self.max_steps = args.env_max_steps
        self.device = args.device
        self.generator = torch.Generator(device=self.device)
        self.reset()

    def reset(self):
        while True:
            self.agent = torch.randint(0, self.size, (2,), generator=self.generator, device=self.device)
            self.goal = torch.randint(0, self.size, (2,), generator=self.generator, device=self.device)
            if not torch.equal(self.agent, self.goal):
                break

        self.steps = 0 # torch.tensor(0, device=self.device)
        self.prev_dist = self._dist(self.agent, self.goal)
        return self._get_obs()

    def _dist(self, a, g):
        # manhattan distance
        return torch.sum(torch.abs(a - g))

    def _get_obs(self):
        s = self.size - 1
        a_norm = self.agent.float() / s
        g_norm = self.goal.float() / s
        d_norm = (g_norm - a_norm)
        manh_norm = torch.abs(d_norm).sum() / 2  # [0,1]

        return torch.cat([a_norm, g_norm, d_norm, manh_norm.unsqueeze(0)])

    def step(self, action):
        # 定义位移向量：up, down, left, right
        move_vectors = torch.tensor([[0, -1], [0, 1], [-1, 0], [1, 0]],
                                    device=self.device, dtype=torch.long)

        # 直接计算新位置
        new_agent = self.agent + move_vectors[action]
        new_agent = torch.clamp(new_agent, 0, self.size - 1)

        self.agent = new_agent
        self.steps += 1

        # 计算距离和奖励
        dist = self._dist(self.agent, self.goal)
        shaping = 0.2 * (self.prev_dist - dist)
        self.prev_dist = dist

        reward = -0.01 + shaping
        done = torch.equal(self.agent, self.goal) or (self.steps >= self.max_steps)

        if torch.equal(self.agent, self.goal):
            reward = torch.tensor(1.0, device=self.device)

        return self._get_obs(), reward, done, {}


    def get_positions(self):
        #return tuple(self.agent.cpu().tolist()), tuple(self.goal.cpu().tolist())

        return self.agent, self.goal


class ActorCriticLSTM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args.device
        self.obs_dim = args.obs_dim
        self.action_dim = args.action_dim
        self.hidden_size = args.hidden_size

        self.obs_enc = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)
        self.actor = nn.Linear(self.hidden_size, self.action_dim)
        self.critic = nn.Linear(self.hidden_size, 1)

        # 正交初始化
        def ortho_init(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=torch.sqrt(torch.tensor(2.0)))
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name or 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)

        self.apply(ortho_init)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.constant_(self.actor.bias, 0)

    def init_hidden(self, batch_size=1):
        h0 = torch.zeros(1, batch_size, self.lstm.hidden_size, device=self.device)
        c0 = torch.zeros(1, batch_size, self.lstm.hidden_size, device=self.device)
        return (h0, c0)

    def forward(self, obs, hidden):
        # obs: [B, T, obs_dim]
        B, T, D = obs.shape
        x = self.obs_enc(obs.view(B * T, D))
        x = x.view(B, T, -1)
        out, hidden_next = self.lstm(x, hidden)
        logits = self.actor(out)               # [B, T, A]
        values = self.critic(out).squeeze(-1)  # [B, T]
        return logits, values, hidden_next


class PPOAgent:
    def __init__(self, args):
        self.device = args.device
        self.rollout_len = args.rollout_len
        self.obs_dim = args.obs_dim
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self.clip_eps = args.clip_eps
        self.epochs = args.epochs
        self.entropy_coef = args.entropy_coef
        self.value_coef = args.value_coef
        self.max_grad_norm = args.max_grad_norm

        self.net = ActorCriticLSTM(args).to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=args.lr)

        self.clear_buffer()

    def clear_buffer(self):
        self.obs_buf = torch.zeros((self.rollout_len, self.obs_dim), device=self.device)
        self.actions_buf = torch.zeros((self.rollout_len), device=self.device)
        self.rewards_buf = torch.zeros((self.rollout_len), device=self.device)
        self.dones_buf = torch.zeros((self.rollout_len), device=self.device)
        self.logprobs_buf = torch.zeros((self.rollout_len), device=self.device)
        self.values_buf = torch.zeros((self.rollout_len), device=self.device)

    @torch.no_grad()
    def act(self, obs, hidden):
        # obs: np array [obs_dim]
        obs_t = obs.unsqueeze(0).unsqueeze(0)
        logits, value, hidden_next = self.net(obs_t, hidden)
        logits = logits.squeeze()
        value = value.squeeze()

        dist = Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)

        return action, logprob, value, hidden_next



    def store(self, obs, action, reward, done, logprob, value, r):
        self.obs_buf[r] = obs
        self.actions_buf[r] = action
        self.rewards_buf[r] = reward
        self.dones_buf[r] = float(done)
        self.logprobs_buf[r] = logprob
        self.values_buf[r] = value

    def compute_gae(self, last_value, last_done):
        rewards = self.rewards_buf
        mask = 1 - self.dones_buf
        values = torch.cat((self.values_buf, last_value.unsqueeze(0)))

        advantages = torch.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * mask[t] - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask[t] * gae
            advantages[t] = gae
        returns = advantages + values[:-1]
        return advantages, returns

    def _roll_forward_to_compute(self, obs_seq, done_seq):
        # obs_seq: [T, obs_dim]; done_seq: [T]
        T = obs_seq.shape[0]
        obs_seq = obs_seq.unsqueeze(0)  # [1, T, D]
        logits_list, values_list = [], []

        hidden = self.net.init_hidden(batch_size=1)
        for t in range(T):
            logits, values, hidden = self.net(obs_seq[:, t:t+1, :], hidden)
            logits_list.append(logits[:, -1, :])
            values_list.append(values[:, -1])

            done_t = done_seq[t].view(1, 1, 1)
            h, c = hidden
            mask = (1.0 - done_t).to(self.device)
            hidden = (h * mask, c * mask)

        logits = torch.stack(logits_list, dim=1).squeeze(0)  # [T, A]
        values = torch.stack(values_list, dim=1).squeeze(0)  # [T]
        return logits, values

    def update(self, last_value, last_done):

        obs = self.obs_buf                  # [T, D]
        actions = self.actions_buf          # [T]
        old_logprobs = self.logprobs_buf    # [T]
        dones = self.dones_buf              # [T]
        old_values = self.values_buf        # [T]

        with torch.no_grad():
            advantages, returns = self.compute_gae(last_value=last_value, last_done=last_done)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.epochs):
            logits, values = self._roll_forward_to_compute(obs, dones)
            dist = Categorical(logits=logits)
            new_logprobs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            # PPO ratio
            ratio = (new_logprobs - old_logprobs).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # 正确的 value clipping：以 old_values 为基准
            values_clipped = old_values + (values - old_values).clamp(-self.clip_eps, self.clip_eps)
            v_loss1 = (values - returns).pow(2)
            v_loss2 = (values_clipped - returns).pow(2)
            value_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()

            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
            self.opt.step()

        self.clear_buffer()




class Exps:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.total_steps = args.total_steps
        self.rollout_len = args.rollout_len
        self.log_interval = args.log_interval
        self.save_root = args.save_root
        self.test_episodes = args.test_episodes
        self.max_steps = args.env_max_steps
        self.set_seed(args.seed)
        self.env = GridWorld(args)
        self.agent = PPOAgent(args)


    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


    def train(self):
        round_episode = int(self.total_steps / (self.max_steps * 0.3))
        episode_rewards = torch.zeros((round_episode))
        eps = 0
        ep_return = 0.0
        obs = self.env.reset()
        hidden = self.agent.net.init_hidden(batch_size=1)

        step = 0
        t0 = time.time()
        last_done = True  # 起始视作上一步已结束
        while step < self.total_steps:
            # 采样一段rollout
            for r in range(self.rollout_len):
                action, logprob, value, hidden_next = self.agent.act(obs, hidden)
                next_obs, reward, done, _ = self.env.step(action)

                self.agent.store(obs, action, reward, done, logprob, value, r)
                ep_return += reward
                step += 1

                hidden = hidden_next
                if done:
                    episode_rewards[eps] = ep_return
                    eps += 1
                    obs = self.env.reset()
                    ep_return = 0.0
                    hidden = self.agent.net.init_hidden(batch_size=1)
                else:
                    obs = next_obs

                last_done = done

                if step >= self.total_steps:
                    break

            # 估计 rollout 的 bootstrap 值
            with torch.no_grad():
                obs_t = obs.unsqueeze(0).unsqueeze(0)
                logits, val_t, _ = self.agent.net(obs_t, hidden)
                last_value = val_t.squeeze()

            # 更新
            last_value = torch.tensor(0, device=self.device) if last_done else last_value
            self.agent.update(last_value, last_done)
            episode_rewards_ = episode_rewards[episode_rewards != 0]
            if len(episode_rewards_) > 0 and step % self.log_interval < self.rollout_len:
                print(f"Steps: {step}/{self.total_steps}, "
                      f"LastEpR: {episode_rewards_[-1]:.3f}, "
                      f"Avg50: {torch.mean(episode_rewards_[-50:]):.3f}, "
                      f"FPS: {(step/(time.time()-t0)):.1f}")
        episode_rewards = episode_rewards[episode_rewards != 0]
        self.save_result(episode_rewards)
        return

    def save_result(self, episode_rewards):
        ep_rewards = episode_rewards.cpu().numpy()
        save_dict = {"net": self.agent.net.state_dict(),
                     "opt": self.agent.opt.state_dict()}
        torch.save(save_dict, args.save_root + "/agent.pth")
        np.save(self.save_root + "/ep_rewards.npy", ep_rewards)
        self.draw_rewads()
        return

    def draw_rewads(self):
        episode_rewards = np.load(self.save_root + "/ep_rewards.npy")
        plt.figure(figsize=(12, 8))
        plt.plot(episode_rewards, alpha = 0.5, label="Episode Return")
        if len(episode_rewards) >= 50:
            smooth = np.convolve(episode_rewards, np.ones(50) / 50, mode='valid')
            plt.plot(range(49, 49 + len(smooth)), smooth, c="#1f77b4", label="Moving Avg(50)")
        plt.axhline(y=3, color='grey', alpha = 0.7, linestyle='--',)
        plt.legend()
        plt.savefig(fname=args.save_root + f'/rewards.jpg', dpi=300, format='jpg', bbox_inches='tight')
        plt.close()
        return



    def test(self):
        cheakpoints = torch.load(self.save_root + "/agent.pth")
        self.agent.net.load_state_dict(cheakpoints["net"])
        self.agent.opt.load_state_dict(cheakpoints["opt"])

        paths = []
        for ep in range(self.test_episodes):
            obs = self.env.reset()
            positions = []
            hidden = self.agent.net.init_hidden(batch_size=1)
            for t in range(200):
                obs_t = obs.unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    logits, _, hidden = self.agent.net(obs_t, hidden)
                    action = torch.argmax(logits[:, -1, :], dim=-1).item()
                (ax, ay), (gx, gy) = self.env.get_positions()
                positions.append((ax, ay, gx, gy))
                obs, reward, done, _ = self.env.step(action)
                if done:
                    (ax, ay), (gx, gy) = self.env.get_positions()
                    positions.append((ax, ay, gx, gy))
                    break
            paths.append(positions)

        size = self.env.size
        fig, axs = plt.subplots(3, 3, figsize=(15, 15))
        for e, axe in enumerate(axs.flatten()):
            path = paths[e]
            ax0, ay0, gx, gy = path[0]
            axe.scatter([ax0 + 0.5], [ay0 + 0.5], c='blue', label='Start')
            axe.scatter([gx + 0.5], [gy + 0.5], c='red', label='Goal')
            axe.plot([p[0] + 0.5 for p in path],  [p[1] + 0.5 for p in path], '-o', markersize=2, label=f'Test_{e}')
            axe.set_xlim(0, size)
            axe.set_ylim(0, size)
            axe.legend()

        fig.tight_layout()
        fig.savefig(fname=self.save_root + f'/test.jpg', dpi=100, format='jpg', bbox_inches='tight')
        plt.close()

        return

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--total_steps", type=int, default=1e5) # 1e5
parser.add_argument("--rollout_len", type=int, default=512)
parser.add_argument("--obs_dim", type=int, default=7)
parser.add_argument("--action_dim", type=int, default=4)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--gamma", type=float, default=0.995)
parser.add_argument("--gae_lambda", type=float, default=0.95)
parser.add_argument("--clip_eps", type=float, default=0.2)
parser.add_argument("--epochs", type=int, default=6)
parser.add_argument("--entropy_coef", type=float, default=0.02)
parser.add_argument("--value_coef", type=float, default=0.5)
parser.add_argument("--max_grad_norm", type=float, default=0.5)
parser.add_argument("--hidden_size", type=int, default=128)
parser.add_argument("--log_interval", type=int, default=2000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--env_max_steps", type=int, default=60)
parser.add_argument("--grids_size", type=int, default=20)
parser.add_argument("--test_episodes", type=int, default=9)
parser.add_argument("--save_root", type=str, default="./result_ppo2")


args = parser.parse_args()
print(f"Using device: {args.device}")


if not os.path.exists(args.save_root):
    os.makedirs(args.save_root)


exp = Exps(args)
exp.train()
exp.draw_rewads()
exp.test()

