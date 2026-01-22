import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from environment import StreetEnv, StreetPicEnv, GridEnv
from agent import AgentPosi
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from draw import Draw_result
import pandas as pd
import os
from agent_comp import *



class PPOAgent:
    def __init__(self, args):
        self.args = args
        self.env_name = args.env_name
        self.device = args.device
        self.rollout_len = args.rollout_len
        self.state_fests = args.state_fests
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self.clip_eps = args.clip_eps
        self.update_times = args.update_times
        self.entropy_coef = args.entropy_coef
        self.value_coef = args.value_coef
        self.max_grad_norm = args.max_grad_norm
        self.net = None
        self.opt = None
        self.clear_buffer()

    def clear_buffer(self):
        self.obs_buf = torch.zeros((self.rollout_len, self.state_fests), device=self.device)
        self.actions_buf = torch.zeros((self.rollout_len), device=self.device)
        self.rewards_buf = torch.zeros((self.rollout_len), device=self.device)
        self.dones_buf = torch.zeros((self.rollout_len), device=self.device)
        self.logprobs_buf = torch.zeros((self.rollout_len), device=self.device)
        self.values_buf = torch.zeros((self.rollout_len), device=self.device)
        if self.env_name == "StreetPicEnv":
            self.views_buf = torch.zeros((self.rollout_len, 1), device=self.device)

    @torch.no_grad()
    def act(self, obs, hidden):
        if self.env_name == "StreetPicEnv":
            obs_t = obs[0].unsqueeze(0)
            logits, value, hidden_next, _ = self.net(obs_t, hidden, obs[1])
        else:
            obs_t = obs.unsqueeze(0)
            logits, value, hidden_next, _ = self.net(obs_t, hidden)
        dist = Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action, logprob, value, hidden_next

    def store(self, obs, action, reward, done, logprob, value, r):
        if self.env_name == "StreetPicEnv":
            self.obs_buf[r] = obs[0]
            self.views_buf[r] = obs[-1]
        else:
            self.obs_buf[r] = obs
        self.actions_buf[r] = action
        self.rewards_buf[r] = reward
        self.dones_buf[r] = done
        self.logprobs_buf[r] = logprob
        self.values_buf[r] = value

    def compute_gae(self, last_value):
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

    def update(self, last_value, views_up = 0):

        obs = self.obs_buf                  # [T, D]
        actions = self.actions_buf          # [T]
        old_logprobs = self.logprobs_buf    # [T]
        old_values = self.values_buf        # [T]

        with torch.no_grad():
            advantages, returns = self.compute_gae(last_value)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.update_times):
            if self.env_name == "StreetPicEnv" :
                logits, values, _, loss_pro = self.net(obs, self.net.init_hidden(batch_size=1), views_up)
            else:
                logits, values, _, loss_pro = self.net(obs, self.net.init_hidden(batch_size=1))
            dist = Categorical(logits=logits)
            new_logprobs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            # PPO ratio
            ratio = (new_logprobs - old_logprobs).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # value clipping
            values_clipped = old_values + (values - old_values).clamp(-self.clip_eps, self.clip_eps)
            v_loss1 = (values - returns).pow(2)
            v_loss2 = (values_clipped - returns).pow(2)
            value_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()

            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy + loss_pro
            #loss = policy_loss + self.value_coef * value_loss + loss_pro #* 10

            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
            self.opt.step()
        self.clear_buffer()


class Exps:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.agent_name = args.agent_name
        self.env_name = args.env_name
        self.total_steps = args.total_steps
        self.rollout_len = args.rollout_len
        self.log_interval = args.log_interval
        self.save_dir = args.save_dir
        self.test_episodes = args.test_episodes
        self.max_steps = args.max_steps
        self.visual = args.visual
        #self.set_seed(args.seed)
        self.Avg50 = 0
        self.build_()

    def take_net(self, obs, hidden):
        if not self.args.env_name == "StreetPicEnv":
            obs_t = obs.unsqueeze(0)
            logits, value, hidden_next, _ = self.agent.net(obs_t, hidden)
        else:
            obs_t = obs[0].unsqueeze(0)
            logits, value, hidden_next, _ = self.agent.net(obs_t, hidden, obs[1])
        return logits, value, hidden_next

    def build_(self):
        dict_net = {'AgentPosi': AgentPosi,

                    "lonlat_lonlat": lonlat_lonlat,
                    "lonlat_marker":lonlat_marker,
                    "pics_lonlat":pics_lonlat,
                    "pics_marker": pics_marker,
                    "pics_marker_conti": pics_marker_conti,
                    "pics_marker_Rconti": pics_marker_Rconti,
                    "pics_marker_disc": pics_marker_disc
                    }

        dict_env = {'StreetPicEnv': StreetPicEnv,
                    'StreetEnv': StreetEnv,
                    "GridEnv":GridEnv
                    }

        self.env = dict_env[self.env_name](self.args)
        self.agent = PPOAgent(self.args)
        self.agent.net = dict_net[self.agent_name](self.args).to(self.device)
        self.agent.opt = torch.optim.Adam(self.agent.net.parameters(), lr=self.args.learning_rate)


    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def train(self):
        round_episode = int(self.total_steps / (self.max_steps * 0.3))
        episode_rewards = torch.zeros((round_episode))
        step_rewards = torch.zeros((int(self.total_steps)))
        eps = 0
        ep_return = 0.0
        obs = self.env.reset()
        hidden = self.agent.net.init_hidden(batch_size=1)


        t0 = time.time()
        last_done = True
        step = 0
        while step < self.total_steps:
            #for r in tqdm(range(self.rollout_len)):
            for r in range(self.rollout_len):
                action, logprob, value, hidden_next = self.agent.act(obs, hidden)
                next_obs, reward, done, _ = self.env.step(action)

                self.agent.store(obs, action, reward, done, logprob, value, r)
                ep_return += reward
                step_rewards[step] = reward
                step += 1
                hidden = hidden_next
                if done:
                    episode_rewards[eps] = ep_return
                    eps += 1
                    obs = self.env.reset()
                    ep_return = 0.0
                    hidden = self.agent.net.init_hidden(batch_size = 1)
                else:
                    obs = next_obs

                last_done = done
                if step >= self.total_steps:
                    break

            with torch.no_grad():
                logits, val_t, _ = self.take_net(obs, hidden)
                last_value = val_t.squeeze()

            last_value = torch.tensor(0, device=self.device) if last_done else last_value
            views = self.env.get_views(self.agent.views_buf.squeeze().long()) if not isinstance(obs[1], int) else 0
            self.agent.update(last_value, views)
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
        torch.save(save_dict, self.save_dir + "/agent.pth")
        np.save(self.save_dir + "/ep_rewards.npy", ep_rewards)
        return


    def test(self):
        cheakpoints = torch.load(self.save_dir + "/agent.pth")
        self.agent.net.load_state_dict(cheakpoints["net"])
        self.agent.opt.load_state_dict(cheakpoints["opt"])

        paths = []
        for ep in range(9):
            obs = self.env.reset()
            hidden = self.agent.net.init_hidden(batch_size=1)
            positions = []
            for t in range(self.max_steps):
                #obs_t = obs.unsqueeze(0)
                logits, _, hidden = self.take_net(obs, hidden)
                #action = torch.argmax(logits)#, dim=-1)#.item()
                dist = Categorical(logits=logits)
                action = dist.sample()
                obs, pos, done = self.env.step(action, iftest = True)
                positions.append(pos)
                if done:
                    positions.append(pos)
                    break
            paths.append(torch.stack(positions, dim=0))

        self.draw_ = Draw_result(self.args)
        self.draw_.draw_trajs(paths)
        self.draw_.draw_rewads()
        self.finish_ratio()
        return

    def finish_ratio_(self):
        finish = 0
        steps_use = np.zeros((self.test_episodes))
        for e in range(self.test_episodes):
            obs = self.env.reset()
            hidden = self.agent.net.init_hidden(batch_size=1)
            for t in range(self.max_steps):
                #obs_t = obs.unsqueeze(0)
                logits, _, hidden = self.take_net(obs, hidden)
                #action = torch.argmax(logits)#, dim=-1)#.item()
                dist = Categorical(logits=logits)
                action = dist.sample()
                obs, _, done = self.env.step(action, iftest = True)
                if done and t < self.max_steps-1:
                    finish += 1
                    break
            steps_use[e] = t
        print(f"finish_ratio:{finish /self.test_episodes:.2f}, ave_steps:{np.mean(steps_use):.2f}")


    def finish_ratio(self):
        finish = 0
        steps_use = np.zeros((self.test_episodes))
        for e in range(self.test_episodes):
            obs = self.env.reset()
            hidden = self.agent.net.init_hidden(batch_size=1)
            for t in range(self.max_steps):
                logits, _, hidden = self.take_net(obs, hidden)
                dist = Categorical(logits=logits)
                action = dist.sample()
                obs, _, done = self.env.step(action, iftest=True)
                if done and t < self.max_steps - 1:
                    finish += 1
                    break
            steps_use[e] = t

        finish_ratio_val = finish / self.test_episodes
        ave_steps_val = np.mean(steps_use)

        print(f"finish_ratio:{finish_ratio_val:.2f}, ave_steps:{ave_steps_val:.2f}")


        csv_file = self.save_dir + '/pic/metric.csv'
        header = not os.path.exists(csv_file)
        pd.DataFrame({
            'finish_ratio': [finish_ratio_val],
            'ave_steps': [ave_steps_val]
        }).to_csv(csv_file, mode='a', header=header, index=False)


