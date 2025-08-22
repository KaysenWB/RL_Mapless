import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from environment import CourierEnv
from agent import AgentPosi, AgentPics, AgentMlp



class PPOTrainer:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.agent_name = args.agent_name
        self.env_name = args.env_name
        self.save_dir = args.save_dir
        self.max_len = args.max_len
        self.num_actions = args.num_actions
        self.state_fests = args.state_fests
        self.epoch = args.epoch
        self.update_times = args.update_times
        self.batch_size = args.batch_size
        self.batch_train = args.batch_train
        self.gamma = args.gamma
        self.clip_epsilon = args.clip_epsilon
        self.ent_coef = args.ent_coef
        self.learning_rate = args.learning_rate
        self.re_len = self.update_times * self.max_len // (self.batch_train // self.batch_size)
        self.acc_return = -1000
        self.de_loss = 1e4
        self.build_()

    def build_(self):
        dict_agent = {'AgentPosi':AgentPosi, "AgentPics": AgentPics, "AgentMlp":AgentMlp}
        dict_env = {'CourierEnv': CourierEnv}

        self.agent = dict_agent[self.agent_name](self.args).to(self.device)
        self.env = dict_env[self.env_name](self.args)
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=self.learning_rate)


    def save_model(self):

        model_path = self.save_dir + '/' + self.agent_name + '_best.tar'
        torch.save({'state_dict': self.agent.state_dict()}, model_path)
        print(f'Saved model: {model_path}')

    def load_model(self):

        model_path = self.save_dir + '/' + self.agent_name + '_best.tar'
        checkpoint = torch.load(model_path)
        self.agent.load_state_dict(checkpoint['state_dict'])
        print(f'Loaded model: {model_path}')

    @torch.no_grad()
    def collect_trajectories(self):

        traj_data = {
            "actions": torch.empty(( self.max_len, self.batch_size, 1), device=self.device),
            "log_prob": torch.empty(self.max_len, self.batch_size, 1, device=self.device),
            "baselines": torch.empty(self.max_len, self.batch_size, 1, device=self.device),
            "env_states": torch.empty((self.max_len, self.batch_size, self.state_fests), device=self.device),
            "dones": torch.empty(self.max_len, self.batch_size, 1, dtype=torch.bool, device=self.device),
            "rewards": torch.empty(self.max_len, self.batch_size, 1, device=self.device)
        }
        if not self.agent_name == 'AgentMlp':
            agent_state = self.agent.initial_state(batch_=self.batch_size)
        env_state, done, reward = self.env.reset()  # view_image, goal, posi or goal, posi

        for t_id in range(self.max_len):
            if not self.agent_name == 'AgentMlp':
                logits, baseline, agent_state = self.agent(env_state, agent_state)
            else:
                logits, baseline = self.agent(env_state)
            policy_dist = Categorical(logits=logits)
            action = policy_dist.sample()
            log_prob = policy_dist.log_prob(action)

            env_state, done, reward = self.env.step(action.unsqueeze(1))

            traj_data["actions"][t_id, :, :] = action.unsqueeze(1)
            traj_data["log_prob"][t_id, :,  :] = log_prob.unsqueeze(1)
            traj_data["baselines"][t_id, :, :] = baseline.unsqueeze(1)
            traj_data["env_states"][ t_id, :, :] = env_state
            traj_data["dones"][t_id, :, :] = done
            traj_data["rewards"][t_id, :,  :] = reward

        return traj_data


    def return_advantage(self, traj):

        traj_dones = traj['dones']
        traj_rewards = traj["rewards"]
        traj_baselines = traj["baselines"]

        traj_return = torch.zeros_like(traj_rewards).to(self.device)
        roll = torch.zeros_like(traj_rewards[-1]).to(self.device)

        for i in reversed(range(self.max_len)):
            roll[traj_dones[i]] = 0
            roll = traj_rewards[i] + roll * self.gamma
            traj_return[i] = roll

        traj_return_s = (traj_return - traj_return.mean()) / (traj_return.std() + 1e-8)
        advantages = traj_return_s - traj_baselines
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        traj['returns'] = traj_return
        traj['returns_s'] = traj_return_s
        traj['advantages'] = advantages

        return traj
    def update(self, traj):
        #Re_Loss = torch.zeros((self.update_times*4, 3)).to(self.device)
        Re_Loss = torch.zeros((self.max_len - self.batch_train, 3)).to(self.device)

        re = 0

        traj_actions = traj['actions']
        traj_env_states = traj['env_states']
        traj_returns = traj['returns']
        traj_returns_s = traj['returns_s']
        traj_log_prob = traj['log_prob']
        traj_advantages = traj['advantages']

        #dataset = torch.utils.data.TensorDataset(
        #    traj_actions, traj_env_states, traj_returns, traj_returns_s,traj_log_prob, traj_advantages)
        #loader = torch.utils.data.DataLoader(dataset, batch_size=self.max_len//4, shuffle=False)
        Batch = (traj_actions, traj_env_states, traj_returns, traj_returns_s,traj_log_prob, traj_advantages)

        for epoch in range(1):
            for i in range(0, self.max_len-self.batch_train, 4):
                Batch_ = [ba[i:i+self.batch_train] for ba in Batch]
                traj_actions, traj_env_states, traj_returns, traj_returns_s, traj_log_prob, traj_advantages = Batch_
                #for batch in loader:
            #    traj_actions, traj_env_states, traj_returns, traj_returns_s, traj_log_prob, traj_advantages = batch

                traj_actions = traj_actions.squeeze()
                traj_log_prob = traj_log_prob.squeeze()
                traj_advantages = traj_advantages.squeeze()
                traj_returns = traj_returns.squeeze()
                traj_returns_s = traj_returns_s.squeeze()

                agent_state = self.agent.initial_state(batch_=self.batch_size)
                logits, baseline, agent_state = self.agent(traj_env_states, agent_state, iftrain=True)

                # Calculate policy loss
                policy_dist = Categorical(logits=logits)
                new_log_probs = policy_dist.log_prob(traj_actions)
                entropy = policy_dist.entropy().mean()

                ratio = (new_log_probs - traj_log_prob).exp()
                surr1 = ratio * traj_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * traj_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(baseline, traj_returns_s)
                loss = policy_loss + 0.5 * value_loss - self.ent_coef * entropy
                #loss = -(new_log_probs * traj_returns_s).mean()
                #loss = value_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 1)
                self.optimizer.step()

                Re_Loss[re, 0] = policy_loss
                Re_Loss[re, 1] = value_loss
                Re_Loss[re, 2] = loss
                re = re + 1
        loss_mean = torch.mean(Re_Loss, 0)



        return loss_mean


    def save_traj(self):
        traj = self.collect_trajectories()
        traj = self.return_advantage(traj)
        Re = torch.mean(traj['returns'])


        env_states = traj['env_states'][:,:,:4]
        env_states = env_states.detach().cpu().numpy()
        np.save(self.save_dir + '/traj_state', env_states)
        print(f'Average Return: {Re:.4f}, Saved Trajectory')

    def train(self):

        for e_ in range(self.epoch):
            print(f'Epoch: {e_}')
            traj = self.collect_trajectories()
            traj = self.return_advantage(traj)
            loss_mean = self.update(traj)

            Re = torch.mean(traj['returns'])
            if Re > self.acc_return:
                self.save_model()
                self.save_traj()
                self.acc_return = Re
            if loss_mean[2] < self.de_loss:
                self.de_loss = loss_mean[2]
            print(
                f"Policy_Loss: {loss_mean[0]:.6f}, Value_loss: {loss_mean[1]:.6f}, Loss: {loss_mean[2]:.6f}, Return: {Re:.4f}")
            print(f'Average Return: {self.acc_return:.4f}, Min_Loss: {self.de_loss:.4f}')


    def test(self):
        self.load_model()
        self.save_traj()


    def update2(self, traj):

        Re_Loss = torch.zeros((self.max_len * self.update_times, 3)).to(self.device)
        re = 0

        for start in range(0, self.update_times):

            if not self.agent_name == 'AgentMlp':
                agent_state = self.agent.initial_state(batch_=self.batch_size)



            for t in range(self.max_len):

                traj_actions = traj['actions'][t].squeeze()
                traj_env_states = traj['env_states'][t]
                traj_returns = traj['returns'][t].squeeze()
                traj_log_prob = traj['log_prob'][t].squeeze()
                traj_advantages = traj['advantages'][t].squeeze()

                if not self.agent_name == 'AgentMlp':
                    logits, baseline, _agent_state = self.agent(traj_env_states, agent_state)
                    agent_state = (_agent_state[0].detach(), _agent_state[1].detach())
                else:
                    logits, baseline = self.agent(traj_env_states)

                # Calculate policy loss
                policy_dist = Categorical(logits=logits)
                new_log_probs = policy_dist.log_prob(traj_actions)
                entropy = policy_dist.entropy().mean()

                ratio = (new_log_probs - traj_log_prob).exp()
                surr1 = ratio * traj_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * traj_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(baseline, traj_returns)
                loss = policy_loss + 0.5 * value_loss - self.ent_coef * entropy
                #loss = -(new_log_probs * traj_returns).mean()

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 1)
                self.optimizer.step()

                Re_Loss[re, 0] = policy_loss
                Re_Loss[re, 1] = value_loss
                Re_Loss[re, 2] = loss
                re = re + 1
        loss_mean = torch.mean(Re_Loss, 0)

        #print(f"Policy_Loss: {loss_mean[0]:.6f}, Value_loss: {loss_mean[1]:.6f}, Loss: {loss_mean[2]:.6f}")

        return loss_mean







