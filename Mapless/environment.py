import numpy as np
import torch
from utils import *




"""
    The environment is still in the training data preparation phase, 
    querying the corresponding training data through actions. 
    This includes: state (input), reward and done (prior weight for loss).
"""
# states
# 1 goal, posi
# 1 goal_emb, posi_emb
# 3 view_image, goal_emb, posi_emb
# 4 + distance, bearing, history, neigs
class CourierEnv:
    def __init__(self, args):

        self.le = args.le
        self.cache_root = args.cache_root
        self.batch_size = args.batch_size
        self.num_actions = args.num_actions
        self.goal_range = args.goal_range
        self.punish =  args.punish
        self.max_len = args.max_len
        self.device = args.device
        self.landmarks = Landmark(args)
        self.cal = Calculating()
        self.start_goal = 0
        self.cur = 0
        self.goal = 0
        self.coor = 0  # different from state, easier to get cur and goal coor.
        self.heading = 0
        self.distance = 0
        self.env_state = 0
        self._initial()

    # env_state, done, reward = self.env.setp(action)

    def _initial (self):

        self.infos = np.load(self.cache_root + '/static_info.npy', allow_pickle=True)
        self.neig = np.load(self.cache_root + '/neig_info.npy', allow_pickle=True)
        self.edge = np.load(self.cache_root + '/neigs_index.npy')

        if not self.le == None:
            area_index = np.load(self.cache_root + '/area_index.npy')
            self.infos = self.infos[area_index]
            self.neig = self.neig[area_index]

        self.length = len(self.neig)
        # self.nodes(tensor) --> coor query; self.neig(numpy) --> id query
        self.nodes = self.infos[:, [1, 2, 8]].astype('float32')
        self.nodes = torch.tensor(self.nodes).to(self.device)
        self.edge = torch.tensor(self.edge).to(self.device)


    def reset(self):

        self.start_goal = torch.randint(0, self.length, (self.batch_size, 2), device=self.device)
        self.cur = self.start_goal[:, 0]
        self.goal = self.start_goal[:, 1]

        cur_coo = self.nodes[self.cur][:, :-1]
        goal_coo = self.nodes[self.goal][:, :-1]

        self.coor = torch.cat((cur_coo, goal_coo), dim=1)
        self.heading = self.cal.heading(self.coor).unsqueeze(1)
        self.distance = self.cal.distance(self.coor).unsqueeze(1)

        self.env_state = self.coor.clone()
        done = torch.zeros((self.batch_size, 1), device=self.device)
        reward = torch.zeros((self.batch_size, 1), device=self.device)

        return self.env_state, done, reward

    def re_neigs_clear(self, neigs, cur):
        # neigs, cur: Tensor[Batch, max_neigs], Tensor[Batch, 1]
        neigs_index_ = neigs - cur
        neigs_index_[neigs_index_ == 0] = np.nan
        return neigs_index_ + cur


    def step(self, actions):
        # actions [batch, 1]
        # find next neigs as cur

        # taking actions
        if self.num_actions == 4:
            shift = actions * 90 # [0, 1, 2, 3]: [forward, right, back, left]
        else:
            shift = (actions - 1) * 90 # [0, 1, 2]: [left, forward, right]
        take_heads = torch.fmod(self.heading + shift, 360)

        # find all neigs
        cur_index = self.cur
        neigs_index = self.edge[cur_index]
        neigs_coor = self.nodes[neigs_index][:, :, :2]

        # cal all neigs heading
        max_neigs = neigs_index.shape[1]
        cur_coor = self.coor[:, :2].unsqueeze(1).repeat(1, max_neigs, 1)
        coor = torch.cat((cur_coor, neigs_coor), dim=-1).reshape(-1, 4)
        neigs_heading = self.cal.heading(coor).reshape(self.batch_size, max_neigs)

        # find the neig can move to
        diff = neigs_heading.clone()
        diff[diff == 0] = 360 * 3
        diff = (diff - take_heads).abs().round(decimals=2)
        _diff = diff - diff.amin(dim=1).unsqueeze(1)
        mask = (diff <= 45) & (_diff == 0) # optional heading & min heading diff

        # if not, stay here
        mask_last = ~torch.any(mask, dim=1).unsqueeze(1)

        # moving next
        mask = torch.cat((mask, mask_last), dim=1)
        _neigs_index = torch.cat((neigs_index, cur_index.unsqueeze(1)),dim=1)
        next_index = _neigs_index[mask]

        # update data
        next_coor = self.nodes[next_index][:, :2]
        cur_next_coor = torch.cat((self.coor[:, :2], next_coor), dim=1)

        self.cur = next_index
        self.coor[:, :2] = next_coor
        self.heading = self.cal.heading(cur_next_coor).unsqueeze(1)
        distance = self.cal.distance(self.coor).unsqueeze(1)

        done = distance < self.goal_range
        reward = self.distance - distance
        reward[reward > 0] = 3
        reward[reward < 0] = 0
        reward[reward == 0] = -1

        if done.any():
            reward[done] = self.max_len * 10
            new_id = torch.where(done)[0]
            new_goal = torch.randint(0, self.length, (len(new_id),))
            new_goal_coor = self.nodes[new_goal][:, :-1]
            self.coor[new_id, 2:] = new_goal_coor

        self.distance = distance
        self.env_state = self.coor.clone()

        return self.env_state, done, reward



if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Mapless_Navigation')
    parser.add_argument('--cache_root', default='./cache', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_actions', default=4, type=int)
    parser.add_argument('--goal_range', default=100, type=int)
    parser.add_argument('--punish', default=-3, type=int)
    parser.add_argument('--max_len', default=200, type=int)


    parser.add_argument('--le', default=[-73.995, -73.97, 40.747, 40.765], type=list)
    args = parser.parse_args()
    env = CourierEnv(args)
    env.reset()

    actions = torch.randint(0, 4, (args.batch_size,1))
    env.step(actions)

    print(';')