import numpy as np
import torch
from utils import Calculating, Standard
from cache import Landmark
from dataloader import Picdataset_Realtime, Picdataset_Cache
import matplotlib.pyplot as plt



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

class StreetEnv:
    def __init__(self, args):
        self.args = args
        self.le = args.le
        self.le_flag = args.le_flag
        self.cache_root = args.cache_root
        self.batch_size = args.batch_size
        self.num_actions = args.num_actions
        self.goal_range = args.goal_range
        self.device = args.device
        self.max_steps = args.max_steps
        self.views = {}
        self.back_pic = args.back_pic
        self.land = Landmark(args)
        self.cal = Calculating()
        self.scaler = Standard()
        self._initial()

    def _initial (self):

        self.infos = np.load(self.cache_root + '/static_info.npy', allow_pickle=True)
        self.neig = np.load(self.cache_root + '/neig_info.npy', allow_pickle=True)
        self.edge = np.load(self.cache_root + '/neigs_index.npy')

        if not self.le == None:
            area_index = np.load(self.cache_root + '/area_index.npy')
            self.infos = self.infos[area_index]
            self.neig = self.neig[area_index]

        self.le = torch.tensor(self.le, device=self.device)
        self.re_le = self.le + (self.le[[1, 0, 3, 2]] - self.le) * 0.2
        self.length = len(self.neig)
        # self.nodes(tensor) --> coor query; self.neig(numpy) --> id query
        self.nodes = self.infos[:, [1, 2, 8]].astype('float32')
        self.nodes = torch.tensor(self.nodes).to(self.device)
        self.scaler.fit(self.nodes[:,:2])
        self.edge = torch.tensor(self.edge).to(self.device)
        self.shift = torch.tensor([22.5 * s for s in range(16)], device=self.device, dtype=torch.float32)


    def waypoints_reset(self):
        num_points = int(self.distance / 100)
        cur_coor = self.nodes[self.cur][0, :-1]
        goal_coor = self.nodes[self.goal][0, :-1]
        lng_p = torch.linspace(cur_coor[0], goal_coor[0], num_points+2, device= self.device)
        lat_p = torch.linspace(cur_coor[1], goal_coor[1], num_points+2, device= self.device)
        self.re_points = torch.stack([lng_p[1:-1], lat_p[1:-1]], dim=1)

    def waypoints_reward(self, reward):
        coor_reps = self.coor[:, :2].repeat(len(self.re_points), 1)
        coor_reps = torch.cat((coor_reps, self.re_points), dim=-1)
        dis_reps = self.cal.distance(coor_reps)
        if torch.any(dis_reps < self.goal_range):
            self.re_points = self.re_points[torch.where(dis_reps > self.goal_range)[0]]
            reward += 1
        return reward

    def _get_obs(self):
        state = self.scaler.trans(self.coor.clone().squeeze())
        #diff = self.coor - self.bl
        #state = torch.round(diff /self.emb).squeeze()
        return state

    def _dist_manh(self, obs):
        return torch.abs(obs[:2] - obs[2:]).sum().unsqueeze(0)

    def reset_conditios(self):
        if self.le_flag =="min":
            #label = self.res[:, 0] != self.res[:, 1]
            lon_, lat_ = self.coor[0, [0, 2]], self.coor[0, [1, 3]]
            label = (lon_ > self.re_le[0]) & (lon_ < self.re_le[1]) & (lat_ > self.re_le[2]) & (lat_ < self.re_le[3])
            label = label.all()
            label = (self.distance >= 40) & label  # & (self.distance <= 500)
        if self.le_flag == "mid":
            lon_, lat_ = self.coor[0, [0, 2]], self.coor[0, [1, 3]]
            label = (lon_ > self.re_le[0]) & (lon_ < self.re_le[1]) & (lat_ > self.re_le[2]) & (lat_ < self.re_le[3])
            label = label.all()
            label = (self.distance >= 400) & label #& (self.distance <= 500)
        return label



    def reset(self):
        while True:
            self.res = torch.randint(0, self.length, (self.batch_size, 2), device=self.device)
            self.cur = self.res[:, 0]
            self.goal = self.res[:, 1]
            self.coor = torch.cat((self.nodes[self.cur][:, :-1], self.nodes[self.goal][:, :-1]), dim=1)
            self.distance = self.cal.distance(self.coor)
            #self.distance = self._dist_manh(self._get_obs())
            if self.reset_conditios():
                break
        self.steps = 0
        # self.waypoints_reset()
        return self._get_obs()  # tenser (states,)


    def step(self, actions, iftest = False):
        self.steps += 1

        # find all neigs and heading
        neigs_index = self.edge[self.cur].squeeze()
        neigs_coor = self.nodes[neigs_index][:, :2]
        coor = self.coor[:, :2].repeat(len(neigs_index), 1)
        neigs_heading = self.cal.heading(torch.cat((coor, neigs_coor), dim=-1))

        # take action and move on
        diff = neigs_heading[neigs_heading != 0]
        diff_ = 180 - torch.abs(torch.abs(diff - self.shift[actions]) - 180)
        next_index = neigs_index[diff_.argmin()].unsqueeze(0)

        # update data
        self.cur = next_index
        self.coor[:, :2] = self.nodes[next_index][:, :2]

        # done and reward
        distance = self.cal.distance(self.coor)
        done = (distance <= self.goal_range) | (self.steps >= self.max_steps)

        reward = (self.distance - distance) * 0.02
        reward += -0.02
        if done.any():
            reward += 1
        # reward = self.waypoints_reward(reward)
        self.distance = distance


        if not iftest:
            return self._get_obs(), reward.squeeze(), done[0], self.views  # tenser (states,), tenser(), bool, {}
        else:
            return self._get_obs(), self.scaler.trans(self.coor.clone().squeeze()), done


    def draw(self, cur_coo, goal_coo):
        map_coor = self.infos[1:, 1:3].astype('float32')
        plt.figure(figsize=(8, 8))
        plt.scatter(map_coor[:, 0], map_coor[:, 1], c='grey', s=0.1, alpha=0.5)
        plt.scatter(cur_coo[0], cur_coo[1], c='blue', label='Start')
        plt.scatter(goal_coo[0], goal_coo[1], c='red', label='Goal')
        plt.scatter(self.re_points[:, 0], self.re_points[:, 1], s=0.5, c='green', label='reward_points')
        imp = plt.imread(self.back_pic)
        plt.imshow(imp, extent=self.le)
        plt.legend()
        #plt.savefig(fname=self.save_dir + f'/test_{e}.jpg', dpi=100, format='jpg', bbox_inches='tight')
        plt.show()










class StreetPicEnv:
    def __init__(self, args):
        self.args = args
        self.le = args.le
        self.le_flag = args.le_flag
        self.visual = args.visual
        self.cache_root = args.cache_root
        self.batch_size = args.batch_size
        self.num_actions = args.num_actions
        self.goal_range = args.goal_range
        self.device = args.device
        self.max_steps = args.max_steps
        self.load_pics = args.load_pics
        self.back_pic = args.back_pic
        self.land = Landmark(args)
        self.cal = Calculating()
        self.scaler = Standard()
        self.views = 0
        self.last = 0
        self._initial()

    def _initial(self):

        self.infos = np.load(self.cache_root + '/static_info.npy', allow_pickle=True)
        self.neig = np.load(self.cache_root + '/neig_info.npy', allow_pickle=True)
        self.edge = np.load(self.cache_root + '/neigs_index.npy')

        if not self.le == None:
            area_index = np.load(self.cache_root + '/area_index.npy')
            self.infos = self.infos[area_index]
            self.neig = self.neig[area_index]

        self.le = torch.tensor(self.le, device=self.device)
        self.re_le = self.le + (self.le[[1, 0, 3, 2]] - self.le) * 0.2
        self.length = len(self.neig)

        # self.nodes(tensor) --> coor query; self.neig(numpy) --> id query
        self.nodes = self.infos[:, [1, 2, 8]].astype('float32')
        self.nodes = torch.tensor(self.nodes).to(self.device)
        self.scaler.fit(self.nodes[:,:2])
        self.edge = torch.tensor(self.edge).to(self.device)
        #self.shift = torch.tensor([22.5 * s for s in range(16)], device=self.device, dtype=torch.float32)
        self.shift = torch.tensor([90 * s for s in range(4)], device=self.device, dtype=torch.float32)

        if self.visual:
            if self.load_pics == "rt":
                self.panoset = Picdataset_Realtime(self.args)
            else:
                self.panoset = Picdataset_Cache(self.args)

    def labels_mark__(self, coor):
        bins = 16
        den = torch.abs(self.le[[1, 0, 3, 2]] - self.le).float()

        epsilon = 1e-6
        labels_ = ((coor - self.le[[0, 2, 0, 2]]) * bins / (den + epsilon)).floor()
        labels_ = torch.clamp(labels_, 0, bins - 1)

        labels = labels_ * bins + labels_[[1, 1, 3, 3]]
        return labels[[0, 2]]

    def labels_mark_(self, coor):
        bins = 16
        # 计算每个维度的范围
        den = torch.abs(self.le[[1, 0, 3, 2]] - self.le).float()

        epsilon = 1e-6
        # 将坐标离散化到 [0, bins-1] 区间
        labels_ = ((coor - self.le[[0, 2, 0, 2]]) * bins / (den + epsilon)).floor()
        labels_ = torch.clamp(labels_, 0, bins - 1)

        # 修正索引计算 - 将二维索引转换为一维索引
        # 假设 labels_ 包含 [x1, y1, x2, y2] 格式的坐标
        labels = labels_[[0]] * bins + labels_[[1]]  # 第一个坐标的一维索引
        labels = torch.cat([labels, labels_[[2]] * bins + labels_[[3]]])  # 第二个坐标的一维索引

        return labels

    def labels_mark(self, coor):
        obs = coor[:2]
        goal = coor[2:]

        bins = 16
        den_lon = 0.005
        den_lat = 0.005
        lower = self.le[[0, 2]].clone()
        label = []
        for nodes in [obs, goal]:
            diff = nodes - lower
            lon = ((diff[0] / den_lon) * bins).floor()
            lat = ((diff[1] / den_lat) * bins).floor()
            tab = lat * bins + lon
            label.append(tab)
        label = torch.stack((label))
        return label


    def grids(self, coor):
        bins = 16
        epsilon = 1e-6
        den = torch.abs(self.le[[1, 0, 3, 2]] - self.le).float()
        labels = ((coor - self.le[[0, 2, 0, 2]]) * bins /  (den + epsilon))
        labels = torch.clamp(labels, 0, bins) / bins
        return labels


    def _get_obs(self):
        state = self.scaler.trans(self.coor.clone().squeeze())
        view = self.visual_learning() if self.visual else 0

        emb_1 = self.land.embbeding(self.coor[:, :2]).squeeze()
        emb_2 = self.land.embbeding(self.coor[:, 2:]).squeeze()

        grids = self.grids(self.coor.clone().squeeze())
        labels = self.labels_mark(self.coor.clone().squeeze())
        sp = self.heading / 360
        sp_bin = ((self.heading / 360) * 16).long()
        state = torch.cat((labels, grids, state, sp, sp_bin, emb_1, emb_2), -1)
        return (state, view, self.cur)

    def visual_learning(self):
        #pano_id = [self.infos[self.cur, 0]]
        cur_head = self.heading - self.nodes[self.cur, -1]
        views = self.panoset.get_(self.cur, cur_head)
        return views

    def get_views(self, ids):
        return self.panoset.get_(ids, 0)

    def reset_conditios(self):
        if self.le_flag =="min":
            #label = self.res[:, 0] != self.res[:, 1]
            lon_, lat_ = self.coor[0, [0, 2]], self.coor[0, [1, 3]]
            label = (lon_ > self.re_le[0]) & (lon_ < self.re_le[1]) & (lat_ > self.re_le[2]) & (lat_ < self.re_le[3])
            label = label.all()
            label = (self.distance >= 40) & label  # & (self.distance <= 500)
        if self.le_flag == "mid":
            lon_, lat_ = self.coor[0, [0, 2]], self.coor[0, [1, 3]]
            label = (lon_ > self.re_le[0]) & (lon_ < self.re_le[1]) & (lat_ > self.re_le[2]) & (lat_ < self.re_le[3])
            label = label.all()
            label = (self.distance >= 400) & label #& (self.distance <= 500)
        return label


    def reset(self):
        while True:
            self.res = torch.randint(0, self.length, (self.batch_size, 2), device=self.device)
            self.cur = self.res[:, 0]
            self.goal = self.res[:, 1]
            self.coor = torch.cat((self.nodes[self.cur][:, :-1], self.nodes[self.goal][:, :-1]), dim=1)
            self.distance = self.cal.distance(self.coor)
            self.heading = self.cal.heading(self.coor)
            #self.distance = self._dist_manh(self._get_obs())
            if self.reset_conditios():
                break
        self.steps = 0
        return self._get_obs()  # tenser (states,)


    def step(self, actions, iftest = False):
        self.steps += 1

        # find all neigs and heading
        neigs_index = self.edge[self.cur].squeeze()
        neigs_coor = self.nodes[neigs_index][:, :2]
        coor = self.coor[:, :2].repeat(len(neigs_index), 1)
        neigs_heading = self.cal.heading(torch.cat((coor, neigs_coor), dim=-1))  # 30, 120, 210, 300

        # take action and move on
        diff = neigs_heading[neigs_heading != 0]
        diff_ = 180 - torch.abs(torch.abs(diff - self.shift[actions]) - 180)
        try:
            next_index = neigs_index[diff_.argmin()].unsqueeze(0)
        except:
            next_index = neigs_index[0].unsqueeze(0)


        # update data
        self.last = self.cur
        self.cur = next_index
        self.coor[:, :2] = self.nodes[next_index][:, :2]

        # done and reward
        distance = self.cal.distance(self.coor)
        done = (distance <= self.goal_range) | (self.steps >= self.max_steps)

        reward = (self.distance - distance) * 0.02
        reward += -0.02
        if done.any():
            reward += 1
        # reward = self.waypoints_reward(reward)
        self.distance = distance
        self.heading = self.cal.heading(self.coor)


        if not iftest:
            return self._get_obs(), reward.squeeze(), done[0], self.views  # tenser (states,), tenser(), bool, {}
        else:
            return self._get_obs(), self.scaler.trans(self.coor.clone().squeeze()), done





class GridEnv:
    def __init__(self, args):
        self.size = args.grids_size
        self.max_steps = args.max_steps
        self.device = args.device
        self.generator = torch.Generator(device=self.device)
        self.move_vectors = torch.tensor([[0, -1], [0, 1], [-1, 0], [1, 0]], device=self.device, dtype=torch.long)
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
        #s = self.size - 1
        a_norm = self.agent.float() #/ s
        g_norm = self.goal.float() #/ s
        """d_norm = (g_norm - a_norm)
        manh_norm = torch.abs(d_norm).sum() / 2  # [0,1]
        obs = torch.cat([a_norm, g_norm, d_norm, manh_norm.unsqueeze(0)])"""
        obs = torch.cat([a_norm, g_norm])
        return obs

    def step(self, action, iftest = None):
        new_agent = self.agent + self.move_vectors[action]
        new_agent = torch.clamp(new_agent, 0, self.size - 1)

        self.agent = new_agent
        self.steps += 1

        dist = self._dist(self.agent, self.goal)
        shaping = 0.2 * (self.prev_dist - dist)
        self.prev_dist = dist

        reward = -0.01 + shaping
        done = torch.equal(self.agent, self.goal) or (self.steps >= self.max_steps)

        if torch.equal(self.agent, self.goal):
            reward = torch.tensor(1.0, device=self.device)
        if not iftest:
            return self._get_obs(), reward, done, {}        # tenser (state_dim), reward = tenser(), done = bool
        else:
            return self._get_obs(), torch.cat((self.agent, self.goal)), done





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
    env = StreetEnv(args)
    env.reset()

    actions = torch.randint(0, 4, (args.batch_size,1))
    env.step(actions)

    print(';')