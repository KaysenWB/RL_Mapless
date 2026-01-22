import os.path

import numpy as np
import matplotlib.pyplot as plt
import torch
from utils import Standard, Calculating, Get_circle

class Draw_result:
    def __init__(self, args):
        self.args = args
        self.save_dir = args.save_dir
        self.env_name = args.env_name
        self.back_extent = args.le
        self.back_pic = args.back_pic
        self.size = args.grids_size
        self.cache_root = args.cache_root
        self.scaler = Standard()

    def draw_trajs(self, paths):
        if self.env_name == "GridEnv":
            self.draw_trajs_grid(paths)
        elif self.env_name == "StreetEnv":
            #self.draw_trajs_street(paths)
            self.draw_trajs_street_multi(paths)
        elif self.env_name == "StreetPicEnv":
            #self.draw_trajs_street(paths)
            self.draw_trajs_street_multi(paths)
        return

    def draw_trajs_grid(self, paths):
        fig, axs = plt.subplots(3, 3, figsize=(15, 15))
        for e, axe in enumerate(axs.flatten()):
            path = paths[e]
            axe.scatter(path[0, 0] + 0.5, path[0, 1] + 0.5, c='blue', label='Start')
            axe.scatter(path[0, -2] + 0.5, path[0, -1] + 0.5, c='red', label='Goal')
            axe.plot(path[:, 0] + 0.5, path[:, 1] + 0.5, '-o', markersize=2, label=f'Test_{e}')
            axe.set_xlim(0, self.size)
            axe.set_ylim(0, self.size)
            axe.legend()
        fig.tight_layout()
        fig.savefig(fname=self.save_dir + f'/test.jpg', dpi=100, format='jpg', bbox_inches='tight')
        plt.close()
        return

    def draw_back_pic(self, map_coor):
        pic_path = self.save_dir + '/back_pic.jpg'
        if not os.path.exists(pic_path):
            plt.figure(figsize=(5, 5))
            plt.scatter(map_coor[:, 0], map_coor[:, 1], c='grey', s=0.3, alpha=0.6)
            plt.axis('off')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.savefig(pic_path, dpi=300, pad_inches=0, bbox_inches='tight', transparent=True)
            plt.close()

    def draw_trajs_street_multi(self, paths):
        infos = np.load(self.cache_root + '/static_info.npy', allow_pickle=True)
        if not self.args.le == None:
            area_index = np.load(self.cache_root + '/area_index.npy')
            infos = infos[area_index]
        map_coor = infos[1:, 1:3].astype('float32')
        self.draw_back_pic(map_coor)

        self.scaler.fit(map_coor)
        paths = [self.scaler.inver_trans(p) for p in paths]
        figsize_ = (14, 13) if self.args.le_flag == "min" else (16, 10)
        fig, axs = plt.subplots(3, 3, figsize= figsize_) # 18, 12, min: 14, 13
        for e, axe in enumerate(axs.flatten()):
            path = paths[e]
            axe.plot(path[:, 0], path[:, 1], '-o', color="green", markersize=2, label=f'Test_{e}')
            axe.scatter(path[0, 0], path[0, 1], c='blue', label='Start', s=20)
            axe.scatter(path[0, 2], path[0, 3], c='red', label='Goal', s=20)
            if self.args.le_flag != "min":
                cir = Get_circle(path[0, 2:], radius=50)
                axe.plot(cir[:, 0], cir[:, 1], color="red",lw=1)

            imp = plt.imread(self.save_dir + f'/back_pic.jpg')
            axe.imshow(imp, extent=self.back_extent)
            axe.legend()
        fig.tight_layout()
        fig.savefig(fname=self.save_dir + f'/test.jpg', dpi=300, format='jpg', bbox_inches='tight')
        plt.close()
        return

    def draw_trajs_street(self, paths):
        infos = np.load('./cache/static_info.npy', allow_pickle=True)
        map_coor = infos[1:, 1:3].astype('float32')
        self.scaler.fit(map_coor)
        paths = [self.scaler.inver_trans(p) for p in paths]
        for e in range(3):
            path = paths[e]
            plt.figure(figsize=(8, 8))
            plt.scatter(map_coor[:, 0], map_coor[:, 1], c='grey', s=0.1, alpha=0.1)
            plt.plot(path[:, 0], path[:, 1], '-o', color ="green", markersize=2, label=f'Test_{e}')
            plt.scatter(path[0, 0], path[0, 1], c='blue', label='Start')
            plt.scatter(path[0, 2], path[0, 3], c='red', label='Goal')
            imp = plt.imread(self.back_pic)
            plt.imshow(imp, extent=self.back_extent)
            plt.legend()
            plt.savefig(fname=self.save_dir + f'/test_{e}.jpg', dpi=100, format='jpg', bbox_inches='tight')
            plt.close()
        return


    def draw_rewads(self):
        episode_rewards = np.load(self.save_dir + "/ep_rewards.npy")
        plt.figure(figsize=(12, 8))
        plt.plot(episode_rewards, alpha=0.5, label="Episode Return")
        if len(episode_rewards) >= 50:
            smooth = np.convolve(episode_rewards, np.ones(50) / 50, mode='valid')
            plt.plot(range(49, 49 + len(smooth)), smooth, c="#1f77b4", label="Moving Avg(50)")
        #plt.axhline(y=3, color='grey', alpha=0.7, linestyle='--', )
        plt.legend()
        plt.savefig(fname=self.save_dir + f'/rewards.jpg', dpi=300, format='jpg', bbox_inches='tight')
        plt.close()
        return


    def extent_mercator(self, coor):
        coor = torch.tensor(coor)
        coor = self.scaler.mercator(coor)
        ex = [coor[:, 0].min(),coor[:, 0].max(), coor[:, 1].min(),coor[:, 1].max()]
        ex = [e.item() for e in ex]
        return ex