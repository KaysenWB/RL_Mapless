
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from torchvision import transforms
import numpy as np
import plyvel
import streetlearn_pb2
import cv2
#import py360convert
from collections import OrderedDict
import matplotlib.pyplot as plt


class StandardScaler:
    def __init__(self):
        self.mean = 0.
        self.std = 1.
        self.trans_pic = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def fit(self, data):
        self.mean = data.mean(0)#.detach().numpy()
        self.std = data.std(0)#.detach().numpy()

    def transform(self, data):
        return (data - self.mean) / self.std
    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class StreetLearnDataset(Dataset):
    def __init__(self, db_root, cache_root, max_buf = 100):
        super().__init__()
        self.db_root = db_root
        self.cache_root = cache_root
        self.infos = np.load(cache_root + '/static_info.npy', allow_pickle=True)
        self.neig = np.load(cache_root + '/neig_info.npy', allow_pickle=True)

        self.buf = OrderedDict()
        self.cap = max_buf
        self.pano = streetlearn_pb2.Pano()
        self.scaler = StandardScaler()

    def scaled_view(self, frame, cur_head):

        view = py360convert.e2p(frame, fov_deg=120, u_deg=cur_head, v_deg=0, out_hw=(800, 800))
        view = self.scaler.trans_pic(view)

        return view


    def load_pano(self, pano_id):

        db = plyvel.DB(self.db_root, create_if_missing=False)
        values = db.get(pano_id.encode('utf-8'))
        self.pano.ParseFromString(values)
        frame = cv2.imdecode(np.frombuffer(self.pano.compressed_image, dtype=np.uint8), cv2.IMREAD_COLOR)
        db.close()

        self.buf[pano_id] = frame
        if len(self.buf) >= self.cap:
            self.buf.popitem(last=False)
        return



    def __getitem__(self, pano_id):
        if pano_id in self.buf:
            self.buf.move_to_end(pano_id)
        else:
            self.load_pano(pano_id)

        frame = self.buf[pano_id]
        info = self.infos[self.infos[:, 0] == pano_id]
        neigs = self.neig[self.neig[:, 0] == pano_id]
        neigs = neigs[neigs != None]

        c_head = info[:, -3][0]
        posi = info[:, 1:3]

        return frame, posi, c_head,  neigs

    def __len__(self):
        return len(self.buf)



    def initial_buff(self, start_p):
        ## unwrited how to get neig around start_point
        coor = self.infos[1::2, :3]
        coor_s = np.repeat(self.infos[start_p:start_p + 1, 1:3], len(coor), axis=0)
        dis = np.sum((coor[:, 1:] - coor_s) ** 2, axis=1)
        indices = np.argpartition(dis, self.cap)[:self.cap]

        neigs = coor[indices, 1:]
        db = plyvel.DB(self.db_root, create_if_missing=False)
        for ne in neigs[:, 0]:
            values = db.get(ne.encode('utf-8'))
            self.pano.ParseFromString(values)
            frame = cv2.imdecode(np.frombuffer(self.pano.compressed_image, dtype=np.uint8), cv2.IMREAD_COLOR)
            self.buf[ne] = frame
        db.close()
        """
            plt.figure(figsize=(8, 8))
            coor_w = self.infos[1:, 1:3].astype('float32')
            plt.scatter(coor_w[:, 0], coor_w[:, 1], c='grey', s=0.1, alpha=0.1)
            plt.scatter(neigs[:, 1], neigs[:, 2], c='red', s=0.1, alpha=0.5)

            back_pic = '/Users/yangkaisen/MyProject/Data/map/white.jpg'
            back_extent = [-74.035, -73.935, 40.695, 40.795]
            imp = plt.imread(back_pic)
            plt.imshow(imp, extent=back_extent)
            plt.savefig(cache_root + "/cache_initial.jpg", dpi=500, bbox_inches='tight')
            plt.show()
        """
        return


        """return {
            'views': views,
            'gps': gps,
            'node_id': node_id,
            'neighbors': neighbors
        }"""





if __name__ == '__main__':
    db_root_h = '/Users/yangkaisen/MyProject/Navigation/manhattan_2021_highres'
    db_root_l = '/Users/yangkaisen/MyProject/Navigation/manhattan_2021_lowres'
    cache_root = './cache'
    cur_head = 110
    daset = StreetLearnDataset(db_root=db_root_h, cache_root=cache_root, max_buf=100)
    #daset.initial_buff(start_p=35000)
    route = daset.route[:, 0]
    get_data = []
    for id in route:
        frame, posi, c_head, neigs = daset[id]
        shift_head = cur_head - c_head
        view = daset.scaled_view(frame, shift_head)
        get_data.append(view)
    print('l')



