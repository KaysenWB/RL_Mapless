import torch

import streetlearn_pb2
import cv2
from torch.utils.data import Dataset
import numpy as np
import plyvel

import torchvision as tov
import pytorch360convert


#from torch360.transforms import equirectangular_to_perspective
from collections import OrderedDict
import torchvision.transforms as tt
from torchvision import models
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


"""
Possible improvements:
Memory: Use ResNet18 as a data preprocessing step in the data loader 
        to reduce the amount of data cached by the model.
Speed: Batch projection to perspective view and batch normalisation with resize.

And: Offline image processing
"""

class StandardScaler:
    def __init__(self, pixel = 84):
        self.mean = 0.
        self.std = 1.
        self.trans_pic = tt.Compose([
            tt.Resize((pixel, pixel)),
            tt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.trans_pic2 = tt.Compose([
            tt.Resize((84, 84)),
            tt.ToTensor(),
            tt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def fit(self, data):
        self.mean = data.mean(0)#.detach().numpy()
        self.std = data.std(0)#.detach().numpy()

    def transform(self, data):
        return (data - self.mean) / self.std
    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class Picdataset_Realtime:
    def __init__(self, args):
        self.db_root = args.db_root_h
        self.cap = args.max_buf
        self.cache_root = args.cache_root
        self.device = args.device
        self.pixel = args.pixel
        self.infos = np.load(self.cache_root + '/static_info.npy', allow_pickle=True)
        self.neig = np.load(self.cache_root + '/neig_info.npy', allow_pickle=True)

        self.buf = OrderedDict()
        self.pano = streetlearn_pb2.Pano()
        self.scaler = StandardScaler(pixel=self.pixel)

    def scaled_view(self, frame, cur_head):

        #view = e2p(frame, fov_deg=120, u_deg=cur_head, v_deg=0, out_hw=(800, 800))
        view = pytorch360convert.e2p(frame, fov_deg=120, h_deg=cur_head, v_deg=0, out_hw=(self.pixel, self.pixel))
        view = self.scaler.trans_pic(view)#.to(self.device)

        return view

    def load_pano(self, pano_id):

        db = plyvel.DB(self.db_root, create_if_missing=False)
        for ke in pano_id:
            self.pano.ParseFromString(db.get(ke.encode('utf-8')))
            frame = tov.io.decode_image(torch.frombuffer(self.pano.compressed_image,dtype=torch.uint8),
                                        mode=tov.io.ImageReadMode.RGB)
            self.buf[ke] = (frame / 255).to(self.device)
            if len(self.buf) >= self.cap:
                self.buf.popitem(last=False)
        db.close()
        return

    def update_buff(self, pano_id):
        new_id = []
        for p_id in pano_id:
            if p_id in self.buf:
                self.buf.move_to_end(p_id)
            else:
                new_id.append(p_id)
        if new_id:
            self.load_pano(new_id)
        return

    def get_(self, pano_id, cur_heading):

        self.update_buff(pano_id)
        frames = [self.buf[i] for i in pano_id]
        #views_ = self.scaled_view(frames, cur_heading)
        views_ = [self.scaled_view(f, cur_heading[i]) for i, f in enumerate(frames)]
        view = torch.stack(views_, dim=0)
        return view


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






class Picdataset_Cache:
    def __init__(self, args):
        self.db_root = args.db_root_h
        self.cache_root = args.cache_root
        self.device = args.device
        self.le = args.le
        self.pixel = args.pixel

        self.infos = np.load(self.cache_root + '/static_info.npy', allow_pickle=True)
        if not self.le == None:
            area_index = np.load(self.cache_root + '/area_index.npy')
            self.infos = self.infos[area_index]

        self.cnn = models.resnet18(weights='IMAGENET1K_V1')
        for param in self.cnn.parameters():
            param.requires_grad = False
        self.pano = streetlearn_pb2.Pano()
        self.scaler = StandardScaler()
        self.pics_buff = 0
        self.initial_buff()


    def initial_buff(self):
        pano_id = self.infos[:, 0]
        pics = []
        db = plyvel.DB(self.db_root, create_if_missing=False)
        for ke in pano_id:
            self.pano.ParseFromString(db.get(ke.encode('utf-8')))
            frame = tov.io.decode_image(torch.frombuffer(self.pano.compressed_image, dtype=torch.uint8), mode=tov.io.ImageReadMode.RGB)
            frame = frame.float() / 225
            view = self.scaler.trans_pic(frame)
            pics.append(view)
        db.close()
        self.pics_buff = torch.stack(pics).to(self.device)
        self.pics_buff = self.cnn(self.pics_buff)
        #torch.save(pp, self.cache_root + "/pics.th")
        return


    def get_(self, pano_id, cur_head):
        frame = self.pics_buff[pano_id]#.squeeze()
        #view = pytorch360convert.e2p(frame, fov_deg=120, h_deg=cur_head, v_deg=0, out_hw=(self.pixel, self.pixel))
        #view = self.scaler.trans_pic(view).unsqueeze(0)  # .to(self.device)
        #view_enc = self.cnn(view)
        return frame














if __name__ == '__main__':
    db_root_h = '/Users/yangkaisen/MyProject/Navigation/manhattan_2021_highres'
    db_root_l = '/Users/yangkaisen/MyProject/Navigation/manhattan_2021_lowres'
    cache_root = './cache'
    cur_head = 110
    daset = Picdataset_Cache(db_root=db_root_h, cache_root=cache_root, max_buf=100)
    #daset.initial_buff(start_p=35000)
    route = daset.route[:, 0]
    get_data = []
    for id in route:
        frame, posi, c_head, neigs = daset[id]
        shift_head = cur_head - c_head
        view = daset.scaled_view(frame, shift_head)
        get_data.append(view)
    print('l')