import numpy as np
import plyvel
from tqdm import tqdm
import cv2
#from Navigation.Streetlearn.streetlearn.proto import streetlearn_pb2
import streetlearn_pb2
import matplotlib.pyplot as plt
import py360convert
import os
from utils import *





class Cache_DB:
    def __init__(self, args):
        self.db_root = args.db_root_h
        self.save_root = args.cache_root
        self.le = args.le

    def Get_static_info(self, pics = 55761):
        db = plyvel.DB(self.db_root, create_if_missing=False)
        infos_title = np.array(['id', 'coords.lng', 'coords.lat', 'snapped_coords.lng', 'snapped_coords.lat',
                                'alt', 'roll_deg', 'pitch_deg', 'heading_deg', 'pano_date', 'country_code'])

        neig_title = np.array(['self', 'neig1', 'neig2', 'neig3', 'neig4', 'neig5', 'neig6'])
        static_info = np.empty((pics + 1, 11), dtype=object)
        neig_info = np.empty((pics + 1, 7), dtype=object)
        static_info[0, :] = infos_title
        neig_info[0, :] = neig_title
        pano = streetlearn_pb2.Pano()

        for pic_id, (key, value) in enumerate(db):
            pano.ParseFromString(value)
            infos = [pano.id, pano.coords.lng, pano.coords.lat, pano.snapped_coords.lng, pano.snapped_coords.lat,
                     pano.alt, pano.roll_deg, pano.pitch_deg, pano.heading_deg, pano.pano_date, pano.country_code ]
            neig = [pano.id] + [ne.id for ne in pano.neighbor]
            static_info[pic_id + 1, :] = infos
            for nd, n in enumerate(neig):
                neig_info[pic_id + 1, nd] = n

        db.close()
        np.save(self.save_root + '/static_info.npy', np.delete(static_info, 46698, axis=0))
        np.save(self.save_root + '/neig_info.npy', np.delete(neig_info, 46698, axis=0))

        return

    def Get_neigs_index(self):
        neigs_id = np.load(self.save_root + '/neig_info.npy', allow_pickle=True)[:, 1:]
        infos = np.load(self.save_root + '/static_info.npy', allow_pickle=True)

        if not self.le == None:
            area_index = np.load(self.save_root + '/area_index.npy')
            neigs_id = neigs_id[area_index]
            infos = infos[area_index]

        padding = np.arange(0, len(neigs_id))
        neigs_index = np.empty_like(neigs_id)
        for i in range(neigs_index.shape[1]):
            neigs_index[:, i] = padding

        for nid, nes in enumerate(tqdm(neigs_id)):
            nes = nes[nes != None]
            inds = np.where(np.isin(infos[:, 0], nes))[0]
            length = len(inds)
            neigs_index[nid, :length] = inds
        np.save(self.save_root + '/neigs_index.npy', neigs_index.astype('int32'))



def Cache_infons(args):

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if not os.path.exists(args.cache_root):
        os.makedirs(args.cache_root)

    if not os.path.exists(args.cache_root + '/static_info.npy'):
        cache = Cache_DB(args)
        cache.Get_static_info()
        print('cache static_info and neigs infos')

    if not os.path.exists(args.cache_root + '/landmark.npy'):
        landmark = Landmark(args)
        print('cache landmark and landmark extent')

    if not os.path.exists(args.cache_root + '/neigs_index.npy'):
        cache = Cache_DB(args)
        cache.Get_neigs_index()
        print('cache neigs index')

    return


class Demo_randome:
    def __init__(self, cache_root):
        self.cache_root = cache_root
        self.demo_root = cache_root + '/demo'
        self.infos = np.load(cache_root + '/static_info.npy', allow_pickle=True)
        self.nes = np.load(cache_root + '/neig_info.npy', allow_pickle=True)
        if os.path.exists(cache_root + '/demo/route.npy'):
            self.route = np.load(cache_root + '/demo/route.npy', allow_pickle=True)

    def Randome_Route(self, start = 35000, rou_len = 1000):
        self.route = np.empty((rou_len, 11), dtype=object)
        last = start
        dele = None
        for r in range(rou_len):
            str = self.nes[last, 1:]
            str = str[str != None]
            strategy = str[str != dele]
            if len(strategy) < 1:
                strategy = str
            dele = self.infos[last, 0]
            act = np.random.choice(strategy)
            last = np.where(self.infos[:, 0] == act)[0]
            self.route[r] = self.infos[last]
        if not os.path.exists(self.demo_root):
            os.mkdir(self.demo_root)
        np.save(self.demo_root + '/route.npy', self.route)
        return

    def Map_Routing(self, back_pic = None, back_extent =None, frame_size=(800, 800), fps=10):
        coor = self.infos[1:, 1:5].astype('float32')
        coor_route = self.route[1:, 1:5].astype('float32')
        #w = 3427, h = 3295
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(self.demo_root + '/data_routing.mp4', fourcc, fps, frame_size)

        for step in tqdm(range(len(self.route))):
            plt.figure(figsize=(8, 8))
            plt.scatter(coor_route[:step, 0], coor_route[:step, 1], c='red', s=1)
            plt.scatter(coor[:, 0], coor[:, 1], c='grey', s=0.1, alpha=0.1)

            imp = plt.imread(back_pic)
            plt.imshow(imp, extent= back_extent)

            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(self.demo_root + '/temp_frame.jpg', dpi=200, bbox_inches='tight')
            plt.close()

            frame = cv2.imread(self.demo_root +'/temp_frame.jpg')
            frame = cv2.resize(frame, frame_size)
            video_writer.write(frame)

        video_writer.release()

        return


    def Pic_Routing(self, db_root, frame_size=(800, 800), fps=10):

        coor_ = self.route[:, 1:3].astype('float32')
        coor = np.concatenate((coor_[:-1, :], coor_[1:, :]), axis=1)
        cal = Calculating()
        heading_ = cal.heading(coor)
        heading = np.append(heading_, heading_[-1])

        route = self.route[:, 0]
        db = plyvel.DB(db_root, create_if_missing=False)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(self.demo_root + '/pic_routing.mp4', fourcc, fps, frame_size)
        pano = streetlearn_pb2.Pano()
        for pid, k in enumerate(tqdm(route)):
            values = db.get(k.encode('utf-8'))
            pano.ParseFromString(values)
            frame = cv2.imdecode(np.frombuffer(pano.compressed_image, dtype=np.uint8), cv2.IMREAD_COLOR)
            heads = -self.route[pid,-3] + heading[pid]
            persp_img = py360convert.e2p(frame, fov_deg=120, u_deg=heads, v_deg=0, out_hw=(800, 800))
            frame = cv2.resize(persp_img, frame_size)
            video_writer.write(frame)

        video_writer.release()

        return





if __name__ == '__main__':
    db_root_h = '/Users/yangkaisen/MyProject/Navigation/manhattan_2021_highres'
    db_root_l = '/Users/yangkaisen/MyProject/Navigation/manhattan_2021_lowres'
    cache_root = './cache'
    le = [74.00]

    back_pic = '/Users/yangkaisen/MyProject/Data/map/white.jpg'
    back_extent = [-74.035, -73.935, 40.695, 40.795]
    Demo = True

    if not os.path.exists(cache_root + '/static_info.npy'):
        cache = Cache_DB(db_root_l, cache_root)
        #cache.Get_static_info()
        #cache.Get_Pics()
        cache.Get_neigs_index()

    if Demo:
        demo = Demo_randome(cache_root)
        #demo.Randome_Route(rou_len=100)
        #demo.Map_Routing(back_pic = back_pic, back_extent = back_extent)
        demo.Pic_Routing(db_root = db_root_h)

    infos = np.load(cache_root + '/static_info.npy', allow_pickle=True)
    nes = np.load(cache_root + '/neig_info.npy', allow_pickle=True)

    print(';')

    plt.figure(figsize=(8, 8))
    coor = infos[1:, 1:5].astype('float32')
    plt.scatter(coor[:, 0], coor[:, 1], c='grey', s=0.1, alpha=0.1)

    route = np.load(cache_root + '/demo/route.npy', allow_pickle=True)
    coor_route = route[1:, 1:5].astype('float32')
    plt.scatter(coor_route[:, 0], coor_route[:, 1], c='red', s=0.2)

    imp = plt.imread(back_pic)
    plt.imshow(imp, extent= back_extent)
    plt.savefig(cache_root + "/pic/streeview.jpg", dpi=500, bbox_inches='tight')
    plt.show()

    print(';')



