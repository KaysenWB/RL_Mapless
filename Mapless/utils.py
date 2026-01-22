import os.path
import numpy as np
import matplotlib.pyplot as plt
import math
import torch

class Calculating:

    def heading_torch(self, coor):

        lon1, lat1, lon2, lat2 = torch.deg2rad(coor.T)
        delta_lon = lon2 - lon1
        y = torch.sin(delta_lon) * torch.cos(lat2)
        x = torch.cos(lat1) * torch.sin(lat2) - torch.sin(lat1) * torch.cos(lat2) * torch.cos(delta_lon)
        headings_rad = torch.atan2(y, x)
        headings_deg = torch.rad2deg(headings_rad)
        headings_deg = (headings_deg + 360)
        headings_deg = (headings_deg + 360) % 360

        return headings_deg

    def heading_np(self, coor):

        lon1, lat1, lon2, lat2 = np.deg2rad(coor.T)
        delta_lon = lon2 - lon1

        y = np.sin(delta_lon) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon)

        headings_rad = np.arctan2(y, x)
        headings_deg = np.degrees(headings_rad)
        headings_deg = (headings_deg + 360) % 360

        return headings_deg

    def distance_torch(self, coor):

        lon1, lat1, lon2, lat2 = torch.deg2rad(coor.T)
        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

        return  c * 6371.0

    def distance_np(self, coor):

        lon1, lat1, lon2, lat2 = np.deg2rad(coor.T)
        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        return  c * 6371.0

    def distance_torch_appro(self, ten1, ten2):
        return torch.sqrt(torch.sum((ten1 - ten2)**2, dim=1))

    def distance_np_appro(self, arr1, arr2):
        return np.sqrt(np.sum((arr1 - arr2) ** 2, axis=1))


    def heading(self, coor):
        # from fist(start) point to second point(end)
        if isinstance(coor, torch.Tensor):
            return self.heading_torch(coor)
        else:
            return self.heading_np(coor)

    def distance(self, coor):
        # Unit: m
        if isinstance(coor, torch.Tensor):
            return self.distance_torch(coor) * 1000
        else:
            return self.distance_np(coor) * 1000

def Get_circle(centers, radius=75, num_points=36, device = 'cpu'):

    # get correponding diff of lon and lat for 100 m, used in env
    #diff = (self.coor[:, 2:] - self.coor[:, :2]).abs()
    #unit = (diff / self.distance) * 100
    #unit = torch.round(torch.mean(unit, dim=0), decimals=4)

    angles = torch.linspace(0, 2 * math.pi, num_points).to(device)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    rad = radius / 100
    diff = torch.tensor([0.0007, 0.0006]) * rad # 100m

    lon = centers[:1].repeat(num_points) + diff[0] * cos
    lat = centers[1:].repeat(num_points) + diff[1] * sin

    return torch.stack((lon, lat), dim=-1)



def Visio(coor_1, coor_2, coor_3, save = None):

    back_pic = '/Users/yangkaisen/MyProject/Data/map/white.jpg'
    back_extent = [-74.035, -73.935, 40.695, 40.795]

    plt.figure(figsize=(8, 8))
    plt.scatter(coor_1[:, 0], coor_1[:, 1], c='grey', s=0.1, alpha=0.1)
    plt.scatter(coor_2[:, 0], coor_2[:, 1], c='grey', s=0.1, alpha=0.3)
    plt.scatter(coor_3[:, 0], coor_3[:, 1], c='red', s=1, alpha=1)

    imp = plt.imread(back_pic)
    plt.imshow(imp, extent=back_extent)

    if not save == None:
        if not os.path.exists(save):
            plt.savefig(save, dpi=500, bbox_inches='tight')
    plt.show()
    return


class Standard:
    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data)
        self.mean = data.mean(0).repeat(2)
        self.std = data.std(0).repeat(2)

    def trans(self, data):
        return (data - self.mean) / self.std

    def inver_trans(self, data):
        return (data * self.std) + self.mean


    def mercator(self, lon_lat):
        scale = 80
        x = torch.deg2rad(lon_lat[:, 0])
        y = torch.log(torch.tan(math.pi / 4 + torch.deg2rad(lon_lat[:, 1]) / 2))
        x_norm = (x - x.min()) / (x.max() - x.min())
        y_norm = (y - y.min()) / (y.max() - y.min())
        out = torch.stack([x_norm * scale, y_norm * scale], dim=1)
        return out


if __name__ == '__main__':
    # test cal_heading
    cal = Calculating()
    coor1 = np.array((0, 0, 1, 1))
    coor2 = np.array((1, 1, 0, 0))
    heading1 = cal.heading(coor1)  # 45
    heading2 = cal.heading(coor2)  # 225


    rout = np.load('/Users/yangkaisen/MyProject/Navigation/Mapless/cache/demo/route.npy', allow_pickle=True)
    coor_ = rout[:,1:3].astype('float32')
    coor = np.concatenate((coor_[:-1,:],coor_[1:,:]),axis=1)
    ten = torch.tensor(coor)

    dis = cal.distance(coor)
    heading = cal.heading(coor)
    print('l')

