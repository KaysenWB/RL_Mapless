import numpy as np
import matplotlib.pyplot as plt




infos = np.load('./cache/static_info.npy', allow_pickle=True)
#neigs = np.load('./cache/neig_info.npy', allow_pickle=True)
traj = np.load('./results/traj_state.npy', allow_pickle=True)
back_pic = './white.png'
back_extent = [-74.035, -73.935, 40.695, 40.795]
map_coor = infos[1:, 1:3].astype('float32')
_ka = 10
ka = 20
save_route= traj[:, _ka:ka]
np.save('./results/save_route.npy',save_route)
see = np.load('./results/save_route.npy')
def show_all_time():

    route_coor = traj[:, _ka:ka, :2]
    start_coor = traj[:1, _ka:ka, :2]
    goal_coor = traj[:, _ka:ka, 2:4]
    star_to_goal = np.concatenate((traj[:1, _ka:ka, :2], traj[-1:, _ka:ka, 2:]))

    plt.figure(figsize=(8, 8))
    plt.scatter(map_coor[:, 0], map_coor[:, 1], c='grey', s=0.1, alpha=0.1)
    plt.scatter(route_coor[:, :, 0], route_coor[:, :, 1], c='green', s=1)
    plt.scatter(goal_coor[:, :, 0], goal_coor[:, :, 1], c='red', s=1)
    #plt.scatter(start_coor[:, :, 0], start_coor[:, :, 1], c='black', s=5)

    for i in range(star_to_goal.shape[1]):
        plt.plot(star_to_goal[:, i, 0], star_to_goal[:, i, 1], color = 'grey', lw = 1)

    imp = plt.imread(back_pic)
    plt.imshow(imp, extent= back_extent)
    plt.savefig("./results/streeview.jpg", dpi=500, bbox_inches='tight')
    plt.show()

    print(';')


def show_for_time():

    route_coor = traj[:, _ka:ka, :]

    for t in range(0, traj.shape[0]+1, 10):
        plt.figure(figsize=(8, 8))
        plt.scatter(map_coor[:, 0], map_coor[:, 1], c='grey', s=0.1, alpha=0.1)
        plt.scatter(route_coor[:t, :, 0], route_coor[:t, :, 1], c='green', s=1)
        plt.scatter(route_coor[:t, :, 2], route_coor[:t, :, 3], c='red', s=1)
        start_goal_ = route_coor[t:t + 1]
        start_goal = np.concatenate((start_goal_[:1, :, :2], start_goal_[:1, :, 2:]))
        for i in range(route_coor.shape[1]):
            plt.plot(start_goal[:, i, 0], start_goal[:, i, 1], color='grey', lw=0.5)
        imp = plt.imread(back_pic)
        plt.imshow(imp, extent=back_extent)
        plt.savefig(f"./results/pic/show_{t}.jpg", dpi=300, bbox_inches='tight')
        #plt.show()

        print(';')

show_all_time()
show_for_time()