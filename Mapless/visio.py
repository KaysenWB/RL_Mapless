import numpy as np
import matplotlib.pyplot as plt
from utils import Calculating




def draw_rewads(res, labels = "None", colors = "None", av_len = 150,  picf = None):

    plt.figure(figsize=(10, 8))

    for rd, r in enumerate(res):
        #plt.plot(r, alpha=0.15, c=colors[rd])
        if len(r) >= av_len:
            smooth = np.convolve(r, np.ones(3) / 3, mode='valid')
            plt.plot(range(3-1, 3-1 + len(smooth)), smooth, alpha=0.15, c=colors[rd])

        if len(r) >= av_len:
            smooth = np.convolve(r, np.ones(av_len) / av_len, mode='valid')
            plt.plot(range(av_len-1, av_len-1 + len(smooth)), smooth, c=colors[rd], label=labels[rd])

        #plt.axhline(y= np.mean(r), color='grey', alpha=0.7, linestyle='--', )
        print(np.mean(r))
    if le == "min":
        plt.ylim(-6, 6)
    #if le == "mid":
    #    plt.ylim(-5, 15)
    plt.legend()
    #plt.savefig(fname=save_dir + f"/reward_{picf}.jpg", dpi=500, format='jpg', bbox_inches='tight')
    plt.show()

    return
le = "min"
save_dir = f"./results/StreetPicEnv_AgentPosi/{le}/pic"
labels = ["lonlat_lonlat", "lonlat_marker", "pics_lonlat", "pics_marker",
          "pics_marker_conti", "pics_marker_Rconti", "pics_marker_disc","pics_marker_head","test"]
colors = ["red","purple", "green", "blue", "grey", "pink", "orange", "yellow","red"]

res = [np.load(save_dir + f"/ep_rewards_{i}.npy") for i in range(8) ]
draw_rewads(res[:4], labels[:4], colors[:4], picf = "f4")
draw_rewads(res[3:], labels[3:], colors[3:], picf = "b4")




"""nodes = np.load("/Users/yangkaisen/MyProject/Navigation/Mapless/cache_mid/static_info.npy", allow_pickle=True)
edges = np.load("/Users/yangkaisen/MyProject/Navigation/Mapless/cache_mid/neig_info.npy", allow_pickle=True)
edges_ind = np.load("/Users/yangkaisen/MyProject/Navigation/Mapless/cache_mid/neigs_index.npy", allow_pickle=True)

area_index = np.load("/Users/yangkaisen/MyProject/Navigation/Mapless/cache_mid/area_index.npy", allow_pickle=True)
nodes = nodes[area_index, :]
edges = edges[area_index, :]

nodes = nodes[area_index, 1:3]
plt.scatter(nodes[:, 0], nodes[:, 1], s=0.1, c="grey")
plt.scatter(nodes[0, 0], nodes[0, 1], s=5, c="red")
plt.show()
print(';')"""