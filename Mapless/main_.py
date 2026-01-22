import os
from exp import Exps
import argparse
import warnings
warnings.filterwarnings('ignore')
from cache import Cache_infons
import shutil



parser = argparse.ArgumentParser( description='Mapless_Navigation')

# data
parser.add_argument('--db_root_l', default='..', type=str)
parser.add_argument('--db_root_h', default='/Users/yangkaisen/MyProject/Navigation/manhattan_2021_highres', type=str)
parser.add_argument('--back_pic', default='./white.png', type=str)
parser.add_argument('--cache_root', default='./cache')
parser.add_argument('--save_dir', default='./results')
parser.add_argument('--le', default=[], type=list)
parser.add_argument('--max_buf', default=100, type=int)
# env and agents
parser.add_argument('--agent_name', default="AgentPosi", type=str, help="AgentPosi, AgentPics, AgentMlp")
parser.add_argument('--env_name', default="StreetPicEnv", type=str, help="StreetEnv, StreetPicEnv, GridEnv")
parser.add_argument('--load_pics', default = "cc", type=str, help=["rt, cc"]) # realtime and cache
parser.add_argument('--num_actions', default=4, type=int)

parser.add_argument('--update_times', default=6, type=int)
parser.add_argument("--grids_size", type=int, default=20)
parser.add_argument('--learning_rate', type=float, default=0.001)#2e-4
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument("--log_interval", type=int, default=2000)#2000
parser.add_argument('--gamma', default=0.995, type=float)
parser.add_argument('--clip_eps', default=0.2, type=float)
parser.add_argument("--entropy_coef", type=float, default=0.004)
parser.add_argument("--value_coef", type=float, default=0.5)
parser.add_argument("--max_grad_norm", type=float, default=0.5)
parser.add_argument("--gae_lambda", type=float, default=0.95)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument('--device', default='cpu', type=str)

# changes
parser.add_argument('--le_flag', default= "min", type=str)
parser.add_argument('--visual', default=True, type=bool)
parser.add_argument('--pixel', default=84, type=bool)
parser.add_argument('--landmark_num', default=100, type=int)
parser.add_argument('--hidden_feats', default=128, type=int)
parser.add_argument('--state_fests', default=52, type=int)
parser.add_argument('--goal_range', default=5, type=int)
parser.add_argument("--total_steps", default=5e5, type=int) #2e6
parser.add_argument('--rollout_len', default=1500, type=int) # 2000
parser.add_argument('--max_steps', default=150, type=int) # 400
parser.add_argument("--test_episodes", type=int, default=300)


args = parser.parse_args()

args.cache_root = args.cache_root + '_' + args.le_flag
args.save_dir = args.save_dir + "/" + args.env_name + '_' + args.agent_name
if not args.env_name == "GridEnv":
    args.save_dir = args.save_dir + "/" + args.le_flag

lerning_extent_dict = {
    "max": [-74.035, -73.935, 40.695, 40.795],
    "mid": [-73.990, -73.976, 40.756, 40.765],
    "min": [-73.990, -73.985, 40.750, 40.755],

    #"mid_5000": [-73.994, -73.97, 40.750, 40.765]
}
args.le = lerning_extent_dict[args.le_flag]

Cache_infons(args)
print('Cache infos finished')

exp = Exps(args)
exp.train()
exp.test()


# conti exps
"""agents_name = ["lonlat_lonlat", "lonlat_marker", "pics_lonlat", "pics_marker",
          "pics_marker_conti", "pics_marker_Rconti", "pics_marker_disc"]

for i in range(len(agents_name)):
    args.agent_name = agents_name[i]
    if i > 1:
        args.visual = True
    exp = Exps(args)
    exp.train()
    exp.test()

    files = [
        ("ep_rewards.npy", f"ep_rewards_{i}.npy"),
        ("test.jpg", f"test_{i}.jpg"),
        ("agent.pth", f"agent_{i}.pth")
    ]
    for src_name, dst_name in files:
        src = f"{args.save_dir}/{src_name}"
        dst = f"{args.save_dir}/pic/{dst_name}"
        shutil.copy2(src, dst)"""
