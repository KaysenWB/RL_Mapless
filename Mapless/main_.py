import os
from trainer import PPOTrainer
import argparse
import warnings
warnings.filterwarnings('ignore')
from cache import Cache_infons


parser = argparse.ArgumentParser( description='Mapless_Navigation')

parser.add_argument('--db_root_l', default='..', type=str)
parser.add_argument('--db_root_h', default='/home/user/Documents/Yangkaisen/Navigations/Streetview/manhattan_2022_highres/manhattan_2021_highres', type=str)
parser.add_argument('--back_pic', default='./white.png', type=str)
parser.add_argument('--cache_root', default='./cache')
parser.add_argument('--save_dir', default='./results')
parser.add_argument('--le', default=[-73.995, -73.97, 40.747, 40.765], type=list,
                    help = "landmark_extent: [lon_min, lon_max, lat_min, lat_max]")

parser.add_argument('--agent_name', default="AgentPosi", type=str, help="AgentPosi, AgentPics, AgentMlp")
parser.add_argument('--env_name', default="CourierEnv", type=str)
parser.add_argument('--max_buf', default=100, type=int)
parser.add_argument('--landmark_num', default=50, type=int)
parser.add_argument('--num_actions', default=3, type=int)
parser.add_argument('--dropout', default=0.5, type=float)
parser.add_argument('--hidden_feats', default=128, type=int)
parser.add_argument('--state_fests', default=4, type=int)
parser.add_argument('--goal_range', default=100, type=int)
parser.add_argument('--punish', default=-3, type=int)

parser.add_argument('--max_len', default=800, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--batch_train', default=320, type=int)
parser.add_argument('--update_times', default=6, type=int)
parser.add_argument('--epoch', default=50, type=int)
parser.add_argument('--device', default='cuda:1', type=str)
parser.add_argument('--gamma', default=0.99, type=float, help='reward for future')
parser.add_argument('--clip_epsilon', default=0.2, type=float,
                    help='strategic shear, 0.1-0.3. 0.3 exploratory and 0.1 stable')
parser.add_argument('--ent_coef', default=0.01, type=float,
                    help='entropy coefficient, 0.0001-0.1. 0.001 strategy is more deterministic, 0.1 more stochastic')
parser.add_argument('--learning_rate', default=3e-2, type=float)

# alter
parser.add_argument('--feed_action_and_reward', default=True, type=bool)
parser.add_argument('--heading_out', default=16, type=int)
parser.add_argument('--bins_lng', default=32, type=int)
parser.add_argument('--bins_lat', default=32, type=int)

args = parser.parse_args()


Cache_infons(args)
print('Cache infos finished')


ppo = PPOTrainer(args)
print('Start Training')
ppo.train()
ppo.test()





