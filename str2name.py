from model import EncoderProcessDecoder
from model_smoother import ModelSmoother
import model_smoother2
import torch
from environment import MazeEnv, KukaEnv, Kuka2Env, SnakeEnv, UR5Env
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def str2name(str, get_data=False, use_obstacle=True, load=False):
    if 'maze2' in str:
        env = MazeEnv(dim=2)
        model_explore = EncoderProcessDecoder(workspace_size=2, config_size=2, embed_size=32, obs_size=2).to(device)
        model_explore_path = 'data/weights/weights_maze.pt'
        model_smooth = ModelSmoother(workspace_size=env.dim, config_size=env.config_dim, embed_size=128, obs_size=6).to(device)
        model_smooth_path = 'data/weights/smooth_2d_attv3.pt'
        data_path = 'data/pkl/maze_prm_4000.pkl'

    elif str == 'maze3':
        env = MazeEnv(dim=3)
        model_explore = EncoderProcessDecoder(workspace_size=2, config_size=3, embed_size=32, obs_size=2).to(device)
        model_explore_path = 'data/weights/weights_maze_3.pt'
        model_smooth = ModelSmoother(workspace_size=env.dim, config_size=env.config_dim, embed_size=128, obs_size=6).to(device)
        model_smooth_path = 'data/weights/smooth_3d_attv3.pt'
        data_path = 'data/pkl/maze_prm_3.pkl'

    elif str == 'kuka7':
        env = KukaEnv()
        model_explore = EncoderProcessDecoder(workspace_size=3, config_size=7, embed_size=64, obs_size=6).to(device)
        model_explore_path = 'data/weights/weights_kuka.pt'
        model_smooth = ModelSmoother(workspace_size=env.dim, config_size=env.config_dim, embed_size=128, obs_size=6).to(device)
        model_smooth_path = 'data/weights/smooth_7d_attv3.pt'
        data_path = 'data/pkl/kuka_prm_4000.pkl'
        
    elif str == 'ur5':
        env = UR5Env()
        model_explore = EncoderProcessDecoder(workspace_size=3, config_size=6, embed_size=32, obs_size=6).to(device)
        model_explore_path = 'data/weights/weights_ur5.pt'
        model_smooth = ModelSmoother(workspace_size=3, config_size=6, embed_size=128, obs_size=6, scale=np.max(env.bound)).to(device)
        model_smooth_path = 'data/weights/smooth_ur5_attv3.pt'
        data_path = 'data/pkl/ur5_prm_3000.pkl'

    elif str == 'snake7':
        env = SnakeEnv(map_file='maze_files/snakes_15_2_3000.npz')
        model_explore = EncoderProcessDecoder(workspace_size=3, config_size=7, embed_size=32, obs_size=2).to(device)
        model_explore_path = 'data/weights/weights_snake.pt'
        model_smooth = ModelSmoother(workspace_size=env.dim, config_size=env.config_dim, embed_size=128, obs_size=6).to(device)
        model_smooth_path = 'data/weights/smooth_snake_attv3.pt'
        data_path = 'data/pkl/snake_prm_3000.pkl'

    elif str == 'kuka13':
        env = KukaEnv(kuka_file="kuka_iiwa/model_3.urdf", map_file="maze_files/kukas_13_3000.pkl")
        model_explore = EncoderProcessDecoder(workspace_size=3, config_size=13, embed_size=32, obs_size=6).to(device)
        model_explore_path = 'data/weights/weights_kuka_13.pt'
        model_smooth = ModelSmoother(workspace_size=env.dim, config_size=env.config_dim, embed_size=128, obs_size=6).to(device)
        model_smooth_path = 'data/weights/smooth_13d_attv3.pt'
        data_path = 'data/pkl/kuka_prm_13.pkl'

    elif str == 'kuka14':
        env = Kuka2Env()
        model_explore = EncoderProcessDecoder(workspace_size=3, config_size=14, embed_size=32, obs_size=6).to(device)
        model_explore_path = 'data/weights/kuka_14.pt'
        model_smooth = ModelSmoother(workspace_size=env.dim, config_size=env.config_dim, embed_size=128, obs_size=6).to(device)
        model_smooth_path = 'data/weights/smooth_14d_attv3.pt'
        data_path = 'data/pkl/kuka_prm_14.pkl'

    if not use_obstacle:
        model_explore_path = model_explore_path.replace('.pt', '_pure.pt')

    if load:
        model_explore.load_state_dict(torch.load(model_explore_path, map_location=device))
        model_explore.to(device)

        model_smooth.load_state_dict(torch.load(model_smooth_path, map_location=device))
        model_smooth.to(device)

    if get_data:
        return env, model_explore, model_explore_path, model_smooth, model_smooth_path, data_path
    else:
        return env, model_explore, model_explore_path, model_smooth, model_smooth_path