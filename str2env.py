from model import EncoderProcessDecoder
from model_smoother import ModelSmoother
# import model_smoother2
import torch
from environment import MazeEnv, KukaEnv, Kuka2Env, SnakeEnv, UR5Env
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def str2env(str):
    if str == 'maze2easy':
        env = MazeEnv(dim=2)
        indexes = np.arange(2000, 3000)
        
    elif str == 'maze2hard':
        env = MazeEnv(dim=2, map_file='maze_files/mazes_hard.npz')
        indexes = np.arange(1000)        

    elif str == 'kuka7':
        env = KukaEnv()
        indexes = np.arange(2000, 3000)
        
    elif str == 'ur5':
        env = UR5Env()
        indexes = np.arange(2000, 3000)

    elif str == 'snake7':
        env = SnakeEnv(map_file='maze_files/snakes_15_2_3000.npz')
        indexes = np.arange(2000, 3000)

    elif str == 'kuka13':
        env = KukaEnv(kuka_file="kuka_iiwa/model_3.urdf", map_file="maze_files/kukas_13_3000.pkl")
        indexes = np.arange(2000, 3000)

    elif str == 'kuka14':
        env = Kuka2Env()
        indexes = np.arange(2000, 3000)

    return env, indexes