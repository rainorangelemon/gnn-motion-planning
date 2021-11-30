import torch
import numpy as np
from environment import KukaEnv
from environment.graph import Graph, EdgeAttribute, create_data
from next_model import EncoderProcessDecoder
from agent import Agent
from replay import Replay, Data
from config import config, set_random_seed, nn_config
from tqdm import tqdm as tqdm
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.plot import plot_graph, plot_edges
import math
import pickle
from algorithm.bit_star_track import BITStar
import time
from utils.plot import plot_edges
from config import set_random_seed
from environment import KukaEnv
from tqdm import tqdm


def path(env, edges, explored_nodes):
    result = []
    state = tuple(env.goal_state)
    while True:
        result.append(explored_nodes.index(state))
        if state == tuple(env.init_state):
            break
        state = edges[state]
    result.reverse()
    return result


solutions = []
INF = float('inf')

environment = KukaEnv()

for _ in tqdm(range(3000)):
    pb = environment.init_new_problem()
    set_random_seed(1234)

    cur_time = time.time()

    BIT = BITStar(environment, batch_size=50)
    nodes, edges, explored_nodes, explored_edges, collision, success, n_samples, times = BIT.plan(INF, time_budget=300, refine_time_budget=0)

    if tuple(environment.goal_state) in explored_nodes:
        solutions.append((nodes, edges, explored_nodes, explored_edges, collision, success, n_samples, times,
                          path(environment, edges, explored_nodes)))
    else:
        solutions.append((nodes, edges, explored_nodes, explored_edges, collision, success, n_samples, times, []))

with open('kuka_bit_explore.pkl', 'wb') as f:
    pickle.dump(solutions, f, pickle.DEFAULT_PROTOCOL)
