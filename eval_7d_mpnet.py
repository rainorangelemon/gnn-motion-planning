import torch
import numpy as np
from environment import KukaEnv
from next_model import EncoderProcessDecoder
from torch_geometric.data import Data
from config import set_random_seed
from tqdm import tqdm as tqdm
from time import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

writer = None
INFINITY = float('inf')


env = KukaEnv()
set_random_seed(1234)
epoch = 2000; iter = 40; graph_size = 200; loop=30
model = EncoderProcessDecoder(workspace_size=3, config_size=7, embed_size=32, map_size=9261).to(device)
model.load_state_dict(torch.load('weights.pt', map_location=device))
model.eval()

pbar = tqdm(range(2000, 3000))
solutions = []
for index in pbar:

    pb = env.init_new_problem(index)

    time0 = time()

    data = Data(x_obstacles=torch.FloatTensor(env.obstacles),
                x_goal=torch.FloatTensor(env.goal_state),
                path=torch.FloatTensor(env.path))
    next_node_index = np.array([np.random.choice(len(data.path)-1)])
    data.target = data.path[0:1, :]

    time_data = time() - time0

    time0 = time()
    path = []
    while not env.in_goal_region(data.target[0, :].data.cpu().numpy()) and len(path) < 100:
        path.append(data.target[0, :].data.cpu().numpy())
        policy = model.reactive_policy(**vars(data.to(device)))
        data.target = data.target + 0.5 * policy / policy.norm(dim=-1, keepdim=True)

    time_infer = time() - time0

    solutions.append((path, len(path), env.in_goal_region(data.target[0, :].data.cpu().numpy())))

    pbar.set_description("data %.2fs, infer %.2fs" % (time_data, time_infer))

print('hello')
