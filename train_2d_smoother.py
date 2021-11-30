import torch
import numpy as np
from environment import MazeEnv
from model_smoother2 import ModelSmoother
from config import config, set_random_seed
from tqdm import tqdm as tqdm
from tensorboardX import SummaryWriter
from algorithm.bit_star import BITStar
from smoother import random_path_smoother
from torch_geometric.utils import add_self_loops


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def obs_data(env):
    free = []
    collided = []
    for i in range(np.random.randint(10, 100)):
        new_sample = env.uniform_sample()
        if env._state_fp(new_sample):
            free.append(new_sample)
        else:
            collided.append(new_sample)
    if not len(free):
        free.append((0., 0.))
    if not len(collided):
        collided.append((0., 0.))
    data = DotDict({
        'free': torch.FloatTensor(free),
        'collided': torch.FloatTensor(collided),
        'obstacles': torch.FloatTensor(env.obstacles),
    })
    return data


def train(replay, model, optimizer):
    if len(replay) <= 8:
        return 0.

    optimizer.zero_grad()

    loss = 0.
    batch_idx = np.random.choice(len(replay), size=8, replace=False)
    for idx in batch_idx:
        path_origin, path_smooth, data = replay[idx]
        data.path = torch.FloatTensor(path_origin)
        data.edge_index = torch.cat((torch.arange(1, len(path_origin)).reshape(1, -1),
                                     torch.arange(0, len(path_origin)-1).reshape(1, -1)), dim=0)
        data.edge_index = torch.cat((data.edge_index, data.edge_index.flip(0)), dim=-1)
        data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=len(data.path))

        path_pred = model(**data, loop=np.random.randint(1, 10))

        loss += torch.nn.MSELoss()(torch.FloatTensor(path_smooth), path_pred)
    loss.backward()

    optimizer.step()
    return loss


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

writer = SummaryWriter()
INFINITY = float('inf')


env = MazeEnv(dim=2)
set_random_seed(1234)
epoch = 2000; iter = 20; loop=30
model = ModelSmoother(workspace_size=2, config_size=2, embed_size=32, obs_size=2).to(device)
try:
    model.load_state_dict(torch.load('data/weights/weights_smooth.pt', map_location=device))
except:
    pass

replay = []
losses = []
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
optimizer.zero_grad()

for iter_i in range(iter):

    indexes = np.random.permutation(epoch)
    pbar = tqdm(indexes)

    for index in pbar:
        env.init_new_problem(index)
        env.set_random_init_goal()

        BIT = BITStar(env)
        BIT.plan(float('inf'), refine_time_budget=0, time_budget=5)
        path_origin = BIT.get_best_path()
        if len(path_origin) < 2:
            continue
        path_smooth = random_path_smoother(BIT.get_best_path(), env.RRT_EPS, env)

        replay.append((path_origin, path_smooth, obs_data(env)))
        loss = train(replay, model, optimizer)
        losses.append(float(loss))
        torch.save(model.state_dict(), 'data/weights/weights_smooth.pt')

        pbar.set_description("loss: %.5f" % np.mean(losses))
        writer.add_scalar('loss', loss)

    losses = []

torch.save(model.state_dict(), 'data/weights/weights_smooth.pt')
writer.close()
