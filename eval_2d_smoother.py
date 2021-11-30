import torch
from environment import MazeEnv
from model_smoother2 import ModelSmoother
from config import set_random_seed
from tensorboardX import SummaryWriter
from algorithm.bit_star import BITStar
from torch_geometric.utils import add_self_loops
from utils.plot import plot_edges


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def obs_data(env):
    free = []
    collided = []
    for i in range(128):
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


def eval(path_origin, data, model):

    data.path = torch.FloatTensor(path_origin)
    data.edge_index = torch.cat((torch.arange(1, len(path_origin)).reshape(1, -1),
                                 torch.arange(0, len(path_origin)-1).reshape(1, -1)), dim=0)
    data.edge_index = torch.cat((data.edge_index, data.edge_index.flip(0)), dim=-1)
    data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=len(data.path))

    path_pred = model(**data, loop=30)

    return path_pred


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
model.eval()

for index in range(2000, 2010):
    env.init_new_problem(index)

    BIT = BITStar(env)
    BIT.plan(float('inf'), refine_time_budget=0, time_budget=5)
    path_origin = BIT.get_best_path()
    if len(path_origin) < 2:
        continue
    path_smooth = eval(path_origin, obs_data(env), model)

    plot_edges(states=path_origin,
               edges={path_origin[i]: path_origin[i + 1] for i in range(len(path_origin) - 1)},
               problem=env.get_problem())

    path_smooth = [tuple(node) for node in list(path_smooth.data.cpu().numpy())]
    plot_edges(states=path_smooth,
               edges={path_smooth[i]: path_smooth[i + 1] for i in range(len(path_smooth) - 1)},
               problem=env.get_problem())
