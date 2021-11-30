import torch
import numpy as np
from environment import MazeEnv
from environment.graph import Graph, EdgeAttribute, create_data
from next_model import EncoderProcessDecoder
from agent import Agent
from replay import Replay, Data
from config import config, set_random_seed, nn_config
from tqdm import tqdm as tqdm
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.plot import plot_graph, plot_edges
from algorithm import RRTS_plan

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

set_random_seed(1234)
if str(device) == 'cpu':
    epoch = 7; iter = 5; graph_size = 500; loop = 20
else:
    epoch = 300; iter = 5; graph_size = 500; loop = 30

env = MazeEnv(dim=2)
model = EncoderProcessDecoder(workspace_size=2, config_size=2, embed_size=32, resolution=env.voxel_r, latent_graph_size=125)
model.load_state_dict(torch.load('weights.pt', map_location=torch.device('cpu')))

agent = Agent(model, env, Replay(), **config)

T = 0
loss_mean = 0
indexes = np.random.permutation(np.arange(2000, 2000 + epoch))
for index in tqdm(indexes):

    env.init_new_problem(index)
    set_random_seed(index)

    import matplotlib.pyplot as plt

    # generate 2 2d grids for the x & y bounds
    y, x = np.meshgrid(np.linspace(-1, 1, 15), np.linspace(-1, 1, 15))
    data = create_data(env, np.hstack((x.reshape(-1, 1), y.reshape((-1, 1)))))
    value, policy = model(**vars(data), train=False, loop=loop)
    z = value.squeeze().data.cpu().numpy().reshape(y.shape)

    fig, ax = plt.subplots(dpi=200)

    c = ax.pcolormesh(x, y, z, cmap='RdBu', vmin=np.min(z), vmax=np.max(z))
    ax.set_title('pcolormesh')
    # set the limits of the plot to the limits of the data
    ax.axis([x.min(), x.max(), y.min(), y.max()])
    fig.colorbar(c, ax=ax)

    plt.title('Values')
    plt.show()

    plot_edges([], {}, env.get_problem())
    #
    # plot_edges(path[:path_index], {}, env.get_problem())
    #
    # highest_value = np.max(value[sample_index[-path_index:]])
    # good_place = (value >= highest_value)
    # plot_edges([sample for i, sample in enumerate(samples) if good_place[sample_index[i]]], [], env.get_problem())