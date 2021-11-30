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
from algorithm.bit_star import BITStar

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

set_random_seed(1234)
if str(device) == 'cpu':
    epoch = 2000; iter = 5; graph_size = 500; loop = 20
else:
    epoch = 300; iter = 5; graph_size = 500; loop = 30

env = MazeEnv(dim=2)
model = EncoderProcessDecoder(2, 1, 1, 1, 32, 3)
model.load_state_dict(torch.load('weights.pt', map_location=torch.device('cpu')))

agent = Agent(model, Replay(), **config)

T = 0
loss_mean = 0
BIT_solutions = []


def informed_sample(c_best, sample_num, vertices):
    samples = env.free_map()
    data = create_data(env, samples + vertices)
    with torch.no_grad():
        value, _, _ = model(**vars(data), loop=loop)
    good_voxel_mask = (value >= value[-len(vertices):].max()).data.cpu().numpy().squeeze()[:len(samples)]
    prob = 1. + good_voxel_mask.astype(float)

    voxel_index = ((np.array(samples) + 1.)*15 / 2.).astype(int)
    voxel_origin = voxel_index*2./15. - 1.

    samples = np.random.choice(len(samples), size=sample_num, p=prob/np.sum(prob))
    samples = voxel_origin[samples, :] + np.random.uniform(low=0, high=2. / 15., size=(sample_num, 2))

    return [tuple(sample) for sample in samples.tolist()]


for index in tqdm(range(2000, 3000)):

    env.init_new_problem(index)
    set_random_seed(index)

    BIT_solutions.append(BITStar(env, T=200, batch_size=50, sampling=informed_sample).plan(float('INF')))

    # if BIT_solutions[-1][-2]==float('inf'):
    #     nodes, edges, collision, success, n_samples = BIT_solutions[-1]
    #     plot_edges(set(nodes)|set(edges.keys()), [(k, v) for k, v in edges.items()], env.get_problem())

print(np.sum([solution[-2]!=float('inf') for solution in BIT_solutions]))
print(np.mean([solution[2] for solution in BIT_solutions if solution[-2]!=float('inf')]))
print(np.mean([solution[-2] for solution in BIT_solutions if solution[-2]!=float('inf')]))