import torch
import numpy as np
from environment import MazeEnv
from next_model import EncoderProcessDecoder
from torch_geometric.data import Data
from config import config, set_random_seed
from tqdm import tqdm as tqdm
from tensorboardX import SummaryWriter
import pickle
from time import time
from algorithm.dijkstra import dijkstra


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
        free.append((0. for _ in range(env.config_dim)))
    if not len(collided):
        collided.append((0. for _ in range(env.config_dim)))
    data = DotDict({
        'free': torch.FloatTensor(free),
        'collided': torch.FloatTensor(collided),
        'obstacles': torch.FloatTensor(env.obstacles),
    })
    return data


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

writer = SummaryWriter()
INFINITY = float('inf')


env = MazeEnv(dim=2)
set_random_seed(1234)
epoch = 2000; iter = 20; graph_size = 200; loop=30
model = EncoderProcessDecoder(workspace_size=2, config_size=2, embed_size=64, obs_size=2).to(device)
try:
    model.load_state_dict(torch.load('weights.pt', map_location=device))
except:
    pass
with open('data/pkl/maze_prm.pkl', 'rb') as f:
    graphs = pickle.load(f)

T = 0
losses = []
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
optimizer.zero_grad()

for iter_i in range(iter):
    indexes = np.random.permutation(epoch)
    pbar = tqdm(indexes)
    for index in pbar:

        index = index % 2000

        pb = env.init_new_problem(index)

        time0 = time()
        points, neighbors, edge_cost, edge_index, edge_free = graphs[index]
        goal_index = np.random.choice(len(points))
        dist, prev = dijkstra(list(range(len(points))), neighbors, edge_cost, goal_index)
        prev[goal_index] = goal_index
        valid_node = (np.array(list(dist.values())) != INFINITY)
        if sum(valid_node) == 1:
            continue

        data = Data(goal=torch.FloatTensor(points[goal_index]),
                    y=torch.FloatTensor(points),
                    dist=torch.FloatTensor(list(dist.values())),
                    prev=torch.FloatTensor(list(prev.values())))
        data.edge_index = torch.LongTensor(edge_index.T)
        data.node_free = data.y.new_zeros(len(data.y), len(data.y))
        data.node_free[data.edge_index[0, :], data.edge_index[1, :]] = torch.FloatTensor(edge_free).squeeze()
        data.node_free = torch.diag(data.node_free, 0)

        time_data = time() - time0

        time0 = time()
        value, policy, node, edge = model.set_problem(**vars(data.to(device)), **obs_data(env), loop=np.random.randint(1, 10))
        value_loss = torch.nn.MSELoss()(value[valid_node].view(-1), data.dist[valid_node].view(-1))
        policy_loss = torch.nn.CrossEntropyLoss()(policy[valid_node], data.prev[valid_node].long())
        node_loss = torch.nn.BCEWithLogitsLoss()(node.squeeze(), data.node_free.squeeze())
        edge_loss = torch.nn.BCEWithLogitsLoss()(edge.squeeze(), torch.FloatTensor(edge_free).squeeze().to(device))
        loss = value_loss + policy_loss + node_loss + edge_loss
        loss.backward()
        losses.append((loss, value_loss, policy_loss, node_loss, edge_loss))
        time_train = time() - time0

        time0 = time()
        if T % 8 == 0:
            optimizer.step()
            optimizer.zero_grad()

            writer.add_scalar('train/total_loss', sum([loss[0] for loss in losses]) / len(losses), T)
            writer.add_scalar('train/value_loss', sum([loss[1] for loss in losses]) / len(losses), T)
            writer.add_scalar('train/policy_loss', sum([loss[2] for loss in losses]) / len(losses), T)
            writer.add_scalar('train/node_loss', sum([loss[3] for loss in losses]) / len(losses), T)
            writer.add_scalar('train/edge_loss', sum([loss[4] for loss in losses]) / len(losses), T)
            losses = []

            torch.save(model.state_dict(), 'data/weights/weights_maze.pt')

        T += 1
        time_bp = time() - time0

        pbar.set_description("data %.2fs, train %.2fs, bp %.2fs" % (time_data, time_train, time_bp))

torch.save(model.state_dict(), 'data/weights/weights_maze.pt')
writer.close()
