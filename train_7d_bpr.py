import torch
import numpy as np
from environment import KukaEnv
from next_model import EncoderProcessDecoder
from torch_geometric.data import Data
from config import config, set_random_seed
from tqdm import tqdm as tqdm
from tensorboardX import SummaryWriter
import pickle
from time import time
from algorithm.dijkstra import dijkstra


def path(prev, start_index, goal_index):
    result = [start_index]
    while start_index!= goal_index:
        start_index = prev[start_index]
        result.append(start_index)
    return result


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

writer = SummaryWriter()
INFINITY = float('inf')


env = KukaEnv()
set_random_seed(1234)
epoch = 2000; iter = 40; graph_size = 200; loop=30; train_collision=0
model = EncoderProcessDecoder(workspace_size=3, config_size=7, embed_size=32, map_size=9261).to(device)
try:
    model.load_state_dict(torch.load('weights.pt', map_location=device))
except:
    pass
with open('kuka_prm.pkl', 'rb') as f:
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

        data = Data(x_obstacles=torch.FloatTensor(env.obstacles),
                    x_goal=torch.FloatTensor(points[goal_index]),
                    y=torch.FloatTensor(points),
                    dist=torch.FloatTensor(list(dist.values())),
                    prev=torch.FloatTensor(list(prev.values())))
        data.edge_index = torch.LongTensor(edge_index.T)
        data.node_free = data.y.new_zeros(len(data.y), len(data.y))
        data.node_free[data.edge_index[0, :], data.edge_index[1, :]] = torch.FloatTensor(edge_free).squeeze()
        data.node_free = torch.diag(data.node_free, 0)
        start_index = np.where(valid_node)[0][np.random.randint(sum(valid_node))]
        good_path = path(prev, start_index, goal_index)
        part_path_index = np.random.randint(len(good_path))
        index_pos = (good_path[part_path_index], prev[good_path[part_path_index]])
        index_neg_a = good_path[np.random.randint(part_path_index+1)]
        index_neg_b = np.random.choice(list(neighbors[index_neg_a]))
        index_neg = (index_neg_a, index_neg_b)

        time_data = time() - time0

        time0 = time()
        if iter_i < train_collision:
            value, policy, node, edge = model.set_problem(**vars(data.to(device)), loop=1, need_softmax=False)
        else:
            value, policy, node, edge = model.set_problem(**vars(data.to(device)), loop=np.random.randint(1, 10), need_softmax=False)
        value_loss = torch.nn.MSELoss()(value[valid_node].view(-1), data.dist[valid_node].view(-1))
        policy[policy==0] = -float('inf')
        policy_loss = torch.nn.CrossEntropyLoss()(policy[valid_node].softmax(dim=-1), data.prev[valid_node].long())
        bpr_loss = -(policy[index_pos] - policy[index_neg]).sigmoid().log().sum()
        node_loss = torch.nn.BCEWithLogitsLoss()(node.squeeze(), data.node_free.squeeze())
        edge_loss = torch.nn.BCEWithLogitsLoss()(edge.squeeze(), torch.FloatTensor(edge_free).squeeze().to(device))
        if iter_i < train_collision:
            loss = node_loss + edge_loss
        else:
            loss = value_loss + policy_loss + bpr_loss + node_loss + edge_loss
        loss.backward()
        losses.append((loss, value_loss, policy_loss, node_loss, edge_loss, bpr_loss))
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
            writer.add_scalar('train/bpr_loss', sum([loss[5] for loss in losses]) / len(losses), T)
            losses = []

            torch.save(model.state_dict(), 'weights.pt')

        T += 1
        time_bp = time() - time0

        pbar.set_description("data %.2fs, train %.2fs, bp %.2fs" % (time_data, time_train, time_bp))

    torch.save(model.state_dict(), 'weights_%s.pt' % str(iter_i))

torch.save(model.state_dict(), 'weights.pt')
writer.close()
