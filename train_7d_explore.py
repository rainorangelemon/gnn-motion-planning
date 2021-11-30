import torch
import numpy as np
from environment import KukaEnv
from model_explore import Explorer
from torch_geometric.data import Data
from config import config, set_random_seed
from tqdm import tqdm as tqdm
from tensorboardX import SummaryWriter
import pickle
from time import time
from torch_sparse import coalesce
from torch_geometric.utils import add_self_loops

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

writer = SummaryWriter()
INFINITY = float('inf')


env = KukaEnv()
set_random_seed(1234)
epoch = 2000; iter = 20; graph_size = 200; loop=30
model = Explorer(workspace_size=3, config_size=7, embed_size=32, map_size=9261).to(device)
try:
    model.load_state_dict(torch.load('weights.pt', map_location=device))
except:
    pass
with open('kuka_bit_explore.pkl', 'rb') as f:
    solutions = pickle.load(f)

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
        nodes, edges, explored_nodes, explored_edges, collision, success, n_samples, times, path = solutions[index]
        if len(path)==0:
            continue
        next_explore_node_index = np.random.randint(1, len(path))
        next_explore_node = path[next_explore_node_index]
        next_explore_edge = (path[next_explore_node_index-1], path[next_explore_node_index])
        try:
            current_tree_last_node = np.random.randint(path[next_explore_node_index-1], next_explore_node)
        except:
            continue
        current_tree_edge = [edge for edge in explored_edges
                             if (edge[0]<=current_tree_last_node and edge[1]<=current_tree_last_node)]
        current_tree_node = explored_nodes[:(current_tree_last_node+1)]

        data = Data(x_obstacles=torch.FloatTensor(env.obstacles),
                    x_goal=torch.FloatTensor(env.goal_state),
                    y=torch.FloatTensor(current_tree_node),
                    y_free=torch.BoolTensor([env._state_fp(np.array(point)) for point in current_tree_node]))
        data.edge_index = torch.LongTensor(np.array(current_tree_edge).T)
        data.edge_index = torch.cat((data.edge_index.flip(0), data.edge_index), dim=-1)
        data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=len(data.y))
        data.edge_index, _ = coalesce(data.edge_index, None, len(data.y), len(data.y))

        time_data = time() - time0

        time0 = time()
        prev_node, next_node = model.set_problem(**vars(data.to(device)), loop=10)
        node_loss = torch.nn.MSELoss()(next_node[next_explore_edge[0]], torch.FloatTensor(explored_nodes[next_explore_node]))
        edge_loss = torch.nn.CrossEntropyLoss()(prev_node.unsqueeze(0), torch.LongTensor([next_explore_edge[0]]))
        n_pos = data.y_free[data.edge_index[1]].float().sum()
        # free_loss = torch.nn.BCEWithLogitsLoss(pos_weight=(data.edge_index.shape[1]-n_pos)/(n_pos+1e-5))(node_free.view(-1), data.y_free[data.edge_index[1]].float().view(-1))
        loss = node_loss + edge_loss  # + free_loss
        loss.backward()
        losses.append((loss, node_loss, edge_loss))  # , free_loss))
        time_train = time() - time0

        time0 = time()
        if T % 8 == 0:
            optimizer.step()
            optimizer.zero_grad()

            writer.add_scalar('train/total_loss', sum([loss[0] for loss in losses]) / len(losses), T)
            writer.add_scalar('train/node_loss', sum([loss[1] for loss in losses]) / len(losses), T)
            writer.add_scalar('train/edge_loss', sum([loss[2] for loss in losses]) / len(losses), T)
            # writer.add_scalar('train/free_loss', sum([loss[3] for loss in losses]) / len(losses), T)
            # writer.add_scalar('train/avg_free', data.y_free[data.edge_index[1]].float().mean(), T)
            # tn = (node_free.view(-1).sigmoid().round()==data.y_free[data.edge_index[1]].float().view(-1))[
            #     ~data.y_free[data.edge_index[1]]].float().mean()
            # tp = (node_free.view(-1).sigmoid().round()==data.y_free[data.edge_index[1]].float().view(-1))[
            #     data.y_free[data.edge_index[1]]].float().mean()
            # if not torch.isnan(tn):
            #     writer.add_scalar('train/true_negative', tn, T)
            # if not torch.isnan(tp):
            #     writer.add_scalar('train/true_positive', tp, T)
            losses = []

            torch.save(model.state_dict(), 'weights.pt')

        T += 1
        time_bp = time() - time0

        pbar.set_description("data %.2fs, train %.2fs, bp %.2fs" % (time_data, time_train, time_bp))

    torch.save(model.state_dict(), 'weights_%s.pt' % str(iter_i))

torch.save(model.state_dict(), 'weights.pt')
writer.close()
