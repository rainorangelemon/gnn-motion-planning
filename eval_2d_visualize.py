import torch
import numpy as np
from environment import MazeEnv
from next_model import EncoderProcessDecoder
from torch_geometric.data import Data
from algorithm.bit_star import BITStar
from config import set_random_seed
from tqdm import tqdm as tqdm
from utils.plot import plot_edges
import pickle
from torch_sparse import coalesce
from torch_geometric.nn import knn_graph
from time import time

n_sample=500
k=30
loop=10
set_random_seed(1234)


def path(policy, index, goal_index, path_length):
    result = []
    policy = policy.data.cpu().numpy()
    i = 0
    while i < path_length:
        result.append(index)
        if index == goal_index:
            break
        assert sum(policy[index]) != 0
        index = policy[index].argmax()
        i += 1
    return result


def radius_init():
    bounds = env.bound
    bounds = np.array(bounds).reshape((2, -1)).T
    ranges = bounds[:, 1] - bounds[:, 0]
    eta = 1.1
    from scipy import special
    # Hypersphere radius calculation
    n = env.config_dim
    unit_ball_volume = np.pi ** (n / 2.0) / special.gamma(n / 2.0 + 1)
    volume = np.abs(np.prod(ranges))
    gamma = (1.0 + 1.0 / n) * volume / unit_ball_volume
    radius_constant = 2 * eta * (gamma ** (1.0 / n))
    return radius_constant


def to_np(tensor):
    return tensor.data.cpu().numpy()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
INFINITY = float('inf')

env = MazeEnv(dim=2)
model = EncoderProcessDecoder(workspace_size=2, config_size=2, embed_size=32, map_size=9261).to(device)
model.load_state_dict(torch.load('data/weights/weights_maze.pt', map_location=device))

solutions = []
model.eval()

pbar = tqdm(range(2000, 3000))
for index in pbar:

    pb = env.init_new_problem(index)
    c0 = env.collision_check_count
    t0 = time()

    forward = time() - t0

    success = False
    points = env.sample_n_points(n_sample)
    data = Data(x_obstacles=torch.cat((torch.FloatTensor(env.obstacles), torch.zeros(len(env.obstacles), 2)), dim=-1),
                x_goal=torch.FloatTensor(env.goal_state),
                y=torch.FloatTensor(points), )
    data.y = torch.cat((torch.FloatTensor(env.init_state).view(1, -1), data.y,
                        torch.FloatTensor(env.goal_state).view(1, -1)), dim=0)
    edge_index = knn_graph(torch.FloatTensor(data.y), k=k, loop=True)
    edge_index = torch.cat((edge_index, edge_index.flip(0)), dim=-1)
    data.edge_index, _ = coalesce(edge_index, None, len(data.y), len(data.y))

    while not success and len(data.y) < 10000:

        explored = [0]
        costs = {0: 0.}
        prev = {0: 0}
        value, policy, node_free, edge_free = model.set_problem(**vars(data.to(device)), loop=loop, need_softmax=False)
        policy[torch.arange(len(data.y)), torch.arange(len(data.y))] = 0
        success = False
        while policy[explored, :].sum() != 0:

            agent = policy[np.array(explored)[torch.where(policy[explored, :] != 0)[0]], torch.where(policy[explored, :] != 0)[1]].argmax()
            # print(policy[np.array(explored)[torch.where(policy[explored, :] != 0)[0]], torch.where(policy[explored, :] != 0)[1]][agent])

            end_a, end_b = torch.where(policy[explored, :] != 0)[0][agent], torch.where(policy[explored, :] != 0)[1][agent]
            end_a, end_b = int(end_a), int(end_b)
            end_a = explored[end_a]
            if env._edge_fp(to_np(data.y[end_a]), to_np(data.y[end_b])):
                explored.append(end_b)
                costs[end_b] = costs[end_a] + np.linalg.norm(to_np(data.y[end_a])-to_np(data.y[end_b]))
                prev[end_b] = end_a
                policy[:, end_b] = 0
                if env.in_goal_region(to_np(data.y[end_b])):
                    success = True
                    cost = costs[end_b]
                    # path = [end_b]
                    # node = end_b
                    # while node != 0:
                    #     path.append(prev[node])
                    #     node = prev[node]
                    # if not (value[path[:-1]] < value[path[1:]]).all():
                    #     print(value[path])
                    break
            else:
                policy[end_a, end_b] = 0
                policy[end_b, end_a] = 0

        if not success:
            # print('----------------------------------------resample----------------------------------------!')
            points = env.sample_n_points(len(data.y)-2+n_sample)
            data.y = torch.cat((data.y[[0], :], torch.FloatTensor(points), data.y[[-1], :]), dim=0)
            edge_index = knn_graph(torch.FloatTensor(data.y), k=k, loop=True)
            edge_index = torch.cat((edge_index, edge_index.flip(0)), dim=-1)
            data.edge_index, _ = coalesce(edge_index, None, len(data.y), len(data.y))

    solutions.append((success, to_np(value[explored].squeeze()), to_np(data.y[explored]),
                      prev, explored, env.collision_check_count-c0, time()-t0, cost))

    bit_solution = BITStar(env, batch_size=50, T=n_sample, sampling=None).plan(INFINITY, time_budget=300,
                                                                           refine_time_budget=0)

    if cost > bit_solution[-3] or (index==2001) or (index==2000):
        plot_edges(points, {tuple(to_np(data.y[node])): tuple(to_np(data.y[parent])) for node, parent in prev.items()},
                   env.get_problem(), title='GNN, Cost: %.2f, ' % cost)
        nodes, edges, collision, success, n_samples, _ = bit_solution
        plot_edges(set(nodes)|set(edges.keys()), edges, env.get_problem(), title='BIT*, Cost: %.2f, ' % bit_solution[-3])

    pbar.set_description("gnn %.2fs, search %.2fs" % (forward, time()-t0-forward))

n_success = sum([s[0] for s in solutions])
print('success rate:', n_success)
print('collision check: ', int(float(sum([s[-3] for s in solutions if s[0]])) / n_success))
print('running time: %.2f' % (float(sum([s[-2] for s in solutions if s[0]])) / n_success))
print('path cost: %.2f' % (float(sum([s[-1] for s in solutions if s[0]])) / n_success))
print('total time: %.2f' % sum([s[-2] for s in solutions]))


with open('%d_%d_%d_maze_gnn.pkl' % (n_sample, loop, k), 'wb') as f:
    pickle.dump(solutions, f, pickle.DEFAULT_PROTOCOL)

# TODO:  1. decide the best k number  2. decide the best looping number  3. reduce running time by 2
