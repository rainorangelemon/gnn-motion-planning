import torch
import numpy as np
from environment import KukaEnv
from next_model import EncoderProcessDecoder
from torch_geometric.data import Data
from config import set_random_seed
from tqdm import tqdm as tqdm
import pickle
from torch_sparse import coalesce
from torch_geometric.nn import knn_graph
from time import time
from collections import defaultdict

n_sample=500
k=30
loop=10
set_random_seed(3412)


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

env = KukaEnv()
model = EncoderProcessDecoder(workspace_size=3, config_size=7, embed_size=32, map_size=9261).to(device)
model.load_state_dict(torch.load('weights_goal.pt', map_location=device))

solutions = []
model.eval()

pbar = tqdm(range(2000, 3000))
for index in pbar:

    pb = env.init_new_problem(index)
    c0 = env.collision_check_count
    t0 = time()

    forward = time() - t0

    points = env.sample_n_points(n_sample)
    data = Data(x_obstacles=torch.FloatTensor(env.obstacles),
                x_goal=torch.FloatTensor(env.goal_state),
                y=torch.FloatTensor(points), )
    data.y = torch.cat((torch.FloatTensor(env.init_state).view(1, -1), data.y,
                        torch.FloatTensor(env.goal_state).view(1, -1)), dim=0)
    edge_index = knn_graph(torch.FloatTensor(data.y), k=k, loop=True)
    edge_index = torch.cat((edge_index, edge_index.flip(0)), dim=-1)
    data.edge_index, _ = coalesce(edge_index, None, len(data.y), len(data.y))

    if env.in_goal_region(to_np(data.y[0])):
        success = True
    else:
        success = False

    while not success and len(data.y) < 10000:

        explored = torch.LongTensor([0])
        costs = defaultdict(lambda: float('inf'))
        costs[0] = 0.
        cost = 0.
        prev = {0: 0}

        value, policy, node_free, edge_free = model.set_problem(**vars(data.to(device)), loop=loop, need_softmax=False)
        policy[torch.arange(len(data.y)), torch.arange(len(data.y))] = 0
        success = False
        while policy[explored, :].sum() != 0:

            good_next_nodes = (policy[explored, :]-float('inf')*(policy[explored, :]==0)).argmax(dim=-1)
            good_next_nodes = good_next_nodes[policy[explored, good_next_nodes]!=0]
            best_next_node = good_next_nodes[value[good_next_nodes].squeeze().argmin()]
            good_prev_nodes = explored[policy[explored, best_next_node]!=0]
            best_prev_node = good_prev_nodes[policy[good_prev_nodes, best_next_node].argmax()]

            end_a, end_b = int(best_prev_node), int(best_next_node)
            if env._edge_fp(to_np(data.y[end_a]), to_np(data.y[end_b])):
                explored = torch.cat((explored, torch.LongTensor([end_b])), dim=0)
                if costs[end_b] > costs[end_a] + np.linalg.norm(to_np(data.y[end_a])-to_np(data.y[end_b])):
                    prev[end_b] = end_a
                    costs[end_b] = costs[end_a] + np.linalg.norm(to_np(data.y[end_a])-to_np(data.y[end_b]))
                policy[:, explored] = 0
                if env.in_goal_region(to_np(data.y[end_b])):
                    success = True
                    cost = costs[end_b]
                    break
            else:
                policy[end_a, end_b] = 0
                policy[end_b, end_a] = 0

        if not success:
            points = env.sample_n_points(len(data.y)-2+n_sample)
            data.y = torch.cat((data.y[[0], :], torch.FloatTensor(points), data.y[[-1], :]), dim=0)
            edge_index = knn_graph(torch.FloatTensor(data.y), k=k, loop=True)
            edge_index = torch.cat((edge_index, edge_index.flip(0)), dim=-1)
            data.edge_index, _ = coalesce(edge_index, None, len(data.y), len(data.y))

    solutions.append((success, to_np(value[explored].squeeze()), to_np(data.y[explored]),
                      to_np(explored), env.collision_check_count-c0, time()-t0, cost))

    pbar.set_description("gnn %.2fs, search %.2fs" % (forward, time()-t0-forward))

n_success = sum([s[0] for s in solutions])
print('success rate:', n_success)
print('collision check: ', int(float(sum([s[-3] for s in solutions if s[0]])) / n_success))
print('running time: %.2f' % (float(sum([s[-2] for s in solutions if s[0]])) / n_success))
print('path cost: %.2f' % (float(sum([s[-1] for s in solutions if s[0]])) / n_success))
print('total time: %.2f' % sum([s[-2] for s in solutions]))


with open('%d_%d_%d_kuka_gnn.pkl' % (n_sample, loop, k), 'wb') as f:
    pickle.dump(solutions, f, pickle.DEFAULT_PROTOCOL)

# TODO:  1. decide the best k number  2. decide the best looping number  3. reduce running time by 2
