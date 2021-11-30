import torch
import numpy as np
from environment import KukaEnv
from model_explore import Explorer
from torch_geometric.data import Data
from config import set_random_seed
from tqdm import tqdm as tqdm
import pickle
from time import time

n_sample=400
eta = 1.1
loop=10


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
set_random_seed(1234)
model = Explorer(workspace_size=3, config_size=7, embed_size=32, map_size=9261).to(device)
model.load_state_dict(torch.load('weights.pt', map_location=device))

solutions = []
model.eval()

pbar = tqdm(range(2000, 3000))
for index in pbar:

    pb = env.init_new_problem(index)
    c0 = env.collision_check_count
    t0 = time()

    forward = time() - t0

    success = False
    data = Data(x_obstacles=torch.FloatTensor(env.obstacles),
                x_goal=torch.FloatTensor(env.goal_state),
                y=torch.FloatTensor([env.init_state]),
                y_free=torch.BoolTensor([True]))
    data.edge_index = torch.LongTensor([[0], [0]])
    cost = [0.]
    while (not success) and (len(data.y) < 500):

        prev_nodes, next_nodes = model.set_problem(**vars(data.to(device)), loop=loop, need_softmax=False)
        prev_node_id = np.random.choice(len(data.y), p=to_np(prev_nodes.view(-1)))
        next_node = to_np(next_nodes[prev_node_id].view(-1))

        if env._edge_fp(to_np(data.y[prev_node_id].view(-1)), next_node):
            data.y = torch.cat((data.y, torch.FloatTensor([next_node])), dim=0)
            data.y_free = torch.cat((data.y_free, torch.BoolTensor([True])), dim=0)
            data.edge_index = torch.cat((data.edge_index,
                                         torch.LongTensor([[prev_node_id, len(data.y)-1, len(data.y)-1],
                                                           [len(data.y)-1, prev_node_id, len(data.y)-1]])), dim=-1)
            cost.append(cost[prev_node_id] + np.linalg.norm(next_node-to_np(data.y[prev_node_id].view(-1))))
            if env.in_goal_region(next_node):
                success = True
                break
        else:
            data.y = torch.cat((data.y, torch.FloatTensor([env.collision_point])), dim=0)
            data.y_free = torch.cat((data.y_free, torch.BoolTensor([False])), dim=0)
            data.edge_index = torch.cat((data.edge_index,
                                         torch.LongTensor([[prev_node_id, len(data.y)-1, len(data.y)-1],
                                                           [len(data.y)-1, prev_node_id, len(data.y)-1]])), dim=-1)
            cost.append(INFINITY)

    solutions.append((success, to_np(data.y), to_np(data.edge_index), env.collision_check_count-c0, time()-t0, cost[-1]))

    pbar.set_description("gnn %.2fs, search %.2fs" % (forward, time()-t0-forward))

n_success = sum([s[-1]!=INFINITY for s in solutions])
print('success rate:', n_success)
print('collision check: ', int(float(sum([s[-3] for s in solutions if s[-1]!=INFINITY])) / n_success))
print('running time: %.2f' % (float(sum([s[-2] for s in solutions if s[-1]!=INFINITY])) / n_success))
print('path cost: %.2f' % (float(sum([s[-1] for s in solutions if s[-1]!=INFINITY])) / n_success))
print('total time: %.2f' % sum([s[-2] for s in solutions]))


with open('%d_%d_%s_kuka_gnn.pkl' % (n_sample, loop, str(eta)), 'wb') as f:
    pickle.dump(solutions, f, pickle.DEFAULT_PROTOCOL)

# TODO:  1. decide the best k number  2. decide the best looping number  3. reduce running time by 2
