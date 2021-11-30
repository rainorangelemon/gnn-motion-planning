import torch
import numpy as np
from environment import MazeEnv
from model_explore import Explorer
from config import config, set_random_seed
from tqdm import tqdm as tqdm
from torch_geometric.utils import add_self_loops
from algorithm.bit_star_tree import BITStar


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def tensor_to_np(tensor):
    return tensor.data.cpu().numpy()


def tuple_to_np(tuple_):
    return np.array(tuple_)


def init_tree(env):
    nodes = [tuple(env.init_state)]
    edges = {}
    ends = []
    return [nodes, edges, ends]


def create_data(tree, env):
    nodes, edges, ends = tree
    edges_idx = [(nodes.index(x), nodes.index(y)) for x, y in edges.items()]

    data = DotDict({
        'x_goal': torch.FloatTensor(env.goal_state),
        'y': torch.FloatTensor(nodes),
        'y_free': torch.LongTensor([True]*len(nodes)),
        'edge_index': torch.LongTensor(edges_idx).T,
    })

    data.edge_index = torch.cat((data.edge_index, data.edge_index.flip(0)), dim=-1)
    data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=len(data.y))

    return data


def policy(model, env, tree, beta):
    if np.random.uniform() <= beta:
        path = bit_solve(env, tree)[1]
        if len(path) >= 1:
            next_node = path[1]
            parent = path[0]
        else:
            data = create_data(tree, env)
            next_node = tensor_to_np(model(**data))
            parent = None
    else:
        data = create_data(tree, env)
        next_node = tensor_to_np(model(**data))
        parent = None
    return tuple(next_node), parent


def find_nearest_node(tree, next_node):
    nodes, edges, ends = tree
    distance = np.linalg.norm(np.array(nodes) - next_node, axis=-1)
    near = np.argmin(distance)
    return nodes[near]


def extend(env, tree, next_node, parent):
    nearest_node = find_nearest_node(tree, next_node) if parent is None else parent
    if env._edge_fp(tuple_to_np(nearest_node), tuple_to_np(next_node)):
        tree[0].append(next_node)
        tree[1][next_node] = nearest_node
    else:
        if env._state_fp(tuple_to_np(next_node)):
            tree[2].append(tuple(env.collision_point))
        else:
            tree[2].append(tuple(next_node))
    return tree


def bit_sample(env, tree, n_queue):
    init_state = env.init_state
    queue = []; i = 0
    for _ in range(n_queue):
        path_length, path = bit_solve(env, tree)
        env.init_state = init_state
        if path_length != float('inf'):
            queue.append((path_length, path[1]))
    return queue


def bit_solve(env, tree):
    BIT = BITStar(env, starts=tree[0])
    nodes, edges, collision, path_length, n_samples, _ = BIT.plan(float('inf'), refine_time_budget=0, time_budget=5)
    return path_length, BIT.get_best_path()


def train(replay, model, optimizer, env):
    if len(replay) <= 8:
        return

    optimizer.zero_grad()
    batch_idx = np.random.choice(len(replay), size=8, replace=False)
    for idx in batch_idx:
        tree, next_node = replay[idx]
        pred_node = model(**create_data(tree, env))
        loss = torch.nn.MSELoss()(torch.FloatTensor(next_node), pred_node)
        loss.backward()
    optimizer.step()


env = MazeEnv(dim=2)
set_random_seed(1234)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

epoch = 2000; iter = 2000; graph_size = 200; loop=30

model = Explorer(workspace_size=2, config_size=2, embed_size=32).to(device)
model.load_state_dict(torch.load('data/weights/weights_explore.pt', map_location=device))
T = 100; beta = 0.; success = 0
losses = []; replay = []
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
optimizer.zero_grad()

for index in tqdm(range(iter)):

    index = index % 2000
    pb = env.init_new_problem(index)

    tree = init_tree(env)
    for k in range(np.random.randint(1, T)):
        next_node, parent = policy(model, env, tree, beta)
        tree = extend(env, tree, next_node, parent)

        if env.in_goal_region(tuple_to_np(tree[0][-1])):
            break

    if env.in_goal_region(tuple_to_np(tree[0][-1])):
        success += 1

print(success)
print('success')
