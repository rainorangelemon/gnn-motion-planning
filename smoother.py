import numpy as np
from copy import deepcopy
from algorithm.bit_star import BITStar
from environment.maze_env import MazeEnv
from utils.plot import plot_edges
from config import set_random_seed
import torch
from torch_geometric.utils import add_self_loops
from algorithm.dijkstra import dijkstra
from collections import defaultdict
from environment.timer import Timer


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def tensor_to_np(tensor):
    return tensor.data.cpu().numpy()


def tuple_to_np(tuple_):
    return np.array(tuple_)


def tensor_to_tuple(tensor):
    return tuple(tensor_to_np(tensor))


def edge_cost(prev, next):
    return np.linalg.norm(tuple_to_np(next)-tuple_to_np(prev))


def path_cost(path):
    return sum([edge_cost(path[node_idx], path[node_idx+1]) for node_idx in range(len(path)-1)])


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def path_t(path):
    cost = path_cost(path)
    result = [0.]
    for prev, next in zip(path[:-1], path[1:]):
        result.append(result[-1]+edge_cost(prev, next)/cost)
    return result


def obs_data(env, free, collided):
    if not len(free):
        free.append([0. for _ in range(env.config_dim)])
    if not len(collided):
        collided.append([0. for _ in range(env.config_dim)])
    free = free[:500]
    collided = collided[:500] 
    data = DotDict({
        'free': torch.FloatTensor(np.array(free)).to(device),
        'collided': torch.FloatTensor(np.array(collided)).to(device),
        'obstacles': torch.FloatTensor(env.obstacles).to(device),
    })
    return data


def random_path_smoother(path, eps, env, iter=100):
    path = deepcopy(path)
    if len(path) > 2:
        for _ in range(iter):
            action = np.random.uniform(-eps, eps, size=env.config_dim)
            node_idx = np.random.randint(1, len(path)-1)
            prev_node = tuple_to_np(path[node_idx])
            new_node = path[node_idx]+action
            if env._state_fp(tuple_to_np(new_node)) and \
                    env._edge_fp(tuple_to_np(new_node), tuple_to_np(path[node_idx-1])) and \
                    env._edge_fp(tuple_to_np(new_node), tuple_to_np(path[node_idx+1])):
                if (np.linalg.norm(path[node_idx+1]-new_node) + np.linalg.norm(path[node_idx-1]-new_node)) < \
                        (np.linalg.norm(path[node_idx+1]-prev_node) + np.linalg.norm(path[node_idx-1]-prev_node)):
                    path[node_idx] = tuple(new_node)

    return path


def create_graph(path, env, prev, next):
    points = path[prev:(next+1)]
    neighbors = defaultdict(list)
    edge_cost = defaultdict(list)
    for p1 in points:
        for p2 in points:
            if env._edge_fp(tuple_to_np(p1), tuple_to_np(p2)):
                neighbors[p1].append(p2)
                edge_cost[p1].append(np.linalg.norm(tuple_to_np(p1)-tuple_to_np(p2)))
    return neighbors, edge_cost


def prune_path(path, env, iter=100):
    for _ in range(iter):
        try:
            len_path = len(path)
            crit_idx = []
            for index, point in enumerate(path):
                if index == 0 or index == (len(path)-1):
                    crit_idx.append(index)
                else:
                    if not env._edge_fp(tuple_to_np(path[index-1]), tuple_to_np(path[index+1])):
                        crit_idx.append(index)

            new_path = list()
            new_path.append(path[0])
            for prev, next in zip(crit_idx[:-1], crit_idx[1:]):
                neighbors, edge_cost = create_graph(path, env, prev, next)
                dists, prevs = dijkstra(path[prev:(next+1)], neighbors, edge_cost, path[prev])
                partial_path = []
                current = path[next]
                while current != path[prev]:
                    partial_path.append(current)
                    current = prevs[current]
                partial_path.reverse()
                new_path.extend(partial_path)
            path = new_path
            if len(path) == len_path:
                return path
        except Exception:
            break
    return path


def joint_smoother(path, env, iter, random_iter=100, prune_iter=100):
    for _ in range(iter):
        path = random_path_smoother(path, env.RRT_EPS, env, iter=random_iter)
        path = prune_path(path, env, iter=prune_iter)
    return path


def joint_smoother_ratio(path, env, iter=5, random_iter=100, prune_iter=100):
    for _ in range(iter):
        path = random_path_smoother(path, env.RRT_EPS, env, iter=random_iter)
        shorten_path = prune_path(path, env, iter=prune_iter)
        random_idx = prune_idx = 0
        while prune_idx != len(shorten_path):
            random_idx_next = random_idx
            while shorten_path[prune_idx] != path[random_idx_next]:
                random_idx_next += 1
            seg_A = np.array(path[random_idx])
            seg_B = np.array(path[random_idx_next])
            for inter_idx in range(random_idx+1, random_idx_next):
                path[inter_idx] = tuple((seg_B - seg_A) * (inter_idx - random_idx) / (random_idx_next - random_idx) + seg_A)
            prune_idx += 1
            random_idx = random_idx_next
    return path

    # joint_path = joint_smoother(path, env, iter, random_iter, prune_iter)
    # joint_path_t = path_t(joint_path)
    # orig_path_t = path_t(path)
    # smooth_path = []
    # for node_t in orig_path_t:
    #     right_idx = np.searchsorted(joint_path_t, node_t, side='right')
    #     if right_idx == len(joint_path_t):
    #         smooth_path.append(joint_path[-1])
    #     else:
    #         smooth_path.append(tuple(tuple_to_np(joint_path[right_idx-1])+
    #                        (tuple_to_np(joint_path[right_idx])-tuple_to_np(joint_path[right_idx-1]))*
    #                        (node_t-joint_path_t[right_idx-1])/(joint_path_t[right_idx]-joint_path_t[right_idx-1])))
    # return smooth_path


def proposed_path_smoother(old_path, new_path, env):
    # try moving in small direction
    
    path = deepcopy(old_path)
    proposes = list(np.arange(1, len(path)-1))
    valid = deepcopy(proposes)
    while len(valid):
        # use valid in the function
        node_idx = np.random.choice(np.array(valid))
        prev_node = tuple_to_np(path[node_idx])
        new_node = new_path[node_idx]
        if env._state_fp(tuple_to_np(new_node)) and \
                env._edge_fp(tuple_to_np(new_node), tuple_to_np(path[node_idx - 1])) and \
                env._edge_fp(tuple_to_np(new_node), tuple_to_np(path[node_idx + 1])):
            if (np.linalg.norm(path[node_idx + 1] - new_node) + np.linalg.norm(path[node_idx - 1] - new_node)) < \
                    (np.linalg.norm(path[node_idx + 1] - prev_node) + np.linalg.norm(path[node_idx - 1] - prev_node)):
                path[node_idx] = tuple(new_node)
                proposes.remove(node_idx)
                valid = deepcopy(proposes)
            else:
                valid.remove(node_idx)
        else:
            valid.remove(node_idx)
    return path


def proposed_path_smootherv2(old_path, new_path, env):
    K = int(np.ceil((np.linalg.norm(np.array(old_path) - np.array(new_path), axis=-1) / env.RRT_EPS).max()))
    path = deepcopy(old_path)
    for _ in range(K):
        diff = 0
        next_path = deepcopy(path)
        # steer
        for i, ns in enumerate(zip(path[1:-1], new_path[1:-1])):
            i = i+1
            old_n, new_n = ns
            dist = np.linalg.norm(old_n - new_n)
            if dist < env.RRT_EPS:
                next_path[i] = new_n
            else:
                next_path[i] = env.interpolate(old_n, new_n, env.RRT_EPS / dist)
            if not env._edge_fp(next_path[i-1], next_path[i]):
                next_path[i] = path[i]
            else:
                diff += np.linalg.norm(next_path[i]-new_n)
        path = next_path
        if diff < 1e-5:
            return path
    return path
        
        
def interpolate_path(env, path, eps=None):
    if eps is None:
        eps = env.RRT_EPS
    path = np.array(path)
    new_path = []
    for n1, n2 in zip(path[:-1], path[1:]):
        dist = np.linalg.norm(n2 - n1)
        K = int(np.ceil(dist / eps))
        for k in range(K):
            new_path.append(n1 + (n2 - n1) * k / K )
    new_path.append(path[-1])
    return new_path
    

def model_smooth(model, free, collided, old_path, env, iter=5): 
    
    for iter_i in range(iter):
        data = obs_data(env, free, collided)
        data.path = torch.FloatTensor(np.array(old_path)).to(device)
        data.edge_index = torch.cat((torch.arange(1, len(old_path)).reshape(1, -1),
                                     torch.arange(0, len(old_path) - 1).reshape(1, -1)), dim=0)
        data.edge_index = torch.cat((data.edge_index, data.edge_index.flip(0)), dim=-1)
        data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=len(data.path))
        data.edge_index = data.edge_index.to(device)
        new_path = model(**data, loop=1).data.cpu().numpy()  
        old_path = proposed_path_smootherv2(old_path, new_path, env)
    
    return old_path


if __name__ == "__main__":
    set_random_seed(1234)
    env = MazeEnv(dim=2)
    for index in range(2010, 2012):
        env.init_new_problem(index)
        BIT = BITStar(env)
        nodes, edges, collision, path_length, n_samples, _ = BIT.plan(float('inf'), refine_time_budget=0, time_budget=5)
        path = BIT.get_best_path()
        plot_edges(states=path, edges={path[i]: path[i+1] for i in range(len(path)-1)}, problem=env.get_problem())
        path = random_path_smoother(BIT.get_best_path(), env.RRT_EPS, env)
        plot_edges(states=path, edges={path[i]: path[i+1] for i in range(len(path)-1)}, problem=env.get_problem())
        path = prune_path(BIT.get_best_path(), env)
        plot_edges(states=path, edges={path[i]: path[i+1] for i in range(len(path)-1)}, problem=env.get_problem())
        path = joint_smoother(BIT.get_best_path(), env, iter=5)
        plot_edges(states=path, edges={path[i]: path[i+1] for i in range(len(path)-1)}, problem=env.get_problem())
        path = joint_smoother_ratio(BIT.get_best_path(), env, iter=5)
        plot_edges(states=path, edges={path[i]: path[i + 1] for i in range(len(path) - 1)}, problem=env.get_problem())

