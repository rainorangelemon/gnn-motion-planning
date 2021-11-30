import torch
import numpy as np
from torch_geometric.data import Data
from config import set_random_seed
from tqdm import tqdm as tqdm
from torch_sparse import coalesce
from torch_geometric.nn import knn_graph
from time import time
from smoother import model_smooth, proposed_path_smoother, joint_smoother, interpolate_path
# from model_smoother2 import ModelSmoother
from str2name import str2name

loop = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def obs_data(env, free, collided):
    # if not len(free):
    #     free.append([0. for _ in range(env.config_dim)])
    # if not len(collided):
    #     collided.append([0. for _ in range(env.config_dim)])

    data = DotDict({
        'free': torch.FloatTensor(free).to(device),
        'collided': torch.FloatTensor(collided)[:len(free)].to(device),
        'obstacles': torch.FloatTensor(env.obstacles).to(device),
    })
    return data


# def path(policy, index, goal_index, path_length):
#     result = []
#     policy = policy.data.cpu().numpy()
#     i = 0
#     while i < path_length:
#         result.append(index)
#         if index == goal_index:
#             break
#         assert sum(policy[index]) != 0
#         index = policy[index].argmax()
#         i += 1
#     return result


def path_cost(path):
    path = np.array(path)
    cost = 0
    for i in range(0, len(path) - 1):
        cost += np.linalg.norm(path[i + 1] - path[i])
    return cost


# def radius(n_sample):
#     bounds = env.bound
#     bounds = np.array(bounds).reshape((2, -1)).T
#     ranges = bounds[:, 1] - bounds[:, 0]
#     eta = 1.1
#     from scipy import special
#     # Hypersphere radius calculation
#     n = env.config_dim
#     unit_ball_volume = np.pi ** (n / 2.0) / special.gamma(n / 2.0 + 1)
#     volume = np.abs(np.prod(ranges))
#     gamma = (1.0 + 1.0 / n) * volume / unit_ball_volume
#     radius_constant = 2 * eta * (gamma ** (1.0 / n))
#     return radius_constant * ((math.log(n_sample) / n_sample) ** (1.0 / n))


def to_np(tensor):
    return tensor.data.cpu().numpy()


def eval_gnn_pure(str, seed, env, indexes, model=None, model_s=None, use_tqdm=False, smooth=True, batch=500, t_max=500,
                  k=30, **kwargs):
    embed_size = 32;
    set_random_seed(seed)
    INFINITY = float('inf')
    if model is None:
        _, model, model_path, model_s, model_s_path = str2name(str, use_obstacle=False)
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.use_obstacles = False
    if model_s is None:
        _, model, mode_path, model_s, model_s_path = str2name(str)
        model_s.load_state_dict(torch.load(model_s_path, map_location=torch.device("cpu")))

    return eval_gnn(str, seed, env, indexes, model, model_s, use_tqdm, smooth, batch, t_max, k, **kwargs)


def eval_gnn(str, seed, env, indexes, model=None, model_s=None, use_tqdm=False, smooth=True, batch=500, t_max=500, k=30,
             **kwargs):
    set_random_seed(seed)
    if model is None:
        _, model, model_path, _, _ = str2name(str)
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    if model_s is None:
        _, _, _, model_s, model_s_path = str2name(str)
        model_s.load_state_dict(torch.load(model_s_path, map_location=torch.device("cpu")))

    solutions = []
    paths = []
    smooth_paths = []
    model.eval()
    model_s.eval()

    pbar = tqdm(indexes) if use_tqdm else indexes
    for index in pbar:

        env.init_new_problem(index)
        result = explore(env, model, model_s, smooth, batch=batch, t_max=t_max, k=k, **kwargs)

        paths.append(result['path'])
        smooth_paths.append(result['smooth_path'])
        solutions.append(
            (result['success'], path_cost(result['path']), path_cost(result['smooth_path']),
             result['c_explore'], result['c_smooth'], result['total'], result['total_explore']))

        if use_tqdm:
            pbar.set_description("gnn %.2fs, search %.2fs, explored %d" %
                                 (result['forward'], result['total'] - result['forward'], len(result['explored'])))

    n_success = sum([s[0] for s in solutions])
    collision_explore = np.mean([s[3] for s in solutions])
    collision = np.mean([(s[3] + s[4]) for s in solutions])
    running_time = float(sum([s[5] for s in solutions if s[0]])) / n_success
    solution_cost = float(sum([(s[2]) for s in solutions if s[0]])) / n_success
    total_time = sum([s[5] for s in solutions])
    total_time_explore = sum([s[6] for s in solutions])

    print('success rate:', n_success)
    print('collision check: %.2f' % collision)
    print('collision check explore: %.2f' % collision_explore)
    print('running time: %.2f' % running_time)
    print('path cost: %.2f' % solution_cost)
    print('total time: %.2f' % total_time)
    print('total time explore: %.2f' % total_time_explore)
    print('')

    return n_success, collision, running_time, solution_cost, total_time, paths, smooth_paths, collision_explore, total_time_explore

    # TODO:  1. decide the best k number  2. decide the best looping number  3. reduce running time by 2


def create_data(free, collided, env, k):
    data = Data(goal=torch.FloatTensor(env.goal_state))
    data.v = torch.cat((torch.FloatTensor(free),
                        torch.FloatTensor(collided)), dim=0)
    # create labels
    data.labels = torch.zeros(len(data.v), 3)
    data.labels[:len(free), 0] = 1
    data.labels[len(free):, 1] = 1
    data.labels[1, 2] = 1
    k1 = int(np.ceil(k * np.log(len(free)) / np.log(100)))
    edge_index = knn_graph(torch.FloatTensor(data.v), k=k1, loop=True)
    edge_index = torch.cat((edge_index, edge_index.flip(0)), dim=-1)
    edge_index_free = knn_graph(torch.FloatTensor(data.v[:len(free)]), k=k1, loop=True)
    edge_index = torch.cat((edge_index, edge_index_free, edge_index_free.flip(0)), dim=-1)
    data.edge_index, _ = coalesce(edge_index, None, len(data.v), len(data.v))
    return data


@torch.no_grad()
def explore(env, model, model_s, smooth=True, batch=500, t_max=1000, k=30, smoother='model', loop=5):
    c0 = env.collision_check_count
    t0 = time()
    forward = 0
    success = False
    path, smooth_path = [], []
    n_batch = batch
    #     n_batch = min(batch, t_max)
    free, collided = env.sample_n_points(n_batch, need_negative=True)
    collided = collided[:len(free)]
    free = [env.init_state] + [env.goal_state] + list(free)

    explored = [0]
    explored_edges = [[0, 0]]
    costs = {0: 0.}
    prev = {0: 0}

    data = create_data(free, collided, env, k)

    # data.edge_index = radius_graph(data.v, radius(len(data.v)), loop=True)
    while not success and (len(free) - 2) <= t_max:

        t1 = time()
        policy = model(**data.to(device).to_dict(), **obs_data(env, free, collided), loop=loop)
        policy = policy.cpu()
        forward += time() - t1

        policy[torch.arange(len(data.v)), torch.arange(len(data.v))] = 0
        policy[:, explored] = 0
        policy[:, data.labels[:, 1] == 1] = 0
        policy[data.labels[:, 1] == 1, :] = 0
        policy[np.array(explored_edges).reshape(2, -1)] = 0
        success = False
        while policy[explored, :].sum() != 0:

            agent = policy[
                np.array(explored)[torch.where(policy[explored, :] != 0)[0]], torch.where(policy[explored, :] != 0)[
                    1]].argmax()

            end_a, end_b = torch.where(policy[explored, :] != 0)[0][agent], torch.where(policy[explored, :] != 0)[1][
                agent]
            end_a, end_b = int(end_a), int(end_b)
            end_a = explored[end_a]
            explored_edges.extend([[end_a, end_b], [end_b, end_a]])
            if env._edge_fp(to_np(data.v[end_a]), to_np(data.v[end_b])):
                explored.append(end_b)
                costs[end_b] = costs[end_a] + np.linalg.norm(to_np(data.v[end_a]) - to_np(data.v[end_b]))
                prev[end_b] = end_a

                policy[:, end_b] = 0
                if env.in_goal_region(to_np(data.v[end_b])):
                    print(env.collision_check_count - c0)
                    success = True
                    cost = costs[end_b]
                    path = [end_b]
                    node = end_b
                    while node != 0:
                        path.append(prev[node])
                        node = prev[node]
                    path.reverse()
                    break
            else:
                policy[end_a, end_b] = 0
                policy[end_b, end_a] = 0

        if not success:
            if not smooth:
                return []

            if (n_batch + len(free) - 2) > t_max:
                break
            # ----------------------------------------resample----------------------------------------
            new_free, new_collided = env.sample_n_points(n_batch, need_negative=True)
            free = free + list(new_free)
            collided = collided + list(new_collided)
            collided = collided[:len(free)]

            data = create_data(free, collided, env, k)

    c_explore = env.collision_check_count - c0
    c1 = env.collision_check_count
    t1 = time()
    if success and smooth:
        path = list(data.v[path].data.cpu().numpy())
        if smoother == 'model':
            smooth_path = model_smooth(model_s, free, collided, path, env)
        elif smoother == 'oracle':
            smooth_path = joint_smoother(path, env, iter=5)
        else:
            smooth_path = path
    c_smooth = env.collision_check_count - c1
    if smooth:
        total_time = time()
        return {'c_explore': c_explore,
                'c_smooth': c_smooth,
                'data': data,
                'explored': explored,
                'forward': forward,
                'total': total_time - t0,
                'total_explore': t1 - t0,
                'success': success,
                't0': t0,
                'path': path,
                'smooth_path': smooth_path,
                'explored_edges': explored_edges}
    else:
        return list(data.v[path].data.cpu().numpy()), free, collided


if __name__ == '__main__':
    from environment import SnakeEnv
    import pybullet as p
    from time import sleep

    env = SnakeEnv(GUI=False)
    env.init_new_problem(2000)
    # for _ in range(100):
    #     env.set_config(env.init_state)
    #     p.stepSimulation()
    #     sleep(0.1)
    #     env.set_config(env.goal_state)
    #     p.stepSimulation()
    #     sleep(0.1)

    result = eval_gnn(str(env), 1234, env, np.arange(2000, 2005), model=None, model_s=None, use_tqdm=True, smooth=True,
                      batch=50, t_max=1000)
    print('hello')
