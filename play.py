import pickle
import numpy as np
from copy import deepcopy

envs = ['maze2easy', 'maze2hard', 'ur5', 'snake7', 'kuka7', 'kuka13', 'kuka14']
metrics = ['success rate', 'collision check', 'running time', 'path cost', 'total time']
titles = ['Success Rate', 'Collision Check', 'Running Time', 'Path Cost', 'Total Time']
metric_values = {}
method_names = ['GNN_pure', 'GNN', 'BIT*', 'NEXT', 'RRT*', 'LazySP*']

# a_file = open("data/results/result_b.txt", "r")
# lines = a_file.readlines()
# a_file.close()
# new_file = open("data/results/result.txt", "w")
#
# for line in lines:
#     if "1000/1000" not in line:
#         new_file.write(line)
#
# new_file.close()


def path_cost(path):
    path = np.array(path)
    cost = 0
    for i in range(0, len(path)-1):
        cost += np.linalg.norm(path[i+1]-path[i])
    return cost


# a = pickle.load(open("data/new_1.p", "rb"))
# for key in list(a.keys()):
#     if np.isnan(a[key][0]):
#         del a[key]
#
# b = pickle.load(open("data/new_2.p", "rb"))
# print(b.keys())
#
# c = pickle.load(open("data/snake.p", "rb"))
# print(c.keys())
#
# d = pickle.load(open("data/pure.p", "rb"))
# print(d.keys())
#
# e = pickle.load(open("data/ur5.p", "rb"))
# print(e.keys())
#
# rd = {**a, **b, **c, **d, **e}
#
# f = pickle.load(open("data/kuka14.p", "rb"))
# f = {('Kuka_14D', k[1], k[2]): v for k, v in f.items()}
# print(f.keys())
# for seed in ['1234', '2341', '3412', '4123']:
#     rd['Kuka_14D', 'GNN_pure', seed] = rd['Kuka_14D', 'GNN', seed] = f['Kuka_14D', 'GNN', '1234']
# costs = [path_cost(p) for p in f['Kuka_14D', 'GNN', '1234'][-1]]
# print(np.mean(np.array(costs)))
# costs = [path_cost(p) for p in rd['Kuka_14D', 'GNN', '1234'][-1]]
# print(np.mean(np.array(costs)))

a = pickle.load(open("data/new_next.p", "rb"))
print(a.keys())

b = pickle.load(open("data/gnn_bit_rrt_new_env.p", "rb"))
print(b.keys())

c = pickle.load(open("data/lazy_sp.p", "rb"))
print(c.keys())

rd = {**a, **b, **c}

for env in envs:
    for seed in ['1234', '2341', '3412', '4123']:
        valid_path = [np.all([len(rd[env, method, seed][5][i]) for method in method_names]) for i in range(1000)]
        for method in method_names:
            costs = [path_cost(p) for p in rd[env, method, seed][5]]
            if 'GNN' in method:
                costs = [path_cost(p) for p in rd[env, method, seed][6]]
            rd[env, method, seed] = list(rd[env, method, seed])
            rd[env, method, seed][3] = np.mean(np.array(costs)[valid_path])
            # print(env, method, seed, np.mean(np.array(costs)[valid_path]))
            if 'GNN' in method:
                no_smoother = method+"_ns"
                rd[env, no_smoother, seed] = deepcopy(rd[env, method, seed])
                costs = [path_cost(p) for p in rd[env, no_smoother, seed][5]]
                rd[env, no_smoother, seed] = list(rd[env, no_smoother, seed])
                rd[env, no_smoother, seed][1] = rd[env, no_smoother, seed][7]
                rd[env, no_smoother, seed][4] = rd[env, no_smoother, seed][8]
                rd[env, no_smoother, seed][3] = np.mean(np.array(costs)[valid_path])
                # print(env, no_smoother, seed, np.mean(np.array(costs)[valid_path]))

    for method in ['GNN_pure_ns', 'GNN_ns'] + method_names:
        rd[env, method, 'Avg'] = tuple(
            [np.mean([rd[env, method, seed][i] for seed in ['1234', '2341', '3412', '4123']]) for i in range(5)])
        print(env, method, 'Avg')
        print('success rate:', rd[env, method, 'Avg'][0])
        print('collision check: %.2f' % rd[env, method, 'Avg'][1])
        print('running time: %.2f' % rd[env, method, 'Avg'][2])
        print('path cost: %.2f' % rd[env, method, 'Avg'][3])
        print('total time: %.2f' % rd[env, method, 'Avg'][4])
        print('')


for env in envs:
    for method in ['GNN_pure_ns', 'GNN_ns'] + method_names:
        rd[env, method, 'Std'] = tuple(
            [np.std([rd[env, method, seed][i] for seed in ['1234', '2341', '3412', '4123']]) for i in range(5)])
        print(env, method, 'Std')
        print('success rate: %.2f' % rd[env, method, 'Std'][0])
        print('collision check: %.2f' % rd[env, method, 'Std'][1])
        print('running time: %.2f' % rd[env, method, 'Std'][2])
        print('path cost: %.2f' % rd[env, method, 'Std'][3])
        print('total time: %.2f' % rd[env, method, 'Std'][4])
        print('')


# from utils.plot import plot_edges
# from config import set_random_seed
# from environment import MazeEnv
# from tqdm import tqdm
# import numpy as np
# import time
# from algorithm.bit_star import BITStar
#
# solutions = []
#
# environment = MazeEnv(dim=2)
#
#
# def sample_empty_points(env):
#     while True:
#         point = np.random.uniform(-1, 1, 2)
#         if env._state_fp(point):
#             return point
#
#
# for _ in tqdm(range(3000)):
#     pb = environment.init_new_problem()
#     set_random_seed(1234)
#
#     cur_time = time.time()
#
#     BIT = BITStar(environment)
#     _ = BIT.plan(float('inf'))
#
#
# print('hello')
