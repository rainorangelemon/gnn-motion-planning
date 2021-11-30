import numpy as np
from config import set_random_seed
import torch
from environment import KukaEnv, Kuka2Env
from torch_geometric.nn import knn_graph
from collections import defaultdict
from time import time
import pickle
from tqdm import tqdm
from torch_sparse import coalesce
from algorithm.bit_star import BITStar
from torch_geometric.data import Data
from algorithm.lazy_sp import LazySP
from eval_gnn import path_cost

INFINITY = float('inf')


def construct_graph(env, points, check_collision=True):
    edge_index = knn_graph(torch.FloatTensor(points), k=6, loop=True)
    edge_index = torch.cat((edge_index, edge_index.flip(0)), dim=-1)
    edge_index_torch, _ = coalesce(edge_index, None, len(points), len(points))
    edge_index = edge_index_torch.data.cpu().numpy().T
    edge_cost = defaultdict(list)
    edge_free = []
    neighbors = defaultdict(list)
    for i, edge in enumerate(edge_index):
        if env._edge_fp(points[edge[0]], points[edge[1]]):
            edge_cost[edge[1]].append(np.linalg.norm(points[edge[1]]-points[edge[0]]))
            edge_free.append(True)
        else:
            edge_cost[edge[1]].append(INFINITY)
            edge_free.append(False)
        neighbors[edge[1]].append(edge[0])
    return edge_cost, neighbors, edge_index, edge_free


def min_dist(q, dist):
    """
    Returns the node with the smallest distance in q.
    Implemented to keep the main algorithm clean.
    """
    min_node = None
    for node in q:
        if min_node is None:
            min_node = node
        elif dist[node] < dist[min_node]:
            min_node = node

    return min_node


def dijkstra(nodes, edges, costs, source):
    q = set()
    dist = {}
    prev = {}

    for v in nodes:       # initialization
        dist[v] = INFINITY      # unknown distance from source to v
        prev[v] = INFINITY      # previous node in optimal path from source
        q.add(v)                # all nodes initially in q (unvisited nodes)

    # distance from source to source
    dist[source] = 0

    while q:
        # node with the least distance selected first
        u = min_dist(q, dist)

        q.remove(u)

        for index, v in enumerate(edges[u]):
            alt = dist[u] + costs[u][index]
            if alt < dist[v]:
                # a shorter path to v has been found
                dist[v] = alt
                prev[v] = u

    return dist, prev


def eval_bit(str, seed, env, indexes, use_tqdm=False, batch=50, t_max=1000, **kwargs):

    set_random_seed(seed)

    time0 = time()
    solutions = []
    paths = []

    pbar = tqdm(indexes) if use_tqdm else indexes
    for problem_index in pbar:

        env.init_new_problem(problem_index)
        bit = BITStar(env, batch_size=batch, T=t_max, sampling=None)
        solution = bit.plan(INFINITY, time_budget=300, refine_time_budget=0)
        solutions.append((solution))
        paths.append(bit.get_best_path())

    # with open('%s_kuka_bit_no_refine.pkl' % (str(n_sample)), 'wb') as f:
    #     pickle.dump(solutions, f, pickle.DEFAULT_PROTOCOL)

    n_success = sum([s[-3] != INFINITY for s in solutions])
    collision = np.mean([s[2] for s in solutions])
    running_time = np.mean([s[-1] for s in solutions if s[-3]!=INFINITY])
    solution_cost = float(sum([s[-3] for s in solutions if s[-3]!=INFINITY])) / n_success
    total_time = sum([s[-1] for s in solutions])

    print('success rate: %d' % n_success)
    print('collision check: %.2f' % collision)
    print('running time: %.2f' % running_time)
    print('path cost: %.2f' % solution_cost)
    print('total time: %.2f' % total_time)
    print('')

    return n_success, collision, running_time, solution_cost, total_time, paths


def eval_lazysp(str, seed, env, indexes, use_tqdm=False, batch=50, t_max=1000, **kwargs):

    set_random_seed(seed)

    time0 = time()
    solutions = []
    paths = []

    pbar = tqdm(indexes) if use_tqdm else indexes
    for problem_index in pbar:

        env.init_new_problem(problem_index)
        lazy_sp = LazySP(env, batch_size=batch, T=t_max)
        solution = lazy_sp.plan()
        solutions.append((solution))
        paths.append(solution[2])

    # with open('%s_kuka_bit_no_refine.pkl' % (str(n_sample)), 'wb') as f:
    #     pickle.dump(solutions, f, pickle.DEFAULT_PROTOCOL)

    n_success = sum([len(p) != 0 for p in paths])
    collision = np.mean([s[1] for s in solutions])
    running_time = np.mean([s[4] for s in solutions if len(s[2])!=0])
    solution_cost = float(sum([path_cost(p) for p in paths if len(p)!=0])) / n_success
    total_time = sum([s[4] for s in solutions])

    print('success rate: %d' % n_success)
    print('collision check: %.2f' % collision)
    print('running time: %.2f' % running_time)
    print('path cost: %.2f' % solution_cost)
    print('total time: %.2f' % total_time)
    print('')

    return n_success, collision, running_time, solution_cost, total_time, paths

