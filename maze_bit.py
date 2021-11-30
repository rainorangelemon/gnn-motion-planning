import numpy as np
from config import set_random_seed
import torch
from environment import KukaEnv
from torch_geometric.nn import knn_graph
from collections import defaultdict
from time import time
import pickle
from tqdm import tqdm
from torch_sparse import coalesce
from algorithm.bit_star import BITStar
from next_model import EncoderProcessDecoder
from torch_geometric.data import Data
from environment import MazeEnv

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


if __name__ == "__main__":

    data = []
    env = MazeEnv(dim=2)

    set_random_seed(4123)

    time0 = time()
    solutions = []
    n_sample = 1000

    for problem_index in tqdm(range(2000, 3000)):

        env.init_new_problem(problem_index)
        solution = BITStar(env, batch_size=50, T=n_sample, sampling=None).plan(INFINITY, time_budget=300, refine_time_budget=1)
        solutions.append(solution)

    with open('%s_maze_bit_no_refine.pkl' % (str(n_sample)), 'wb') as f:
        pickle.dump(solutions, f, pickle.DEFAULT_PROTOCOL)

    n_success = sum([s[-3] != INFINITY for s in solutions])
    print('success rate:', n_success)
    print('collision check: ', int(float(sum([s[2] for s in solutions if s[-3]!=INFINITY])) / n_success))
    print('running time: %.2f' % (float(sum([s[-1] for s in solutions if s[-3]!=INFINITY])) / n_success))
    print('path cost: %.2f' % (float(sum([s[-3] for s in solutions if s[-3]!=INFINITY])) / n_success))
    print('total time: %.2f' % sum([s[-1] for s in solutions]))

    print('hello')

