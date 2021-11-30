import numpy as np
import torch
from environment import KukaEnv, MazeEnv
from environment import Kuka2Env
from torch_geometric.nn import knn_graph
from collections import defaultdict
from time import time
import pickle
from tqdm import tqdm
from torch_sparse import coalesce

INFINITY = float('inf')


def construct_graph(env, points, check_collision=True):
    edge_index = knn_graph(torch.FloatTensor(points), k=5, loop=True)
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
    prev[source] = source

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
    n_sample = [50, 200, 1000]
    # env = MazeEnv(dim=2,  map_file="maze_files/mazes_4000.npz")
    env = KukaEnv(map_file='maze_files/kukas_7_4000.pkl')
    # env = KukaEnv(kuka_file="kuka_iiwa/model_3.urdf", map_file="maze_files/kukas_13_3000.pkl")
    # env = Kuka2Env()

    time0 = time()

    # for n in n_sample:
    for problem_index in tqdm(range(4000)):

        env.init_new_problem(problem_index)
        points = env.uniform_sample(n=np.random.randint(100, 400))
        edge_cost, neighbors, edge_index, edge_free = construct_graph(env, points)

        data.append((points, neighbors, edge_cost, edge_index, edge_free))

        # dist, prev = dijkstra(list(range(len(points))), neighbors, edge_cost, 0)
        # valid_goal = np.logical_and(np.array(list(dist.values())) != INFINITY, np.array(list(dist.values()))!=0)
        # goal_index = np.random.choice(len(valid_goal), p=valid_goal.astype(float)/sum(valid_goal))
        #
        # print(time()-time0)
        # print('yes')

    with open('data/pkl/kuka_prm_4000.pkl', 'wb') as f:
        pickle.dump(data, f, pickle.DEFAULT_PROTOCOL)

