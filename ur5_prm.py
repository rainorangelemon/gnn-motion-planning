import numpy as np
import torch
from environment import KukaEnv, MazeEnv, SnakeEnv, UR5Env
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
    problems = []
    n_sample = [50, 200, 1000]
    # env = MazeEnv(dim=2,  map_file="maze_files/mazes_4000.npz")
    env = UR5Env(map_file='data/pkl/problems_ur5_300.pkl')
    # env = KukaEnv(kuka_file="kuka_iiwa/model_3.urdf", map_file="maze_files/kukas_13_3000.pkl")
    # env = Kuka2Env()

    time0 = time()

    # for n in n_sample:\
    pbar = tqdm(range(300))
    for problem_index in pbar:
        for _ in range(10):

            env.init_new_problem(problem_index)
            points = np.vstack((env.sample_n_points(n=300), env.uniform_sample(200)))
            edge_cost, neighbors, edge_index, edge_free = construct_graph(env, points)

            for source_index in range(len(points)):
                dist, prev = dijkstra(list(range(len(points))), neighbors, edge_cost, source_index)
                valid_goal = np.logical_and(np.array(list(dist.values())) != INFINITY, np.array(list(dist.values()))!=0)

                if np.sum(valid_goal) == 0:
                    continue

                goal_index = np.array(list(dist.values()))[valid_goal].argmax()
                goal_index = np.arange(len(dist))[valid_goal][goal_index]
                problems.append(env.problems[problem_index])
                data.append((points, neighbors, edge_cost, edge_index, edge_free))
                break

        #
        # print(time()-time0)
        # print('yes')

    index_2000 = np.random.permutation(2000).tolist()
    index_3000 = index_2000 + np.random.permutation(np.arange(2000, 3000)).tolist()

    with open('data/pkl/ur5_prm_3000.pkl', 'wb') as f:
        pickle.dump([data[i] for i in index_3000], f, pickle.DEFAULT_PROTOCOL)

    with open('data/pkl/problems_ur5_3000.pkl', 'wb') as f:
        pickle.dump([problems[i] for i in index_3000], f, pickle.DEFAULT_PROTOCOL)

    # with np.load('maze_files/snakes_15_2_3000.npz') as f:
    #     maps = f['maps']
    #     init_states = f['init_states']
    #     goal_states = f['goal_states']
    #
    print('yes')




