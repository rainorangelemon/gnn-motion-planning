import numpy as np
from environment.env_config import LIMITS
from copy import deepcopy
from collections import defaultdict
from torch_geometric.nn import knn_graph
import torch
from torch_geometric.utils import remove_self_loops, add_self_loops
import heapq
import networkx as nx
import collections
from torch_geometric.data import Data


class EdgeAttribute:
    Collided = 0
    Free = 1
    Unknown = 2


class Graph:

    def __init__(self, env, k=6):
        self.env = env
        self.dim = env.dim
        self.k = k  # the number of neighbors for k-nn of edges
        self.V = []
        self.V_attr = []  # the attribute of vertices
        self.E = []
        self.E_attr = {}  # the attribute of edges

        # hyper parameters
        self.eta = 1.1

    def radius(self, n_samples):
        from scipy import special
        # Hypersphere radius calculation
        n = self.env.dim
        unit_ball_volume = np.pi**(n/2.0) / special.gamma(n/2.0+1)
        volume = np.abs(np.prod(LIMITS))*(2**n)
        gamma = (1.0 + 1.0/n) * volume / unit_ball_volume
        self.radius_constant = 2 * self.eta * (gamma**(1.0/n))
        q =  n_samples
        r = self.radius_constant * ((np.log(q) / q) ** (1.0 / self.env.dim))
        return r

    def initialize(self, n_samples, self_loop=True):
        '''
        :param n_samples: the number of samples to sample in the whole space
        :param self_loop: boolean, True if add edge to itself
        :return: the instance
        '''
        env = self.env
        self.V.extend([tuple(env.init_state), tuple(env.goal_state)])
        self.V_attr.extend([True, True])

        for i in range(n_samples):
            sample = env.uniform_sample()
            if env._point_in_free_space(sample):
                self.V.append(tuple(sample))
                self.V_attr.append(True)

        # for sample in env.obs_map():
        #     self.V.append(tuple(sample))
        #     self.V_attr.append(False)

        self.r = self.radius(np.sum(np.array(self.V_attr)))

        # construct edges
        for i, point in enumerate(self.V):
            dists = env.distance(np.array(self.V), point)
            near = np.where(dists <= self.r)[0]
            for j in near:
                if i!=j:
                    if self.V_attr[i] and self.V_attr[j]:
                        self.E.append((i, j))
                        self.E_attr[i, j] = EdgeAttribute.Unknown
                    else:
                        self.E.append((i, j))
                        self.E_attr[i, j] = EdgeAttribute.Collided

        return self

    def update(self, edge_index, edge, no_collision, collision_point):
        if no_collision:
            self.E_attr[edge] = EdgeAttribute.Free
            self.E_attr[edge[1], edge[0]] = EdgeAttribute.Free
        else:
            self.E_attr[edge] = EdgeAttribute.Collided
            self.E_attr[edge[1], edge[0]] = EdgeAttribute.Collided
            # # add collision point
            # self.V.append(tuple(collision_point))
            # self.V_attr.append(False)

    def finish(self):
        # construct edges
        self.E = []
        self.E_mask = []
        for i, point in enumerate(self.V):
            dists = self.env.distance(np.array(self.V), point)
            near = np.where(dists <= self.r)[0]
            for j in near:
                if i!=j:
                    self.E.append((i, j))
                    if (i, j) in self.E_attr and self.V_attr[i] and self.V_attr[j]:
                        self.E_mask.append(True)
                    else:
                        self.E_mask.append(False)

        self.G = nx.DiGraph()
        self.G.add_nodes_from(self.V)
        self.G.add_edges_from([edge for edge, attr in self.E_attr.items() if attr==EdgeAttribute.Free])

        lengths = nx.all_pairs_shortest_path_length(self.G)
        pair_to_length_dict = {}
        for x, yy in lengths:
            for y, l in yy.items():
                if l >= 1:
                    pair_to_length_dict[x, y] = l
        if max(pair_to_length_dict.values()) < 1:
            raise ValueError("All shortest paths are below the minimum length")
        # The node pairs which exceed the minimum length.
        self.node_pairs = list(pair_to_length_dict)

        counts = collections.Counter(pair_to_length_dict.values())
        prob_per_length = 1.0 / len(counts)
        self.probabilities = [
            prob_per_length / counts[pair_to_length_dict[x]] for x in self.node_pairs
        ]

    def random_problem(self):
        # Choose the start and end points.
        i = np.random.choice(len(self.node_pairs), p=self.probabilities)
        start, end = self.node_pairs[i]
        path = self.path = nx.dijkstra_path(self.G, source=start, target=end, weight="distance")
        data =  self.create_graph(start, end)

        y_node = np.zeros(len(self.V)).astype(int)
        y_node[list(path)] = 1

        edge_path = []
        for node_i, node_j in zip(path[:-1], path[1:]):
            edge_path.append((node_i, node_j))

        y_edge = np.zeros(len(self.E)).astype(int)
        for index, edge in enumerate(self.E):
            if tuple(edge) in edge_path:
                y_edge[index] = 1

        data.y_node = torch.LongTensor(y_node)
        data.y_edge = torch.LongTensor(y_edge)
        data.y_edge_free = torch.LongTensor(np.array(list(self.E_attr.values())))
        data.y_mask_node = torch.BoolTensor(np.array(self.V_attr))
        data.y_mask_edge = torch.BoolTensor(np.array(self.E_mask))

        return data

    def create_graph(self, start_idx, end_idx, start_sets=None, no_collision_edge=False):
        if start_sets is None:
            start_sets = {start_idx}
        self.start, self.end = start_idx, end_idx
        points = self.V
        edges = self.E
        points = np.array(points)
        start = np.array(points[start_idx])
        end = np.array(points[end_idx])
        edges = np.array(edges).astype(int)
        # create graph
        x = np.hstack((points - start, points - end))
        x = np.hstack((x,
                       # np.array(list(weight.values())).astype(float).reshape((-1, 1)),
                       np.array([point in start_sets for point in range(len(points))]).astype(float).reshape((-1, 1)),
                       (np.arange(len(points)) == end_idx).astype(float).reshape((-1, 1)),
                       np.array(self.V_attr).reshape((-1, 1))))
        edge_attr = np.array(points[edges[:, 0]] - points[edges[:, 1]])
        edge_attr = np.hstack((edge_attr, np.linalg.norm(edge_attr, axis=-1).reshape((-1, 1))))

        if no_collision_edge:
            edge_mask = (torch.LongTensor(np.array(list(self.E_attr.values())))!=EdgeAttribute.Collided)
        else:
            edge_mask = torch.ones(len(self.E)).bool()

        return Data(x=torch.FloatTensor(x), edge_index=torch.LongTensor(edges).T[:, edge_mask],
                    edge_attr=torch.FloatTensor(edge_attr)[edge_mask, :],)


def radius(n_samples, dim):
    eta = 1.1

    from scipy import special
    # Hypersphere radius calculation
    n = dim
    unit_ball_volume = np.pi**(n/2.0) / special.gamma(n/2.0+1)
    volume = np.abs(np.prod(LIMITS))*(2**n)
    gamma = (1.0 + 1.0/n) * volume / unit_ball_volume
    radius_constant = 2 * eta * (gamma**(1.0/n))
    q = n_samples
    r = radius_constant * ((np.log(q) / q) ** (1.0 / dim))
    return r


def create_data(env, y_config):

    points, obs_mask = env.obs_map()

    data = Data(x_map=torch.FloatTensor(points),
                x_obs_mask=torch.LongTensor(obs_mask),
                x_goal=torch.FloatTensor(env.goal_state),
                y=torch.FloatTensor(y_config))

    return data
