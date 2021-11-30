import numpy as np
import math
import yaml
import heapq
import time
from shapely.geometry import Point, LineString, Polygon
from descartes import PolygonPatch
from shapely import affinity
import itertools
from time import time
from environment.timer import Timer
from maze_prm import dijkstra
from torch_geometric.nn import knn_graph
import torch
from torch_sparse import coalesce
from collections import defaultdict

INF = float("inf")


class LazySP:
    def __init__(self, environment, batch_size=100, T=1000):

        self.env = environment

        start, goal, bounds = tuple(environment.init_state), tuple(environment.goal_state), environment.bound

        self.start = start
        self.goal = goal

        self.bounds = bounds
        self.bounds = np.array(self.bounds).reshape((2, -1)).T
        self.ranges = self.bounds[:, 1] - self.bounds[:, 0]
        self.dimension = environment.config_dim

        # This is the tree
        self.edges = dict()  # key = point，value = parent

        self.samples = []
        self.invalid_edges = set()
        self.valid_edges = set()

        self.r = INF
        self.batch_size = batch_size
        self.T, self.T_max = 0, T
        self.eta = 1.1  # tunable parameter
        self.obj_radius = 1
        self.resolution = 3

        self.n_collision_points = 0
        self.n_free_points = 2

    def setup_planning(self):
        # add start and goal to the samples
        self.samples.extend([self.goal, self.start])

        # Computing the sampling space
        radius_constant = self.radius_init()

        return radius_constant

    def radius_init(self):
        from scipy import special
        # Hypersphere radius calculation
        n = self.dimension
        unit_ball_volume = np.pi ** (n / 2.0) / special.gamma(n / 2.0 + 1)
        volume = np.abs(np.prod(self.ranges)) * self.n_free_points / (self.n_collision_points + self.n_free_points)
        gamma = (1.0 + 1.0 / n) * volume / unit_ball_volume
        radius_constant = 2 * self.eta * (gamma ** (1.0 / n))
        return radius_constant

    def informed_sample(self, sample_num):
        sample_array = []
        cur_num = 0
        while cur_num < sample_num:
            random_point = self.get_random_point()
            if self.is_point_free(random_point):
                sample_array.append(random_point)
                cur_num += 1
        return sample_array

    def get_random_point(self):
        point = self.bounds[:, 0] + np.random.random(self.dimension) * self.ranges
        return tuple(point)

    def is_point_free(self, point):
        if self.dimension == 2:
            result = self.env._state_fp(np.array(point))
        elif self.dimension == 3:
            result = self.env._state_fp(np.array(point))
        else:
            result = self.env._state_fp(np.array(point))
        if result:
            self.n_free_points += 1
        else:
            self.n_collision_points += 1
        return result

    def is_edge_free(self, edge):
        result = self.env._edge_fp(np.array(edge[0]), np.array(edge[1]))
        return result
    
    def get_path(self, prev, start, goal):
        path = [start]
        current = start
        while current != goal:
            current = prev[current]
            path.append(current)
        return path

    def path_length_calculate(self, path):
        path_length = 0
        for i in range(len(path) - 1):
            path_length += self.distance(path[i], path[i + 1])
        return path_length
    
    def construct_graph(self, k, points, env):
        points = np.array(points)
        edge_index = knn_graph(torch.FloatTensor(points), k=k, loop=True)
        edge_index = torch.cat((edge_index, edge_index.flip(0)), dim=-1)
        edge_index_torch, _ = coalesce(edge_index, None, len(points), len(points))
        edge_index = edge_index_torch.data.cpu().numpy().T
        edge_cost = defaultdict(list)
        neighbors = defaultdict(list)
        for i, edge in enumerate(edge_index):
            if (edge[0], edge[1]) not in self.invalid_edges:
                edge_cost[edge[1]].append(np.linalg.norm(points[edge[1]]-points[edge[0]]))
                neighbors[edge[1]].append(edge[0])
        return edge_cost, neighbors, edge_index
    
    def remove_neighbor(self, edge_cost, neighbors, n1, n2):
        index = neighbors[n1].index(n2)
        edge_cost[n1].pop(index)
        neighbors[n1].pop(index)
        index = neighbors[n2].index(n1)
        edge_cost[n2].pop(index)
        neighbors[n2].pop(index)        

    def plan(self):
        collision_checks = self.env.collision_check_count

        self.setup_planning()
        init_time = time()

        while self.T < self.T_max:
            self.samples.extend(self.informed_sample(self.batch_size))
            print(self.env.collision_check_count-collision_checks)
            self.T += self.batch_size
            
            q = len(self.samples)
            self.r = self.radius_init() * ((math.log(q) / q) ** (1.0 / self.dimension))
            self.k = int(np.ceil(10*np.log(q)/np.log(100)))
            edge_cost, neighbors, edge_index = self.construct_graph(self.k, self.samples, self.env)

            while True:  # continue until Dijkstra finds that the graph is infeasible
                dist, prev = dijkstra(list(range(len(self.samples))), neighbors, edge_cost, 0)
                if dist[1] != float('inf'):
                    feasible = True
                    path = self.get_path(prev, 1, 0)
                    for n1, n2 in zip(path[:-1], path[1:]):
                        if (n1, n2) in self.valid_edges:
                            continue
                        elif (n1, n2) in self.invalid_edges:
                            assert False, "You shouldn't find invalid edges from Dijkstra solution"
                            feasible = False
                        else:
                            # check the collision status
                            free = self.is_edge_free((self.samples[n1], self.samples[n2]))
                            if free:
                                self.valid_edges.add((n1,n2))
                                self.valid_edges.add((n2,n1))                                
                            else:
                                self.invalid_edges.add((n1,n2))
                                self.invalid_edges.add((n2,n1))
                                self.remove_neighbor(edge_cost, neighbors, n1, n2)
                                feasible = False
                                break
                    
                    if feasible:
                        return self.samples, self.env.collision_check_count - collision_checks, \
                            [self.samples[n] for n in path], self.T, time() - init_time, \
                            self.valid_edges, self.invalid_edges
                else:
                    break
            
        return self.samples, self.env.collision_check_count - collision_checks, [], self.T, time() - init_time, \
                self.valid_edges, self.invalid_edges


if __name__ == '__main__':
    from utils.plot import plot_edges
    from config import set_random_seed
    from environment import MazeEnv
    from tqdm import tqdm

    solutions = []

    environment = MazeEnv(dim=2)


    def sample_empty_points(env):
        while True:
            point = np.random.uniform(-1, 1, 2)
            if env._state_fp(point):
                return point


    for _ in tqdm(range(3000)):
        pb = environment.init_new_problem()
        set_random_seed(1234)

        cur_time = time.time()

        BIT = BITStar(environment)
        nodes, edges, collision, success, n_samples = BIT.plan(INF)

        solutions.append((nodes, edges, collision, success, n_samples))

        plot_edges(set(nodes)|set(edges.keys()), edges, environment.get_problem())

    print('hello')
