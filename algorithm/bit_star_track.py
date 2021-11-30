import numpy as np
import math
import yaml
import heapq
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from shapely.geometry import Point, LineString, Polygon
from descartes import PolygonPatch
from shapely import affinity
import itertools
from time import time
from collections import defaultdict

INF = float("inf")


class BITStar:
    def __init__(self, environment, maxIter=5, plot_flag=False, batch_size=200, T=1000, sampling=None):
        self.env = environment

        start, goal, bounds = tuple(environment.init_state), tuple(environment.goal_state), environment.bound

        self.start = start
        self.goal = goal

        self.bounds = bounds
        self.bounds = np.array(self.bounds).reshape((2, -1)).T
        self.ranges = self.bounds[:, 1] - self.bounds[:, 0]
        self.dimension = environment.config_dim

        # This is the tree
        self.vertices = []
        self.edges = dict()  # key = point，value = parent
        self.g_scores = dict()

        self.samples = []
        self.vertex_queue = []
        self.edge_queue = []
        self.old_vertices = set()

        self.maxIter = maxIter
        self.r = INF
        self.batch_size = batch_size
        self.T, self.T_max = 0, T
        self.eta = 1.1  # tunable parameter
        self.obj_radius = 1
        self.resolution = 3

        # the parameters for informed sampling
        self.c_min = self.distance(self.start, self.goal)
        self.center_point = None
        self.C = None

        # whether plot the middle planning process
        self.plot_planning_process = plot_flag

        if sampling is None:
            self.sampling = self.informed_sample
        else:
            self.sampling = sampling

        self.n_collision_points = 0
        self.n_free_points = 2

    def setup_planning(self):
        # add goal to the samples
        self.samples.append(self.goal)
        self.g_scores[self.goal] = INF

        # add start to the tree
        self.vertices.append(self.start)
        self.g_scores[self.start] = 0

        # Computing the sampling space
        self.informed_sample_init()
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

    def informed_sample_init(self):
        self.center_point = np.array([(self.start[i] + self.goal[i]) / 2.0 for i in range(self.dimension)])
        a_1 = (np.array(self.goal) - np.array(self.start)) / self.c_min
        id1_t = np.array([1.0] * self.dimension)
        M = np.dot(a_1.reshape((-1, 1)), id1_t.reshape((1, -1)))
        U, S, Vh = np.linalg.svd(M, 1, 1)
        self.C = np.dot(np.dot(U, np.diag([1] * (self.dimension - 1) + [np.linalg.det(U) * np.linalg.det(np.transpose(Vh))])), Vh)

    def sample_unit_ball(self):
        u = np.random.normal(0, 1, self.dimension)  # an array of d normally distributed random variables
        norm = np.sum(u ** 2) ** (0.5)
        r = np.random.random() ** (1.0 / self.dimension)
        x = r * u / norm
        return x

    def informed_sample(self, c_best, sample_num, vertices):
        if c_best < float('inf'):
            c_b = math.sqrt(c_best ** 2 - self.c_min ** 2) / 2.0
            r = [c_best / 2.0] + [c_b] * (self.dimension - 1)
            L = np.diag(r)
        sample_array = []
        cur_num = 0
        while cur_num < sample_num:
            if c_best < float('inf'):
                x_ball = self.sample_unit_ball()
                random_point = tuple(np.dot(np.dot(self.C, L), x_ball) + self.center_point)
            else:
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
            result = self.env._point_in_free_space(np.array(point))
        elif self.dimension == 3:
            result = self.env._stick_in_free_space(np.array(point))
        else:
            result = self.env._point_in_free_space(np.array(point))
        if result:
            self.n_free_points += 1
        else:
            self.n_collision_points += 1
        return result

    def is_edge_free(self, edge):
        result = self.env._edge_fp(np.array(edge[0]), np.array(edge[1]))
        # self.T += self.env.k
        if edge[0] not in self.explored_nodes:
            self.explored_nodes.append(edge[0])
        id0 = self.explored_nodes.index(edge[0])
        if result:
            if edge[1] not in self.explored_nodes:
                self.explored_nodes.append(edge[1])
            id1 = self.explored_nodes.index(edge[1])
        else:
            collision_point = tuple(self.env.collision_point)
            if collision_point not in self.explored_nodes:
                self.explored_nodes.append(collision_point)
            id1 = self.explored_nodes.index(collision_point)
        self.explored_edges.append((id0, id1))
        return result

    def get_g_score(self, point):
        # gT(x)
        if point == self.start:
            return 0
        if point not in self.edges:
            return INF
        else:
            return self.g_scores.get(point)

    def get_f_score(self, point):
        # f^(x)
        return self.heuristic_cost(self.start, point) + self.heuristic_cost(point, self.goal)

    def actual_edge_cost(self, point1, point2):
        # c(x1,x2)
        if not self.is_edge_free([point1, point2]):
            return INF
        return self.distance(point1, point2)

    def heuristic_cost(self, point1, point2):
        # Euler distance as the heuristic distance
        return self.distance(point1, point2)

    def distance(self, point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))

    def get_edge_value(self, edge):
        # sort value for edge
        return self.get_g_score(edge[0]) + self.heuristic_cost(edge[0], edge[1]) + self.heuristic_cost(edge[1],
                                                                                                       self.goal)

    def get_point_value(self, point):
        # sort value for point
        return self.get_g_score(point) + self.heuristic_cost(point, self.goal)

    def bestVertexQueueValue(self):
        if not self.vertex_queue:
            return INF
        else:
            return self.vertex_queue[0][0]

    def bestEdgeQueueValue(self):
        if not self.edge_queue:
            return INF
        else:
            return self.edge_queue[0][0]

    def prune_edge(self, c_best):
        edge_array = list(self.edges.items())
        for point, parent in edge_array:
            if self.get_f_score(point) > c_best or self.get_f_score(parent) > c_best:
                self.edges.pop(point)

    def prune(self, c_best):
        self.samples = [point for point in self.samples if self.get_f_score(point) < c_best]
        self.prune_edge(c_best)
        vertices_temp = []
        for point in self.vertices:
            if self.get_f_score(point) <= c_best:
                if self.get_g_score(point) == INF:
                    self.samples.append(point)
                else:
                    vertices_temp.append(point)
        self.vertices = vertices_temp

    def expand_vertex(self, point):
        # get the nearest value in vertex for every one in samples where difference is less than the radius
        neigbors_sample = []
        for sample in self.samples:
            if self.distance(point, sample) <= self.r:
                neigbors_sample.append(sample)

        # add an edge to the edge queue is the path might improve the solution
        for neighbor in neigbors_sample:
            estimated_f_score = self.heuristic_cost(self.start, point) + \
                                self.heuristic_cost(point, neighbor) + self.heuristic_cost(neighbor, self.goal)
            if estimated_f_score < self.g_scores[self.goal]:
                heapq.heappush(self.edge_queue, (self.get_edge_value((point, neighbor)), (point, neighbor)))

        # add the vertex to the edge queue
        if point not in self.old_vertices:
            neigbors_vertex = []
            for ver in self.vertices:
                if self.distance(point, ver) <= self.r:
                    neigbors_vertex.append(ver)
            for neighbor in neigbors_vertex:
                if neighbor not in self.edges or point != self.edges.get(neighbor):
                    estimated_f_score = self.heuristic_cost(self.start, point) + \
                                        self.heuristic_cost(point, neighbor) + self.heuristic_cost(neighbor, self.goal)
                    if estimated_f_score < self.g_scores[self.goal]:
                        estimated_g_score = self.get_g_score(point) + self.heuristic_cost(point, neighbor)
                        if estimated_g_score < self.get_g_score(neighbor):
                            heapq.heappush(self.edge_queue, (self.get_edge_value((point, neighbor)), (point, neighbor)))

    def get_best_path(self):
        path = []
        if self.g_scores[self.goal] != INF:
            path.append(self.goal)
            point = self.goal
            while point != self.start:
                point = self.edges[point]
                path.append(point)
            path.reverse()
        return path

    def path_length_calculate(self, path):
        path_length = 0
        for i in range(len(path) - 1):
            path_length += self.distance(path[i], path[i + 1])
        return path_length

    def plan(self, pathLengthLimit, refine_time_budget=None, time_budget=None, seed=1234):
        collision_checks = self.env.collision_check_count
        if time_budget is None:
            time_budget = INF
        if refine_time_budget is None:
            refine_time_budget = 10

        radius_constant = self.setup_planning()
        path = []
        init_time = time()

        self.explored_nodes = []
        self.explored_edges = []

        while self.T < self.T_max and (time() - init_time < time_budget):
            if not self.vertex_queue and not self.edge_queue:
                c_best = self.g_scores[self.goal]
                path = self.get_best_path()
                self.prune(c_best)
                self.samples.extend(self.sampling(c_best, self.batch_size, self.vertices))
                self.T += self.batch_size

                self.old_vertices = set(self.vertices)
                self.vertex_queue = [(self.get_point_value(point), point) for point in self.vertices]
                heapq.heapify(self.vertex_queue)  # change to op priority queue
                q = len(self.vertices) + len(self.samples)
                self.r = self.radius_init() * ((math.log(q) / q) ** (1.0 / self.dimension))

            try:
                while self.bestVertexQueueValue() <= self.bestEdgeQueueValue():
                    _, point = heapq.heappop(self.vertex_queue)
                    self.expand_vertex(point)
            except Exception as e:
                if (not self.edge_queue) and (not self.vertex_queue):
                    continue
                else:
                    raise e

            best_edge_value, bestEdge = heapq.heappop(self.edge_queue)

            # Check if this can improve the current solution
            if best_edge_value < self.g_scores[self.goal]:
                actual_cost_of_edge = self.actual_edge_cost(bestEdge[0], bestEdge[1])
                actual_f_edge = self.heuristic_cost(self.start,
                                                    bestEdge[0]) + actual_cost_of_edge + self.heuristic_cost(
                    bestEdge[1], self.goal)
                if actual_f_edge < self.g_scores[self.goal]:
                    actual_g_score_of_point = self.get_g_score(bestEdge[0]) + actual_cost_of_edge
                    if actual_g_score_of_point < self.get_g_score(bestEdge[1]):
                        self.g_scores[bestEdge[1]] = actual_g_score_of_point
                        self.edges[bestEdge[1]] = bestEdge[0]
                        if bestEdge[1] not in self.vertices:
                            self.samples.remove(bestEdge[1])
                            self.vertices.append(bestEdge[1])
                            heapq.heappush(self.vertex_queue, (self.get_point_value(bestEdge[1]), bestEdge[1]))

                        self.edge_queue = [item for item in self.edge_queue if item[1][1] != bestEdge[1] or \
                                           self.get_g_score(item[1][0]) + self.heuristic_cost(item[1][0], item[1][
                            1]) < self.get_g_score(item[1][0])]
                        heapq.heapify(
                            self.edge_queue)  # Rebuild the priority queue because it will be destroyed after the element is removed

            else:
                self.vertex_queue = []
                self.edge_queue = []
            if self.g_scores[self.goal] < pathLengthLimit and (time() - init_time > refine_time_budget):
                break
        return self.samples, self.edges, self.explored_nodes, self.explored_edges, self.env.collision_check_count - collision_checks, \
               self.g_scores[self.goal], self.T, time() - init_time


if __name__ == '__main__':
    from utils.plot import plot_edges
    from config import set_random_seed
    from environment import MazeEnv
    from tqdm import tqdm

    solutions = []

    environment = MazeEnv(dim=3)


    def sample_empty_points(env):
        while True:
            point = np.random.uniform(-1, 1, 2)
            if env._point_in_free_space(point):
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
