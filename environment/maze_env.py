import numpy as np
from .env_config import RRT_EPS, LIMITS, STICK_LENGTH


class MazeEnv:
    '''
    Interface class for maze environment
    '''

    RRT_EPS = RRT_EPS
    voxel_r = 1./15

    def __init__(self, dim, map_file=None):
        # print("Initializing environment...")
        self.dim = dim
        self.config_dim = dim
        self.collision_check_count = 0

        # load map from file
        if map_file is None:
            map_file = 'maze_files/mazes_15_%d_3000.npz' % dim
        # print("loading mazes from %s" % map_file)
        with np.load(map_file) as f:
            self.maps = f['maps']
            self.init_states = f['init_states']
            self.goal_states = f['goal_states']

        self.size = self.maps.shape[0]
        self.width = self.maps.shape[1]
        if dim == 2:
            self.bound = (-1, -1, 1, 1)
        else:
            self.bound = (-1, -1, -0.4, 1, 1, 0.4)
        self.order = list(range(self.size))
        self.episode_i = 0
        self.collision_point = None

    def __str__(self):
        return 'maze'+str(self.config_dim)

    def init_new_problem(self, index=None):
        '''
        Initialize a new planning problem
        '''
        if index is None:
            index = self.episode_i
        self.map = self.maps[self.order[index]]
        self.width = self.map.shape[0]
        self.init_state = self.init_states[self.order[index]]
        self.goal_state = self.goal_states[self.order[index]]
        self.episode_i += 1
        self.episode_i = (self.episode_i) % len(self.order)

        self.collision_point = None

        # if index==2000:
        #     self.map[:] = 1
        #     self.map[1:-1, 1:-1] = 0
        #     self.init_state = np.array([7, 1]) / 15 - 0.5
        #     self.goal_state = np.array([7, 14]) / 15 - 0.5
        #     self.map[:, 7] = 1
        #     self.map[9, 7] = 0
        #
        # elif index==2001:
        #     self.map[:] = 1
        #     self.map[1:-1, 1:-1] = 0
        #     self.init_state = np.array([7, 1]) / 15 - 0.5
        #     self.goal_state = np.array([7, 14]) / 15 - 0.5
        #     self.map[4:10, 7] = 1
        #     self.map[4, 4:7] = 1
        #     self.map[9, 4:7] = 1

        self.obstacles = []

        for i in range(self.map.shape[0]):
            for j in range(self.map.shape[1]):
                if self.map[i, j] == 1:
                    self.obstacles.append((i, j))
        self.obstacles = np.array(self.obstacles) / self.map.shape[0] - 0.5

        self.collision_check_count = 0

        return self.get_problem()

    def sample_n_points(self, n, need_negative=False):
        if need_negative:
            negative = []
        samples = []
        for i in range(n):
            while True:
                sample = self.uniform_sample()
                if (self.dim==2 and self._point_in_free_space(sample)) or (self.dim==3 and self._stick_in_free_space(sample)):
                    samples.append(sample)
                    break
                elif need_negative:
                    negative.append(sample)
        if not need_negative:
            return samples
        else:
            return samples, negative

    def sample_empty_points(self):
        while True:
            point = self.uniform_sample()
            if self.dim == 2:
                if self._point_in_free_space(point):
                    return point
            if self.dim == 3:
                if self._stick_in_free_space(point):
                    return point

    def set_random_init_goal(self):
        while True:
            init, goal = self.sample_empty_points(), self.sample_empty_points()
            if np.sum(np.abs(init - goal)) != 0:
                break
        self.init_state, self.goal_state = init, goal

    def get_problem(self):
        problem = {
            "map": self.map,
            "init_state": self.init_state,
            "goal_state": self.goal_state
        }
        return problem

    def uniform_sample(self, n=1):
        '''
        Uniformlly sample in the configuration space
        '''
        sample = np.random.uniform(-LIMITS[:self.dim], LIMITS[:self.dim], (n, self.dim))
        if n==1:
            return sample.reshape(-1)
        else:
            return sample

    def distance(self, from_state, to_state):
        '''
        Distance metric
        '''
        diff = np.abs(to_state - from_state)
        if diff.ndim == 1:
            diff = diff.reshape(1, -1)

        if self.dim >= 3:
            diff[:, 2] = np.min((diff[:, 2], np.abs(diff[:, 2] - 2 * LIMITS[2])), axis=0)
            assert (np.abs(diff[:, 2]) <= LIMITS[2]).all()

        return np.sqrt(np.sum(diff ** 2, axis=-1))

    def interpolate(self, from_state, to_state, ratio):
        diff = to_state - from_state

        if self.dim >= 3:
            if np.abs(diff[2]) > LIMITS[2]:
                if diff[2] > 0:
                    diff[2] -= 2 * LIMITS[2]
                else:
                    diff[2] += 2 * LIMITS[2]
            assert np.abs(diff[2]) <= LIMITS[2]

        new_state = from_state + diff * ratio

        if self.dim >= 3:
            if np.abs(new_state[2]) > LIMITS[2]:
                if new_state[2] > 0:
                    new_state[2] -= 2 * LIMITS[2]
                else:
                    new_state[2] += 2 * LIMITS[2]
            assert np.abs(new_state[2]) <= LIMITS[2]

        return new_state

    def in_goal_region(self, state):
        '''
        Return whether a state(configuration) is in the goal region
        '''
        return self.distance(state, self.goal_state) < RRT_EPS and \
               self._state_fp(state)

    def step(self, state, action=None, new_state=None, check_collision=True):
        '''
        Collision detection module
        '''
        # must specify either action or new_state
        if action is not None:
            new_state = state + action

        new_state[:2] = new_state[:2].clip(-LIMITS[:-1], LIMITS[:-1])
        if self.dim >= 3:
            if np.abs(new_state[2]) > LIMITS[2]:
                if new_state[2] > 0:
                    new_state[2] -= 2 * LIMITS[2]
                else:
                    new_state[2] += 2 * LIMITS[2]
            assert np.abs(new_state[2]) <= LIMITS[2]

        action = new_state - state

        if not check_collision:
            return new_state, action

        done = False
        no_collision = self._edge_fp(state, new_state)
        if no_collision and self.in_goal_region(new_state):
            done = True

        return new_state, action, no_collision, done

    def obs_map(self, resolution=voxel_r):
        xs = np.arange(-1, 1, resolution)
        ys = np.arange(-1, 1, resolution)
        xs, ys = np.meshgrid(xs, ys)
        points = np.concatenate((xs.reshape(-1, 1), ys.reshape(-1, 1)), axis=-1)
        obs = np.zeros(points.shape[0]).astype(bool)
        for index, point in enumerate(points):
            x, y = point
            if self.map[tuple(self._transform(np.array([x, y]), self.width))] == 1:
                obs[index] = True
        return points, obs

    def get_robot_points(self, config):
        return [config]

    def free_map(self, w=15):
        free_points = []
        for x in range(self.map.shape[0]):
            for y in range(self.map.shape[1]):
                if self.map[x, y] == 0:
                    free_points.append(np.array([1. / w + x * 2. / w - 1., 1. / w + y * 2. / w - 1]))
        return free_points

    # =====================internal collision check module=======================

    # transform a state into a discretized grid coordinate
    def _transform(self, state, w=15):
        coord = ((np.array(state)[:2].flatten() + 1.0) * w / 2.0).astype(int)
        coord[coord > w - 1] = w - 1
        return coord
    
    def _inverse_transform(self, coord, w=15):
        state = (np.array(coord) * 2.0 / w) - 1.0
        return state

    def _end_points(coord=None, l=None, center=None, theta=None, a=None, \
                    b=None):
        if theta is None:
            theta = coord[2] / LIMITS[2] * np.pi
        orient = np.array([np.cos(theta), np.sin(theta)])
        if l is None:
            l = STICK_LENGTH

        if a is None and b is None:
            if center is None:
                center = np.array(coord[:2])
            a = center - l / 2. * orient
            b = center + l / 2. * orient
        else:
            if a is not None:
                b = a + l * orient
            if b is not None:
                a = b - l * orient

        return a, b

    def _valid_state(self, state):
        return (state >= -LIMITS[:state.size]).all() and \
               (state <= LIMITS[:state.size]).all()

    def _point_in_free_space(self, state):
        assert state.size == 2
        if not self._valid_state(state):
            self.collision_point = state
            return False

        self.collision_check_count += 1
        return self.map[tuple(self._transform(state, self.width))] == 0

    def _stick_in_free_space(self, state):
        self.k = 0
        assert state.size == 3

        if not self._valid_state(state):
            return False

        a, b = MazeEnv._end_points(state)
        if not self._point_in_free_space(a) or not self._point_in_free_space(b):
            self.collision_point = state
            return False

        return self._iterative_check_segment(a, b)

    def _state_fp(self, state):
        assert state.size == 2 or state.size == 3 or state.size == 5

        if state.size == 2:
            return self._point_in_free_space(state)
        elif state.size == 3:
            return self._stick_in_free_space(state)

    def _iterative_check_segment(self, left, right):
        assert left.size == 2 and right.size == 2

        left_coord = np.array(self._transform(left, self.width), dtype=int)
        right_coord = np.array(self._transform(right, self.width), dtype=int)
        if (np.sum(np.abs(left_coord - right_coord)) > 1) and (np.sum(np.abs(left - right)) > self.RRT_EPS):
            mid = (left + right) / 2.0
            self.k += 1
            if not self._state_fp(mid):
                self.collision_point = mid
                return False
            return self._iterative_check_segment(left, mid) and self._iterative_check_segment(mid, right)

        return True

    def _edge_fp(self, state, new_state):
        self.k = 0
        assert state.size == new_state.size

        if not self._valid_state(state) or not self._valid_state(new_state):
            return False
        if not self._state_fp(state) or not self._state_fp(new_state):
            return False

        if state.size == 2:
            return self._iterative_check_segment(state, new_state)
        else:

            disp = new_state - state
            if np.abs(disp[2]) > LIMITS[2]:
                if disp[2] > 0:
                    disp[2] -= 2 * LIMITS[2]
                else:
                    disp[2] += 2 * LIMITS[2]
            assert np.abs(disp[2]) <= LIMITS[2]

            d = self.distance(state, new_state)
            K = int(d / 0.015)
            for k in range(1, K):
                c = state + k * 1. / K * disp

                if state.size == 3:
                    ca, cb = MazeEnv._end_points(c)
                    self.k += 1
                    if not self._edge_fp(ca, cb):
                        return False
            return True
