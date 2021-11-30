import numpy as np
import pybullet as p
from time import sleep
import pybullet_data
import pickle
from time import time
from environment.timer import Timer


class KukaEnv:
    '''
    Interface class for maze environment
    '''

    RRT_EPS = 0.5
    voxel_r = 0.1

    def __init__(self, GUI=False, kuka_file="kuka_iiwa/model_0.urdf", map_file='maze_files/kukas_7_3000.pkl'):
        # print("Initializing environment...")

        self.dim = 3
        self.kuka_file = kuka_file

        self.collision_check_count = 0
        self.collision_time = 0

        self.maps = {}
        self.episode_i = 0
        self.collision_point = None

        if GUI:
            p.connect(p.GUI, options='--background_color_red=1.0 --background_color_green=1.0 --background_color_blue=1.0')
        else:
            p.connect(p.DIRECT)

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, lightPosition = [0, 0, 0.1])

        with open(map_file, 'rb') as f:
            self.problems = pickle.load(f)

        self.timer = Timer()

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.kukaId = p.loadURDF(kuka_file, [0, 0, 0], [0, 0, 0, 1], useFixedBase=True)
        p.performCollisionDetection()

        self.config_dim = p.getNumJoints(self.kukaId)

        if self.config_dim == 7:
            target = p.getDebugVisualizerCamera()[11]
            p.resetDebugVisualizerCamera(
                cameraDistance=2,
                cameraYaw=-176.64,
                cameraPitch=-30.31,
                cameraTargetPosition=[0, 0, 0])
        else:
            pass

        self.pose_range = [(p.getJointInfo(self.kukaId, jointId)[8], p.getJointInfo(self.kukaId, jointId)[9]) for
                           jointId in
                           range(p.getNumJoints(self.kukaId))]
        self.bound = np.array(self.pose_range).T.reshape(-1)

        self.kukaEndEffectorIndex = self.config_dim-1

        p.setGravity(0, 0, -10)

        self.order = list(range(len(self.problems)))
        self.episode_i = 0

    def __str__(self):
        return 'kuka'+str(self.config_dim)

    def init_new_problem(self, index=None):
        '''
        Initialize a new planning problem
        '''
        if index is None:
            self.index = self.episode_i
        else:
            self.index = index

        self.timer.start()

        obstacles, start, goal, path = self.problems[index]

        self.episode_i += 1
        self.episode_i = (self.episode_i) % len(self.order)
        self.collision_check_count = 0
        self.collision_time = 0

        p.resetSimulation()
        self.kukaId = p.loadURDF(self.kuka_file, [0, 0, 0], [0, 0, 0, 1], useFixedBase=True)
        p.performCollisionDetection()

        self.collision_point = None

        self.obstacles = obstacles
        self.init_state = start
        self.goal_state = goal
        self.path = path

        for halfExtents, basePosition in obstacles:
            self.create_voxel(halfExtents, basePosition)

        self.timer.finish(Timer.CREATE)

        return self.get_problem()

    def set_random_init_goal(self):
        while True:
            points = self.sample_n_points(n=2)
            init, goal = points[0], points[1]
            if np.sum(np.abs(init - goal)) != 0:
                break
        self.init_state, self.goal_state = init, goal

    def aug_path(self):
        result = [self.init_state]
        path = np.array(self.path)
        agent = np.array(path[0])
        next_index = 1
        while next_index < len(path):
            if np.linalg.norm(self.path[next_index]-agent) <= self.RRT_EPS:
                agent = path[next_index]
                next_index += 1
            else:
                agent = agent + self.RRT_EPS * (path[next_index]-agent) / np.linalg.norm(path[next_index]-agent)

            result.append(np.array(agent))
        return result

    def get_problem(self, width=15, index=None):
        if index is None:
            problem = {
                "map": np.array(self.obs_map(width)[1]).astype(float),
                "init_state": self.init_state,
                "goal_state": self.goal_state
            }
            self.maps[self.index] = problem
            return problem
        else:
            return self.maps[index]

    def obs_map(self, num):
        resolution = 2./(num-1)
        grid_pos = [np.linspace(-1., 1., num=num) for i in range(3)]
        points_pos = np.meshgrid(*grid_pos)
        points_pos = np.concatenate((points_pos[0].reshape(-1, 1), points_pos[1].reshape(-1, 1), points_pos[2].reshape(-1, 1)),
                       axis=-1)
        points_obs = np.zeros(points_pos.shape[0]).astype(bool)

        for obstacle in self.obstacles:
            obstacle_size, obstacle_base = obstacle
            limit_low, limit_high = obstacle_base - obstacle_size, obstacle_base + obstacle_size
            limit_low[2], limit_high[2] = limit_low[2] - 0.4, limit_high[2] - 0.4  # translate the point
            bools = []
            for i in range(3):
                obs_mask = np.zeros(num).astype(bool)
                obs_mask[max(int((limit_low[i]+1)/resolution), 0):min((1+int((limit_high[i]+1)/resolution)), 1+int(2./resolution))] = True
                bools.append(obs_mask)
            current_obs = np.meshgrid(*bools)
            current_obs = np.concatenate((current_obs[0].reshape(-1, 1), current_obs[1].reshape(-1, 1), current_obs[2].reshape(-1, 1)),
                       axis=-1)
            points_obs = np.logical_or(points_obs, np.all(current_obs, axis=-1))
        return points_pos.reshape((num, num, num, -1)), points_obs.reshape((num, num, num))

    def get_robot_points(self, config, end_point=True):
        points = []
        for i in range(p.getNumJoints(self.kukaId)):
            p.resetJointState(self.kukaId, i, config[i])
        if end_point:
            point = p.getLinkState(self.kukaId, self.kukaEndEffectorIndex)[0]
            point = (point[0], point[1], point[2] - 0.4)
            return point
        for effector in range(self.kukaEndEffectorIndex + 1):
            point = p.getLinkState(self.kukaId, effector)[0]
            point = (point[0], point[1], point[2] - 0.4)
            points.append(point)
        return points

    def create_voxel(self, halfExtents, basePosition):
        groundColId = p.createCollisionShape(p.GEOM_BOX, halfExtents=halfExtents)
        groundVisID = p.createVisualShape(shapeType=p.GEOM_BOX,
                                          rgbaColor=np.random.uniform(0, 1, size=3).tolist() + [0.8],
                                          specularColor=[0.4, .4, 0],
                                          halfExtents=halfExtents)
        groundId = p.createMultiBody(baseMass=0,
                                     baseCollisionShapeIndex=groundColId,
                                     baseVisualShapeIndex=groundVisID,
                                     basePosition=basePosition)
        return groundId

    def sample_n_points(self, n, need_negative=False):
        if need_negative:
            negative = []
        samples = []
        for i in range(n):
            while True:
                sample = self.uniform_sample()
                if self._point_in_free_space(sample):
                    samples.append(sample)
                    break
                elif need_negative:
                    negative.append(sample)
        if not need_negative:
            return samples
        else:
            return samples, negative

    def uniform_sample(self, n=1):
        '''
        Uniformlly sample in the configuration space
        '''
        self.timer.start()
        sample = np.random.uniform(np.array(self.pose_range)[:, 0], np.array(self.pose_range)[:, 1], size=(n, self.config_dim))
        if n==1:
            self.timer.finish(Timer.SAMPLE)
            return sample.reshape(-1)
        else:
            self.timer.finish(Timer.SAMPLE)
            return sample

    def distance(self, from_state, to_state):
        '''
        Distance metric
        '''

        to_state = np.maximum(to_state, np.array(self.pose_range)[:, 0])
        to_state = np.minimum(to_state, np.array(self.pose_range)[:, 1])
        diff = np.abs(to_state - from_state)

        return np.sqrt(np.sum(diff ** 2, axis=-1))

    def interpolate(self, from_state, to_state, ratio):
        diff = to_state - from_state

        new_state = from_state + diff * ratio
        new_state = np.maximum(new_state, np.array(self.pose_range)[:, 0])
        new_state = np.minimum(new_state, np.array(self.pose_range)[:, 1])

        return new_state

    def in_goal_region(self, state):
        '''
        Return whether a state(configuration) is in the goal region
        '''
        return self.distance(state, self.goal_state) < self.RRT_EPS and \
               self._state_fp(state)

    def step(self, state, action=None, new_state=None, check_collision=True):
        '''
        Collision detection module
        '''
        # must specify either action or new_state
        if action is not None:
            new_state = state + action

        new_state = np.maximum(new_state, np.array(self.pose_range)[:, 0])
        new_state = np.minimum(new_state, np.array(self.pose_range)[:, 1])

        action = new_state - state

        if not check_collision:
            return new_state, action

        done = False
        no_collision = self._edge_fp(state, new_state)
        if no_collision and self.in_goal_region(new_state):
            done = True

        return new_state, action, no_collision, done

    def set_config(self, c, kukaId=None):
        if kukaId is None:
            kukaId = self.kukaId
        for i in range(p.getNumJoints(kukaId)):
            p.resetJointState(kukaId, i, c[i])
        p.performCollisionDetection()

    def plot(self, path, make_gif=False):
        path = np.array(path)

        p.resetSimulation()

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # p.loadURDF("plane.urdf", [0, 0, -1], useFixedBase=True)

        for halfExtents, basePosition in self.obstacles:
            self.create_voxel(halfExtents, basePosition)

        self.kukaId = p.loadURDF(self.kuka_file, [0, 0, 0], [0, 0, 0, 1], useFixedBase=True, flags=p.URDF_IGNORE_COLLISION_SHAPES)
        self.pose_range = [(p.getJointInfo(self.kukaId, jointId)[8], p.getJointInfo(self.kukaId, jointId)[9]) for
                           jointId in
                           range(p.getNumJoints(self.kukaId))]
        self.bound = np.array(self.pose_range).T.reshape(-1)

        self.set_config(path[0])

        target_kukaId = p.loadURDF(self.kuka_file, [0, 0, 0], [0, 0, 0, 1], useFixedBase=True, flags=p.URDF_IGNORE_COLLISION_SHAPES)
        self.set_config(path[-1], target_kukaId)

        prev_pos = p.getLinkState(self.kukaId, self.kukaEndEffectorIndex)[0]
        final_pos = p.getLinkState(target_kukaId, self.kukaEndEffectorIndex)[0]

        p.setGravity(0, 0, -10)
        p.stepSimulation()

        gifs = []
        current_state_idx = 0

        while True:
            current_state = path[current_state_idx]
            disp = path[current_state_idx + 1] - path[current_state_idx]

            d = self.distance(path[current_state_idx], path[current_state_idx + 1])

            new_kuka = p.loadURDF(self.kuka_file, [0, 0, 0], [0, 0, 0, 1], useFixedBase=True,
                                  flags=p.URDF_IGNORE_COLLISION_SHAPES)
            for data in p.getVisualShapeData(new_kuka):
                color = list(data[-1])
                color[-1] = 0.5
                p.changeVisualShape(new_kuka, data[1], rgbaColor=color)

            K = int(np.ceil(d / 0.2))
            for k in range(0, K):

                c = path[current_state_idx] + k * 1. / K * disp
                self.set_config(c, new_kuka)
                new_pos = p.getLinkState(new_kuka, self.kukaEndEffectorIndex)[0]
                p.addUserDebugLine(prev_pos, new_pos, [1, 0, 0], 10, 0)
                prev_pos = new_pos
                p.loadURDF("sphere2red.urdf", new_pos, globalScaling=0.05, flags=p.URDF_IGNORE_COLLISION_SHAPES)
                if make_gif:
                    gifs.append(p.getCameraImage(width=1080, height=720, lightDirection=[0, 0, -1], shadow=0,
                                                 renderer=p.ER_BULLET_HARDWARE_OPENGL)[2])

            current_state_idx += 1
            if current_state_idx == len(path) - 1:
                self.set_config(path[-1], new_kuka)
                p.addUserDebugLine(prev_pos, final_pos, [1, 0, 0], 10, 0)
                p.loadURDF("sphere2red.urdf", final_pos, globalScaling=0.05, flags=p.URDF_IGNORE_COLLISION_SHAPES)
                break

        return gifs

    # =====================internal collision check module=======================

    def _valid_state(self, state):
        return (state >= np.array(self.pose_range)[:, 0]).all() and \
               (state <= np.array(self.pose_range)[:, 1]).all()

    def _point_in_free_space(self, state):
        t0 = time()
        if not self._valid_state(state):
            return False

        for i in range(p.getNumJoints(self.kukaId)):
            p.resetJointState(self.kukaId, i, state[i])
        p.performCollisionDetection()
        if len(p.getContactPoints(self.kukaId)) == 0:
            self.collision_check_count += 1
            self.collision_time += time() - t0
            return True
        else:
            self.collision_point = state
            self.collision_check_count += 1
            self.collision_time += time() - t0
            return False

    def _state_fp(self, state):
        self.timer.start()
        free = self._point_in_free_space(state)
        self.timer.finish(Timer.VERTEX_CHECK)
        return free

    def _iterative_check_segment(self, left, right):
        if np.sum(np.abs(left - left)) > 0.1:
            mid = (left + right) / 2.0
            self.k += 1
            if not self._state_fp(mid):
                self.collision_point = mid
                return False
            return self._iterative_check_segment(left, mid) and self._iterative_check_segment(mid, right)

        return True

    def _edge_fp(self, state, new_state):
        self.timer.start()
        self.k = 0
        assert state.size == new_state.size

        if not self._valid_state(state) or not self._valid_state(new_state):
            self.timer.finish(Timer.EDGE_CHECK)
            return False
        if not self._point_in_free_space(state) or not self._point_in_free_space(new_state):
            self.timer.finish(Timer.EDGE_CHECK)
            return False

        disp = new_state - state

        d = self.distance(state, new_state)
        K = int(d / self.RRT_EPS)
        for k in range(0, K):
            c = state + k * 1. / K * disp
            if not self._point_in_free_space(c):
                self.timer.finish(Timer.EDGE_CHECK)
                return False
        self.timer.finish(Timer.EDGE_CHECK)
        return True
