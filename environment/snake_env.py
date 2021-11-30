import numpy as np
import pybullet as p
from time import sleep
import pybullet_data
import pickle
from copy import deepcopy
import transforms3d

from environment.timer import Timer
from time import time

class SnakeEnv:
    '''
    Interface class for maze environment
    '''

    RRT_EPS = 0.1
    voxel_r = 0.1
    height = 0.5

    def __init__(self, map_file='maze_files/snakes_15_2_3000.npz', GUI=False):
        # print("Initializing environment...")

        with np.load(map_file) as f:
            self.maps = f['maps']
            self.init_states = f['init_states']
            self.goal_states = f['goal_states']

        self.timer = Timer()
        self.dim = 2
        self.config_dim = 7  # first two is x, y for base position; later five is the 5 DoF snake
        self.collision_check_count = 0

        self.size = self.maps.shape[0]
        self.width = self.maps.shape[1]
        self.episode_i = 0
        self.order = list(range(self.size))
        self.collision_point = None

        if GUI:
            p.connect(p.GUI, options='--background_color_red=1.0 --background_color_green=1.0 --background_color_blue=1.0')
        else:
            p.connect(p.DIRECT)

        p.resetDebugVisualizerCamera(
            cameraDistance=3,
            cameraYaw=0,
            cameraPitch=-89.8,
            cameraTargetPosition=[0, 0, 10])

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, lightPosition = [0, 0, 0.1])

        self.pose_range = [(-9, 9), (-9, 9)] + [(-np.pi, np.pi) for _ in range(5)]
        self.bound = np.array(self.pose_range).T.reshape(-1)

        p.setGravity(0, 0, -10)
        p.setRealTimeSimulation(0)

    def __str__(self):
        return 'snake'+str(self.config_dim)

    def create_maze(self, map):
        self.mazeIds = []
        numHeightfieldRows = 30
        numHeightfieldColumns = 30
        for j in range(len(map)):
            for i in range(len(map[0])):
                if map[i, j]:
                    self.mazeIds.append(self.create_voxel([0.7, 0.7, 1], [1.4*i-10.5, 1.4*j-10.5, 0]))
        return self.mazeIds

    def create_voxel(self, halfExtents, basePosition):
        groundColId = p.createCollisionShape(p.GEOM_BOX, halfExtents=halfExtents)
        groundVisID = p.createVisualShape(shapeType=p.GEOM_BOX,
                                          rgbaColor=[0.5, 0.5, 0.5, 1],
                                          specularColor=[0.4, .4, 0],
                                          halfExtents=halfExtents)
        groundId = p.createMultiBody(baseMass=0,
                                     baseCollisionShapeIndex=groundColId,
                                     baseVisualShapeIndex=groundVisID,
                                     basePosition=basePosition)
        # textureId = p.loadTexture("environment/red.png")
        # p.changeVisualShape(groundId, -1, textureUniqueId=textureId)
        return groundId

    def create_snake(self, phantom=False):
        sphereRadius = 0.25
        alpha = 0.5 if phantom else 1.
        snakeId = p.loadURDF('environment/snake.urdf', useFixedBase=False, flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)
        p.resetBasePositionAndOrientation(snakeId, [0, 0, SnakeEnv.height], [0, 0, 0, 1])

        # anistropicFriction = [1, 0.01, 0.01]
        # p.changeDynamics(sphereUid, -1, lateralFriction=2, anisotropicFriction=anistropicFriction)
        # p.getNumJoints(sphereUid)
        # for i in range(p.getNumJoints(sphereUid)):
        #     p.getJointInfo(sphereUid, i)
        #     p.changeDynamics(sphereUid, i, lateralFriction=2, anisotropicFriction=anistropicFriction)

        # pose_range = [(p.getJointInfo(sphereUid, jointId)[8], p.getJointInfo(sphereUid, jointId)[9]) for jointId in
        #               range(p.getNumJoints(sphereUid))]

        red = [0.95, 0.1, 0.1, 1]
        green = [0.1, 0.8, 0.1, 1]
        yellow = [1.0, 0.8, 0., 1]
        blue = [0.3, 0.3, 0.8, 1]
        colors = [red, blue, green, yellow]
        for i, data in enumerate(p.getVisualShapeData(snakeId, -1)):
            color = colors[i % 4]
            p.changeVisualShape(snakeId, i - 1, rgbaColor=[color[0], color[1], color[2], alpha])

        if phantom:
            for id in list(range(p.getNumJoints(snakeId))) + [-1]:
                p.setCollisionFilterGroupMask(snakeId, id, 0, 0)

        return snakeId

    def set_config(self, config, snakeId=None, sphereRadius=0.25):
        if snakeId is None:
            snakeId = self.snakeId
        mazeIds = self.mazeIds
        p.resetBaseVelocity(snakeId, [0, 0, 0], [0, 0, 0])

        quat = transforms3d.euler.euler2quat(0, 0, config[3])
        p.resetBasePositionAndOrientation(snakeId, list(config[:2]) + [SnakeEnv.height], np.concatenate((quat[1:], quat[0].reshape(1))))

        for i in range(len(config[3:])):
            p.resetJointState(snakeId, i * 2 + 1, config[i + 2])
        p.performCollisionDetection()

        if len(p.getContactPoints(snakeId)) == 0: #and len(p.getClosestPoints(snakeId, mazeId, 0.3)) == 0 and len(p.getContactPoints(snakeId, snakeId)) == 0:
            return True
        else:
            self.collision_point = config
            return False

    def reset(self, map):
        self.timer.start()
        p.resetSimulation()

        plane = p.createCollisionShape(p.GEOM_PLANE)
        p.createMultiBody(0, plane)

        self.map = map
        self.mazeIds = self.create_maze(map)
        self.snakeId = self.create_snake()

        self.obstacles = []

        for i in range(self.map.shape[0]):
            for j in range(self.map.shape[1]):
                if self.map[i, j] == 1:
                    self.obstacles.append((i, j))
        self.obstacles = np.array(self.obstacles) / self.map.shape[0] - 0.5

        p.setGravity(0, 0, -10)
        p.setRealTimeSimulation(0)

        self.timer.finish(Timer.CREATE)

        return self.snakeId

    def get_problem(self):
        problem = {
            "map": self.map,
            "init_state": self.init_state,
            "goal_state": self.goal_state
        }
        return problem
    
    def init_new_problem(self, index=None):
        '''
        :param i: the index of problem
        :param start: start config
        :param goal: goal config
        :return:
        '''
        if index is None:
            index = self.episode_i
        self.episode_i += 1
        self.episode_i = (self.episode_i) % len(self.order)
        
        self.collision_check_count = 0

        map = self.maps[index]
        self.reset(map)

        self.collision_point = None

        self.map = map
        self.init_state = self.init_states[index]
        self.goal_state = self.goal_states[index]
        
        return self.get_problem()

    def set_random_init_goal(self):
        while True:
            points = self.sample_n_points(n=2)
            init, goal = points[0], points[1]
            if np.sum(np.abs(init - goal)) != 0:
                break
        self.init_state, self.goal_state = init, goal

    def get_robot_points(self, config):
        return deepcopy(config[:2])

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

    def plot(self, map, path, make_gif=False):
        # self.reset(map)
        path = np.array(path)
        self.set_config(path[0])

        # p.setGravity(0, 0, -10)
        # p.stepSimulation()

        # if make_gif:
        #     for _ in range(100):
        #         p.stepSimulation()
        #         sleep(0.1)

        gifs = []
        current_state_idx = 0

        new_snake = self.create_snake(phantom=False)
        self.set_config(path[-1], snakeId=new_snake)

        while True:
            current_state = path[current_state_idx]
            disp = path[current_state_idx + 1] - path[current_state_idx]

            d = self.distance(path[current_state_idx], path[current_state_idx + 1])
            K = int(d / 0.5)
            for k in range(0, K):
                new_snake = self.create_snake(phantom=True)
                c = path[current_state_idx] + k * 1. / K * disp
                self.set_config(c, snakeId=new_snake)
                # p.stepSimulation()
                if make_gif:
                    gifs.append(p.getCameraImage(width=1100, height=900, lightDirection=[1, 1, 1], shadow=1,
                                                 renderer=p.ER_BULLET_HARDWARE_OPENGL)[2])
            current_state_idx += 1
            if current_state_idx == (len(path)-1):
                break

        return gifs

    # =====================internal collision check module=======================

    def _valid_state(self, state):
        return (state >= np.array(self.pose_range)[:, 0]).all() and \
               (state <= np.array(self.pose_range)[:, 1]).all()

    def _point_in_free_space(self, state):
        if not self._valid_state(state):
            return False

        self.collision_check_count += 1
        free = self.set_config(state)
        return free

    def _state_fp(self, state):
        self.timer.start()
        free = self._point_in_free_space(state)
        self.timer.finish(Timer.VERTEX_CHECK)
        return free

    def _edge_fp(self, state, new_state):
        self.timer.start()
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


if __name__ == '__main__':
    from algorithm.bit_star import BITStar
    import os
    from PIL import Image

    os.chdir('../')

    with np.load('maze_files/mazes_15_2_3000.npz') as f:
        maps = f['maps']
        init_states = f['init_states']
        goal_states = f['goal_states']

    env = SnakeEnv(GUI=True)
    # env.reset(env.maps[100])
    # env.set_random_init_goal()
    env.init_new_problem(2008)
    # env.goal_state = [-2.81186462, 6.18925902, -1.16216249, 0.0629849, 2.45711669, -1.74097965, 2.94462461]
    # env.init_state = [-1.69683885, -5.92519854, 1.73319886, 0.92728558, -1.96924572, -1.74118831, 0.77791534]
    print(env.init_state, env.goal_state)

    env.init_state[:2] = np.array([7, -7])
    # for _ in range(100):
    #     env.set_config(env.init_state)
    #     p.stepSimulation()
    #     sleep(0.1)
    #     env.set_config(env.goal_state)
    #     p.stepSimulation()
    #     sleep(0.1)

    bit = BITStar(env, batch_size=50, T=10000,)
    _ = bit.plan(float('inf'), refine_time_budget=0, time_budget=300)
    print(bit.get_best_path())
    path = bit.get_best_path()
    # path = [(7.0,
    #   -7.0,
    #   -2.2316785026021484,
    #   -0.6982155427305496,
    #   -1.7026055686183108,
    #   -2.2750685595969387,
    #   -0.7935623193376258),
    #  (7.7971982154729425,
    #   -4.514027412003557,
    #   -1.1217781906536572,
    #   0.4499082566090169,
    #   -1.4788919906294569,
    #   -2.4888743743726893,
    #   2.7928726155421595),
    #  (5.842520872069766,
    #   -4.362698864373027,
    #   -0.3185010166263327,
    #   0.5231599462986178,
    #   -1.4462935072804026,
    #   -2.24708000468008,
    #   1.8186289509825428),
    #  (6.502127680614812,
    #   -5.587544135696805,
    #   1.1352833079352322,
    #   -0.5192452674063435,
    #   1.2079167102398252,
    #   1.586719422769188,
    #   -0.49158718152146497),
    #  (5.572290411778328,
    #   -3.3724499682792795,
    #   0.8532643394700927,
    #   0.4049983374684665,
    #   2.2172360257031434,
    #   -0.21298751749249156,
    #   -1.3544382700718522),
    #  (5.046989732086933,
    #   -0.012520824695647192,
    #   0.7196129512917149,
    #   -1.05229647244001,
    #   2.49405013825964,
    #   -2.103347233046974,
    #   -2.7448071972824333),
    #  (2.8495166306638957,
    #   -0.940439037061088,
    #   0.5040835825076622,
    #   0.5490898962805182,
    #   -1.7360726304016523,
    #   -0.6091793128013903,
    #   -0.6113870145576041),
    #  (1.0423233993437933,
    #   0.13056787118757818,
    #   -1.035404830361883,
    #   1.379828406705979,
    #   0.26370480058915424,
    #   -1.8315900247194787,
    #   -2.8223281547104664),
    #  (-0.7480666291196112,
    #   2.1142989419577347,
    #   0.6880977083066524,
    #   0.1949605139293853,
    #   0.16190605877502717,
    #   -2.3476258146522633,
    #   -0.0955135739073274),
    #  (0.8141519632456546,
    #   1.7878123511369317,
    #   -0.9767611825078442,
    #   1.1119283979302015,
    #   2.486850511633169,
    #   -0.30249913116393534,
    #   -0.9551073757129136),
    #  (-0.12175308116658279,
    #   5.139560882189393,
    #   0.14911268269471822,
    #   0.7353251906920013,
    #   2.1317307556553313,
    #   -1.6335431335958814,
    #   1.7973851739290678),
    #  (-2.4610480882568737,
    #   7.677772950167167,
    #   -0.29323111361631415,
    #   0.35202545145961617,
    #   1.2686291383017823,
    #   -0.3766528933564648,
    #   0.8908446788822824),
    #  (-3.985934567335537,
    #   7.9706649283711,
    #   -0.6955009432245691,
    #   -0.2574287272006788,
    #   -0.23494922561001053,
    #   0.5007743066400554,
    #   1.2388329136981415)]
    gifs = env.plot(env.map, path, make_gif=False)

    # print(np.sum([t[1] - t[0] for t in env.timer.log if t[2] == Timer.VERTEX_CHECK]))
    # print(np.sum([t[1] - t[0] for t in env.timer.log if t[2] == Timer.EDGE_CHECK]))
    # print(env.timer.log[-1][1] - env.timer.log[0][1])
    # print(env.timer.log[0][1] - env.timer.log[0][0], env.timer.log[0])
    # import pickle
    # pickle.dump(env.timer.log, open('data/timer.pkl', 'wb'))

    for _ in range(1000):
        p.performCollisionDetection()
        sleep(0.1)

    # # Setup the 4 dimensional array
    # a_frames = []
    # for im_frame in gifs:
    #     a_frames.append(np.asarray(im_frame))
    # a = np.stack(a_frames)
    #
    # print("Array shape:", a.shape)  # (31, 240, 320, 3)
    #
    # ims = [Image.fromarray(a_frame) for a_frame in a]
    # ims[0].save("snake.gif", save_all=True, append_images=ims[1:], loop=0, duration=1)
    # print('hello')
