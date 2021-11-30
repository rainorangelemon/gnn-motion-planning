from environment import MazeEnv, UR5Env, SnakeEnv, KukaEnv, Kuka2Env
import torch
from train_next import str2next
from str2name import str2name
from algorithm.tsa import NEXT_plan
from config import set_random_seed
from eval_gnn import explore
import numpy as np
from tqdm import tqdm
import pybullet as p


def set_camera():
    p.resetDebugVisualizerCamera(
        cameraDistance=1.21,
        cameraYaw=-562,
        cameraPitch=-40,
        cameraTargetPosition=[0, 0, 0.7])


set_random_seed(1234)
env = UR5Env(GUI=True)

UCB_type = 'kde'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cuda = True if torch.cuda.is_available() else False
model, model_path = str2next(str(env), env)
model.net.load_state_dict(torch.load(model_path, map_location=device))

pb_i = 2409
env.problems[pb_i][1] = np.array([2.505271885727191, -1.4526602255197456, -0.07454906993932209, 3.965407939765909-2*np.pi, 3.75139287706858, 1.3096146445390127])
env.problems[pb_i][2] = np.array([2.494185, -2.66182248, -0.56985172,  5.48891048-2*np.pi, -4.00185468+2*np.pi, 5.2398285-2*np.pi])
# env.problems[pb_i][0].pop(0)
# env.problems[pb_i][0].pop(1)
#
# pb = env.init_new_problem(pb_i)
# model.set_problem(env.get_problem())
# solution = NEXT_plan(
#     env=env,
#     model=model,
#     T=1000,
#     g_explore_eps=0.1,
#     stop_when_success=True,
#     UCB_type=UCB_type
# )
# print([j[0] for j in p.getJointStates(env.ur5, env.joints)])
# set_camera()
# set_random_seed(1234)
# env.plot(solution[0].path()[0])
# for i in range(0, len(solution[0].non_terminal_states)):
#     if i % 100 == 0:
#         new_ur5 = p.loadURDF("ur5/ur5.urdf", [0, 0, 0], [0, 0, 0, 1], useFixedBase=True,
#                              flags=p.URDF_IGNORE_COLLISION_SHAPES)
#         for data in p.getVisualShapeData(new_ur5):
#             color = list(data[-1])
#             color[-1] = 0
#             p.changeVisualShape(new_ur5, data[1], rgbaColor=color)
#     env.set_config(solution[0].non_terminal_states[i], new_ur5)
#     new_pos = p.getLinkState(new_ur5, env.tip_index)[0]
#     s = p.loadURDF("sphere2red.urdf", new_pos, globalScaling=0.05, flags=p.URDF_IGNORE_COLLISION_SHAPES)
#     p.changeVisualShape(s, -1, rgbaColor=[0.7, 0.7, 0.7, 0.7])
# import pybullet as p
# from time import sleep
# for i in range(1000):
#     sleep(0.1)
#     p.performCollisionDetection()


set_random_seed(1234)
_, model_explore, _, model_smooth, _ = str2name(str(env), load=True)
env.init_new_problem(pb_i)
result = explore(env, model_explore, model_smooth, t_max=1000, batch=50, k=30)
# nodes = data.y[explored].data.cpu().numpy()
nodes = result['data'].y.data.cpu().numpy()

set_camera()
set_random_seed(1234)
env.plot(result['smooth_path'])
for i in range(0, len(nodes)):
    if i % 100 == 0:
        new_ur5 = p.loadURDF("ur5/ur5.urdf", [0, 0, 0], [0, 0, 0, 1], useFixedBase=True,
                             flags=p.URDF_IGNORE_COLLISION_SHAPES)
        for data in p.getVisualShapeData(new_ur5):
            color = list(data[-1])
            color[-1] = 0
            p.changeVisualShape(new_ur5, data[1], rgbaColor=color)
    env.set_config(nodes[i], new_ur5)
    new_pos = p.getLinkState(new_ur5, env.tip_index)[0]
    s = p.loadURDF("sphere2red.urdf", new_pos, globalScaling=0.05, flags=p.URDF_IGNORE_COLLISION_SHAPES)
    if i not in result['explored']:
        p.changeVisualShape(s, -1, rgbaColor=[0.7, 0.7, 0.7, 0.7])
    else:
        p.changeVisualShape(s, -1, rgbaColor=[0, 0, 0.7, 0.7])

import pybullet as p
from time import sleep
for i in range(1000):
    sleep(0.1)
    p.performCollisionDetection()
