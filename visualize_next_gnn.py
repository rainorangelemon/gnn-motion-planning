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

'------------------------------------------------------ur5------------------------------------------------------'

# set_random_seed(1234)
# env = UR5Env(GUI=True)
#
# UCB_type = 'kde'
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# cuda = True if torch.cuda.is_available() else False
# model, model_path = str2next(str(env), env)
# model.net.load_state_dict(torch.load(model_path, map_location=device))
#
# pb_i = 2740
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
#
# for i in range(0, len(solution[0].non_terminal_states), 5):
#     if i % 100 == 0:
#         new_ur5 = p.loadURDF("ur5/ur5.urdf", [0, 0, 0], [0, 0, 0, 1], useFixedBase=True,
#                              flags=p.URDF_IGNORE_COLLISION_SHAPES)
#         for data in p.getVisualShapeData(new_ur5):
#             color = list(data[-1])
#             color[-1] = 0.5
#             p.changeVisualShape(new_ur5, data[1], rgbaColor=color)
#     env.set_config(solution[0].non_terminal_states[i], new_ur5)
#     new_pos = p.getLinkState(new_ur5, env.tip_index)[0]
#     p.loadURDF("sphere2red.urdf", new_pos, globalScaling=0.05, flags=p.URDF_IGNORE_COLLISION_SHAPES)
#
# import pybullet as p
# from time import sleep
# for i in range(1000):
#     sleep(0.1)
#     p.performCollisionDetection()


# set_random_seed(1234)
# _, model_explore, _, model_smooth, _ = str2name(str(env), load=True)
# env.init_new_problem(pb_i)
# c0, cost, data, explored, forward, success, t0, value, path, smooth_path = \
#     explore(env, model_explore, model_smooth, t_max=1000, batch=1000)
# nodes = data.y[explored].data.cpu().numpy()
# # for i in range(len(nodes)):
# #     new_snake = env.create_snake(phantom=True)
# #     env.set_config(nodes[i], new_snake)
# set_random_seed(1234)
# env.plot(smooth_path)
#
# import pybullet as p
# from time import sleep
# for i in range(1000):
#     sleep(0.1)
#     p.performCollisionDetection()

'------------------------------------------------------snake------------------------------------------------------'

# set_random_seed(1234)
# env = SnakeEnv(GUI=True)
# pb_i = 2008
#
# UCB_type = 'kde'
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# cuda = True if torch.cuda.is_available() else False
#
# model, model_path = str2next(str(env), env)
# model.net.load_state_dict(torch.load(model_path, map_location=device))
# pb = env.init_new_problem(pb_i)
# env.init_state[:2] = np.array([7, -7])
# model.set_problem(env.get_problem())
#
# solution = NEXT_plan(
#     env=env,
#     model=model,
#     T=1000,
#     g_explore_eps=0.5,
#     stop_when_success=True,
#     UCB_type=UCB_type
# )
#
# if not solution[1]:
#     env.reset(env.map)
#     env.set_config(solution[0].non_terminal_states[0])
#     for i in range(0, len(solution[0].non_terminal_states), 10):
#         new_snake = env.create_snake(phantom=True)
#         env.set_config(solution[0].non_terminal_states[i], new_snake)
#
# import pybullet as p
# from time import sleep
# for i in range(1000):
#     sleep(0.1)
#     p.performCollisionDetection()
#
# set_random_seed(1234)
# _, model_explore, _, model_smooth, _ = str2name(str(env), load=True)
# env.init_new_problem(pb_i)
# env.init_state[:2] = np.array([7, -7])
# c0, cost, data, explored, forward, success, t0, value, path, smooth_path = explore(env, model_explore, model_smooth)
# env.reset(env.map)
# env.set_config(env.init_state)
# env.init_state[:2] = np.array([7, -7])
# env.set_config(env.init_state)
# nodes = data.y[explored].data.cpu().numpy()
# # for i in range(len(nodes)):
# #     new_snake = env.create_snake(phantom=True)
# #     env.set_config(nodes[i], new_snake)
# env.plot(env.map, smooth_path)
#
# import pybullet as p
# from time import sleep
# for i in range(1000):
#     sleep(0.1)
#     p.performCollisionDetection()

'------------------------------------------------------kuka7------------------------------------------------------'

set_random_seed(3412)
env = KukaEnv(GUI=True)
p.resetDebugVisualizerCamera(
    cameraDistance=2.25,
    cameraYaw=-325,
    cameraPitch=-32,
    cameraTargetPosition=[0, 0, 0])
pb_i = 2070
env.problems[pb_i][0].pop(3)

pb = env.init_new_problem(pb_i)
env.set_config(env.init_state)
target_kukaId = p.loadURDF(env.kuka_file, [0, 0, 0], [0, 0, 0, 1], useFixedBase=True,
                           flags=p.URDF_IGNORE_COLLISION_SHAPES)
env.set_config(env.goal_state, target_kukaId)
from time import sleep
for i in range(1000):
    sleep(0.1)
    p.performCollisionDetection()

# UCB_type = 'kde'
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# cuda = True if torch.cuda.is_available() else False
# model, model_path = str2next(str(env), env)
# model.net.load_state_dict(torch.load(model_path, map_location=device))
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
# p.resetDebugVisualizerCamera(
#     cameraDistance=2.25,
#     cameraYaw=-325,
#     cameraPitch=-32,
#     cameraTargetPosition=[0, 0, 0])
# for i in range(0, len(solution[0].non_terminal_states), 5):
#     if i % 100 == 0:
#         new_kuka = p.loadURDF(env.kuka_file, [0, 0, 0], [0, 0, 0, 1], useFixedBase=True,
#                               flags=p.URDF_IGNORE_COLLISION_SHAPES)
#         for data in p.getVisualShapeData(new_kuka):
#             color = list(data[-1])
#             color[-1] = 0.5
#             p.changeVisualShape(new_kuka, data[1], rgbaColor=color)
#     env.set_config(solution[0].non_terminal_states[i], new_kuka)
#     new_pos = p.getLinkState(new_kuka, env.kukaEndEffectorIndex)[0]
#     p.loadURDF("sphere2red.urdf", new_pos, globalScaling=0.05, flags=p.URDF_IGNORE_COLLISION_SHAPES)
#
# import pybullet as p
# from time import sleep
# for i in range(1000):
#     sleep(0.1)
#     p.performCollisionDetection()


# set_random_seed(3412)
# _, model_explore, _, model_smooth, _ = str2name(str(env), load=True)
# env.init_new_problem(pb_i)
# c0, cost, data, explored, forward, success, t0, value, path, smooth_path = \
#     explore(env, model_explore, model_smooth, t_max=1000, batch=1000)
# nodes = data.y[explored].data.cpu().numpy()
# # for i in range(len(nodes)):
# #     new_snake = env.create_snake(phantom=True)
# #     env.set_config(nodes[i], new_snake)
# set_random_seed(3412)
# p.resetDebugVisualizerCamera(
#     cameraDistance=2.25,
#     cameraYaw=-325,
#     cameraPitch=-32,
#     cameraTargetPosition=[0, 0, 0])
# env.plot(smooth_path)
#
# import pybullet as p
# from time import sleep
# for i in range(1000):
#     sleep(0.1)
#     p.performCollisionDetection()

'------------------------------------------------------kuka14------------------------------------------------------'

set_random_seed(1234)
env = Kuka2Env(GUI=True)
pb_i = 1839

# UCB_type = 'kde'
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# cuda = True if torch.cuda.is_available() else False
# model, model_path = str2next(str(env), env)
# model.net.load_state_dict(torch.load(model_path, map_location=device))
#
# pb = env.init_new_problem(pb_i)
# model.set_problem(env.get_problem())
#
# solution = NEXT_plan(
#     env=env,
#     model=model,
#     T=1000,
#     g_explore_eps=0.1,
#     stop_when_success=True,
#     UCB_type=UCB_type
# )
#
# for i in range(0, len(solution[0].non_terminal_states)):
#     if i % 10 == 0:
#         new_kuka = p.loadURDF(env.kuka_file, [-0.5, 0, 0], [0, 0, 0, 1], useFixedBase=True,
#                               flags=p.URDF_IGNORE_COLLISION_SHAPES)
#         new_kuka2 = p.loadURDF(env.kuka_file, [0.5, 0, 0], [0, 0, 0, 1], useFixedBase=True,
#                                flags=p.URDF_IGNORE_COLLISION_SHAPES)
#
#         for data in p.getVisualShapeData(env.kukaId):
#             color = list(data[-1])
#             color[-1] = 0.5
#             p.changeVisualShape(new_kuka, data[1], rgbaColor=color)
#
#         for data in p.getVisualShapeData(env.kukaId2):
#             color = list(data[-1])
#             color[-1] = 0.5
#             p.changeVisualShape(new_kuka2, data[1], rgbaColor=color)
#
#     env.set_config(solution[0].non_terminal_states[i], new_kuka, new_kuka2)
#     new_pos1 = p.getLinkState(new_kuka, env.kukaEndEffectorIndex)[0]
#     new_pos2 = p.getLinkState(new_kuka2, env.kukaEndEffectorIndex)[0]
#     p.loadURDF("sphere2red.urdf", new_pos1, globalScaling=0.05, flags=p.URDF_IGNORE_COLLISION_SHAPES)
#     p.loadURDF("sphere2red.urdf", new_pos2, globalScaling=0.05, flags=p.URDF_IGNORE_COLLISION_SHAPES)
#
# import pybullet as p
# from time import sleep
# for i in range(1000):
#     sleep(0.1)
#     p.performCollisionDetection()


# set_random_seed(1234)
# _, model_explore, _, model_smooth, _ = str2name(str(env), load=True)
# env.init_new_problem(pb_i)
# c0, cost, data, explored, forward, success, t0, value, path, smooth_path = \
#     explore(env, model_explore, model_smooth, t_max=1000, batch=1000)
# nodes = data.y[explored].data.cpu().numpy()
# # for i in range(len(nodes)):
# #     new_snake = env.create_snake(phantom=True)
# #     env.set_config(nodes[i], new_snake)
# set_random_seed(1234)
# env.plot(smooth_path)
#
# import pybullet as p
# from time import sleep
# for i in range(1000):
#     sleep(0.1)
#     p.performCollisionDetection()


print('hello')
