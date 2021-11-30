from environment import MazeEnv, UR5Env, SnakeEnv, KukaEnv, Kuka2Env
import torch
from train_next import str2next
from str2name import str2name
from algorithm.tsa import NEXT_plan, RRTS_plan
from config import set_random_seed
from eval_gnn import explore
import numpy as np
from tqdm import tqdm
import pybullet as p
from time import time, sleep
from algorithm.bit_star import BITStar
from eval_gnn import path_cost
from environment.timer import Timer


def set_camera():
    p.resetDebugVisualizerCamera(
        cameraDistance=2.25,
        cameraYaw=-252,
        cameraPitch=-35,
        cameraTargetPosition=[0, 0, 0])


visualize = True
set_random_seed(3412)
env = KukaEnv(GUI=visualize)
set_camera()
pb_i = 2992
env.problems[2094][0][1][1][:2] = np.array([0, 0.5])

if visualize:
    pb = env.init_new_problem(pb_i)
    env.set_config(env.init_state)
    target_kukaId = p.loadURDF(env.kuka_file, [0, 0, 0], [0, 0, 0, 1], useFixedBase=True,
                               flags=p.URDF_IGNORE_COLLISION_SHAPES)
    env.set_config(env.goal_state, target_kukaId)
    for i in range(100):
        sleep(0.1)
        p.performCollisionDetection()
    input()

# '-------------------------------------------------BIT*----------------------------------------------'

# set_random_seed(3412)
# env.init_new_problem(pb_i)
# bit_star = BITStar(env, batch_size=500, T=1002, timer=env.timer)
# _, _, c, length, _, t = bit_star.plan(float('inf'), 0, 180)
# print('BIT*', t, env.collision_time, c, path_cost(bit_star.get_best_path()))
# print('VC: ', np.sum([l[1]-l[0] for l in env.timer.log if l[2]==Timer.VERTEX_CHECK]))
# print('VC len: ', len([l[1]-l[0] for l in env.timer.log if l[2]==Timer.VERTEX_CHECK]))
# print('EC: ', np.sum([l[1]-l[0] for l in env.timer.log if l[2]==Timer.EDGE_CHECK]))
# print('EC len: ', len([l[1]-l[0] for l in env.timer.log if l[2]==Timer.EDGE_CHECK]))
# print('HEAP: ', np.sum([l[1]-l[0] for l in env.timer.log if l[2]==Timer.HEAP]))
# print('EXPAND: ', np.sum([l[1]-l[0] for l in env.timer.log if l[2]==Timer.EXPAND]))
# print('NN: ', np.sum([l[1]-l[0] for l in env.timer.log if l[2]==Timer.NN]))
# if visualize:
#     set_random_seed(3412)
#     set_camera()
#     env.plot(bit_star.get_best_path())
#     input()
#     #
#     # import pybullet as p
#     # from time import sleep
#     # for i in range(1000):
#     #     sleep(0.1)
#     #     p.performCollisionDetection()
    
# '-------------------------------------------------GNN----------------------------------------------'

# set_camera()
# set_random_seed(3412)
# _, model_explore, _, model_smooth, _ = str2name(str(env), load=True)
# env.init_new_problem(pb_i)
# set_camera()
# c0 = env.collision_check_count
# set_random_seed(1234)
# result = explore(env, model_explore, model_smooth, t_max=1002, batch=500, k=30)
# print('GNN', result['total'], result['total_explore'], env.collision_time, env.collision_check_count-c0, result['forward'], path_cost(result['smooth_path']))
# print(len(result['explored']))
# # nodes = result['data'].y[result['explored']].data.cpu().numpy()
# if visualize:
#     pass
#     # for i in range(len(nodes)):
#     #     new_snake = env.create_snake(phantom=True)
#     #     env.set_config(nodes[i], new_snake)
#     set_random_seed(3412)
#     set_camera()
#     env.plot(result['smooth_path'])
#     input()

# '-------------------------------------------------RRT*----------------------------------------------'

# set_camera()
# set_random_seed(3412)
# pb = env.init_new_problem(pb_i)
# c0 = env.collision_check_count
# t0 = time()
# solution = RRTS_plan(
#     env=env,
#     T=40000,
#     stop_when_success=True,
# )
# print('RRT*', time()-t0, env.collision_time, env.collision_check_count-c0)
# if visualize:
# #     p.resetDebugVisualizerCamera(
# #         cameraDistance=2.25,
# #         cameraYaw=-325,
# #         cameraPitch=-32,
# #         cameraTargetPosition=[0, 0, 0])
#     for i in range(0, len(solution[0].non_terminal_states), 5):
#         if i % 100 == 0:
#             new_kuka = p.loadURDF(env.kuka_file, [0, 0, 0], [0, 0, 0, 1], useFixedBase=True,
#                                   flags=p.URDF_IGNORE_COLLISION_SHAPES)
#             for data in p.getVisualShapeData(new_kuka):
#                 color = list(data[-1])
#                 color[-1] = 0.5
#                 p.changeVisualShape(new_kuka, data[1], rgbaColor=color)
#         env.set_config(solution[0].non_terminal_states[i], new_kuka)
#         new_pos = p.getLinkState(new_kuka, env.kukaEndEffectorIndex)[0]
#         p.loadURDF("sphere2red.urdf", new_pos, globalScaling=0.05, flags=p.URDF_IGNORE_COLLISION_SHAPES)
#     input()

'-------------------------------------------------NEXT----------------------------------------------'

set_camera()
set_random_seed(3412)
UCB_type = 'kde'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cuda = True if torch.cuda.is_available() else False
model, model_path = str2next(str(env), env)
model.net.load_state_dict(torch.load(model_path, map_location=device))
set_random_seed(3412)
pb = env.init_new_problem(pb_i)
model.set_problem(env.get_problem())
t0 = time()
c0 = env.collision_check_count
solution = NEXT_plan(
    env=env,
    model=model,
    T=5000,
    g_explore_eps=0.1,
    stop_when_success=True,
    UCB_type=UCB_type
)
print('NEXT', time()-t0, env.collision_time, env.collision_check_count-c0)
if visualize:
    set_camera()
    set_random_seed(3412)
    # env.plot(solution[0].path()[0])
    for i in range(0, len(solution[0].non_terminal_states), 5):
        if i % 100 == 0:
            new_kuka = p.loadURDF(env.kuka_file, [0, 0, 0], [0, 0, 0, 1], useFixedBase=True,
                                  flags=p.URDF_IGNORE_COLLISION_SHAPES)
            for data in p.getVisualShapeData(new_kuka):
                color = list(data[-1])
                color[-1] = 0.5
                p.changeVisualShape(new_kuka, data[1], rgbaColor=color)
        env.set_config(solution[0].non_terminal_states[i], new_kuka)
        new_pos = p.getLinkState(new_kuka, env.kukaEndEffectorIndex)[0]
        p.loadURDF("sphere2red.urdf", new_pos, globalScaling=0.05, flags=p.URDF_IGNORE_COLLISION_SHAPES)
    input()
