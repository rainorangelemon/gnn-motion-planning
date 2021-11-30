import numpy as np
from environment import MazeEnv, KukaEnv, Kuka2Env
from next_model import Model3D
from algorithm import NEXT_plan, RRTS_plan
from config import set_random_seed
from utils.plot import plot_edges as plot_tree
from tqdm import tqdm
from algorithm.bit_star import BITStar
import torch
from time import time


def tensor_to_np(tensor):
    return tensor.data.cpu().numpy()


def tuple_to_np(tuple_):
    return np.array(tuple_)


def eval_rrt(str, seed, env, indexes, use_tqdm=False, t_max=1000, **kwargs):
    set_random_seed(seed)
    UCB_type = 'kde'

    rrt_solutions = []
    times = []
    pbar = tqdm(indexes) if use_tqdm else indexes
    for i in pbar:
        env.init_new_problem(i)

        t0 = time()
        rrt_solutions.append(NEXT_plan(
            env=env,
            model=None,
            T=t_max,
            g_explore_eps=1.,
            stop_when_success=True,
            UCB_type=UCB_type
        ))
        times.append(time()-t0)

    n_success = np.sum([solution[1] for solution in rrt_solutions])
    collision = np.mean(
        [solution[0].cumulated_collision_checks[-1] - solution[0].cumulated_collision_checks[1] for solution in
         rrt_solutions])
    running_time = np.mean([t for t, solution in zip(times, rrt_solutions) if solution[1]])
    solution_cost = np.mean([solution[0].path_lengths[-1] for solution in rrt_solutions if solution[1]])
    total_time = sum(times)

    print('success rate:', n_success)
    print('collision check: %.2f' % collision)
    print('running time: %.2f' % running_time)
    print('path cost: %.2f' % solution_cost)
    print('total time: %.2f' % total_time)
    print('')

    return n_success, collision, running_time, solution_cost, total_time, [s[0].path()[0] for s in rrt_solutions]
