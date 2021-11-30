import numpy as np
from environment import MazeEnv, KukaEnv, Kuka2Env
from next_model import Model3D, Model2D
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


def str2next(str, env):
    cuda = True if torch.cuda.is_available() else False
    if str == 'maze2':
        model = Model2D(env=env, cuda=cuda, dim=env.config_dim)
        model_path = 'data/weights/next_2.pt'
    elif str == 'maze3':
        model = Model2D(env=env, cuda=cuda, dim=env.config_dim)
        model_path = 'data/weights/next_3.pt'
    elif str == 'snake7':
        model = Model2D(env=env, cuda=cuda, dim=env.config_dim)
        model_path = 'data/weights/next_snake.pt'
    elif str == 'ur5':
        model = Model3D(env=env, cuda=cuda, dim=env.config_dim, point_dim=3)
        model_path = 'data/weights/next_ur5.pt'        
    elif str == 'kuka7':
        model = Model3D(env=env, cuda=cuda, dim=env.config_dim, point_dim=3)
        model_path = 'data/weights/next_7.pt'
    elif str == 'kuka13':
        model = Model3D(env=env, cuda=cuda, dim=env.config_dim, point_dim=3)
        model_path = 'data/weights/next_13.pt'
    if str == 'kuka14':
        model = Model3D(env=env, cuda=cuda, dim=env.config_dim, point_dim=6)
        model_path = 'data/weights/next_14.pt'
    return model, model_path


def eval_next(str, seed, env, indexes, use_tqdm=False, t_max=1000, **kwargs):
    set_random_seed(seed)
    UCB_type = 'kde'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cuda = True if torch.cuda.is_available() else False

    model, model_path = str2next(str, env)
    model.net.load_state_dict(torch.load(model_path, map_location=device))
    
    next_solutions = []
    times = []
    explore_eps = 0.1
    pbar = tqdm(indexes) if use_tqdm else indexes
    for i in pbar:
        pb = env.init_new_problem(i)
        model.set_problem(pb)

        t0 = time()
        next_solutions.append(NEXT_plan(
            env=env,
            model=model,
            T=t_max,
            g_explore_eps=explore_eps,
            stop_when_success=True,
            UCB_type=UCB_type
        ))
        times.append(time()-t0)

    n_success = np.sum([solution[1] for solution in next_solutions])
    collision = np.mean([solution[0].cumulated_collision_checks[-1] - solution[0].cumulated_collision_checks[1] for solution in next_solutions])
    running_time = np.mean([t for t, solution in zip(times, next_solutions) if solution[1]])
    solution_cost = np.mean([solution[0].path_lengths[-1] for solution in next_solutions if solution[1]])
    total_time = sum(times)

    print('success rate:', n_success)
    print('collision check: %.2f' % collision)
    print('running time: %.2f' % running_time)
    print('path cost: %.2f' % solution_cost)
    print('total time: %.2f' % total_time)
    print('')

    return n_success, collision, running_time, solution_cost, total_time, [s[0].path()[0] for s in next_solutions]


if __name__ == '__main__':
    from environment import SnakeEnv
    env = SnakeEnv(GUI=True)

    # results = []
    # for seed in [1234, 2341, 3412, 4123]:
    #     print(str(env), 'NEXT', seed)
    #     result = eval_next(str(env), seed, env, indexes, use_tqdm=True)
    #     results.append(result)
    #     result_total[str(env), 'NEXT', str(seed)] = result
    #
    # print(str(env), 'NEXT', 'Avg')
    # print('success rate:', np.mean([result[0] for result in results]))
    # print('collision check: %.2f' % np.mean([result[1] for result in results]))
    # print('running time: %.2f' % np.mean([result[2] for result in results]))
    # print('path cost: %.2f' % np.mean([result[3] for result in results]))
    # print('total time: %.2f' % np.mean([result[4] for result in results]))
    # print('')
    # result_total[str(env), 'NEXT', 'Avg'] = tuple([np.mean([result[i] for result in results]) for i in range(5)])
    #
    # pickle.dump(result_total, open("data/snake.p", "wb"))

    set_random_seed(1234)
    UCB_type = 'kde'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cuda = True if torch.cuda.is_available() else False

    model, model_path = str2next(str(env), env)
    model.net.load_state_dict(torch.load(model_path, map_location=device))
    pbar = tqdm(np.arange(2000, 3000))
    i = 2008
    pb = env.init_new_problem(i)
    env.init_state[:2] = np.array([7, -7])
    model.set_problem(env.get_problem())

    t0 = time()
    solution = NEXT_plan(
        env=env,
        model=model,
        T=1000,
        g_explore_eps=0.1,
        stop_when_success=True,
        UCB_type=UCB_type
    )

    if not solution[1]:
        env.reset(env.map)
        env.set_config(solution[0].non_terminal_states[0])
        for i in range(0, len(solution[0].non_terminal_states), 10):
            new_snake = env.create_snake(phantom=True)
            env.set_config(solution[0].non_terminal_states[i], new_snake)

        print('hello')