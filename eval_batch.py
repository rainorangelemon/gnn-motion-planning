from eval_gnn import eval_gnn
from eval_next import eval_next
from eval_bit import eval_bit
from eval_rrt import eval_rrt
import numpy as np
from environment import MazeEnv, KukaEnv, Kuka2Env
import pickle

env_names = ['Maze_2D_Hard', 'Kuka_7D']
envs = [
    MazeEnv(dim=2, map_file='maze_files/mazes_hard.npz'),
    KukaEnv(),
    ]
indexeses = [np.arange(1000), np.arange(2000, 3000)]
seeds = [1234]
methods = [eval_gnn]
t_maxs = [50, 100, 200, 300, 500, 1000]
result_total = {}


for env_name, env, indexes in zip(env_names, envs, indexeses):
    for t_max in t_maxs:
        results = []
        for seed in seeds:
            print(env_name, t_max, seed)
            result = eval_gnn(env.__str__(), seed, env, indexes, use_tqdm=True, batch=t_max, t_max=t_max)
            results.append(result)
            result_total[env_name, t_max, str(seed)] = result

            pickle.dump(result_total, open("data/batch.p", "wb"))

        print(env_name, t_max, 'Avg')
        print('success rate:', np.mean([result[0] for result in results]))
        print('collision check: %.2f' % np.mean([result[1] for result in results]))
        print('running time: %.2f' % np.mean([result[2] for result in results]))
        print('path cost: %.2f' % np.mean([result[3] for result in results]))
        print('total time: %.2f' % np.mean([result[4] for result in results]))
        print('')
        result_total[env_name, t_max, 'Avg'] = tuple([np.mean([result[i] for result in results]) for i in range(5)])

print(result_total)
