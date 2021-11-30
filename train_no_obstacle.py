from train_explorer import train_explorer
from train_smoother import train_smoother
import torch
from environment import MazeEnv, KukaEnv, strs, SnakeEnv, Kuka2Env, UR5Env
from str2name import str2name

for str in ['ur5']:
    epoch = 2000
    env, model_explore, model_explore_path, model_smooth, model_smooth_path, data_path = str2name(str=str, get_data=True, use_obstacle=False)
    train_explorer(epoch, data_path, model_explore, model_explore_path, env, use_obstacle=False)

#     if torch.cuda.is_available():
#         from google.colab import files
#         files.download(model_explore_path)


from eval_gnn import eval_gnn
import numpy as np
import pickle

env_names = ['Maze_2D_Easy', 'Maze_2D_Normal', 'Maze_2D_Hard', 'Kuka_7D', 'snake7', 'ur5', 'Kuka_13D', 'Kuka_14D']
envs = [
    MazeEnv(dim=2, map_file='maze_files/mazes_easy.npz'),
    MazeEnv(dim=2, map_file='maze_files/mazes_normal.npz'),
    MazeEnv(dim=2, map_file='maze_files/mazes_hard.npz'),
    KukaEnv(),
    SnakeEnv(),
    UR5Env(),
    KukaEnv(kuka_file="kuka_iiwa/model_3.urdf", map_file="maze_files/kukas_13_3000.pkl"),
    Kuka2Env(),
    ]
indexeses = [np.arange(1000), np.arange(1000), np.arange(1000), np.arange(2000, 3000),
             np.arange(2000, 3000), np.arange(2000, 3000), np.arange(2000, 3000)]
seeds = [1234, 2341, 3412, 4123]
methods = [eval_gnn]
method_names = ['GNN_pure']
result_total = {}


for env_name, env, indexes in zip(env_names, envs, indexeses):
    for seed in seeds:
        results = []

        print(env_name, 'GNN_pure', seed)
        env, model_explore, model_explore_path, model_smooth, model_smooth_path = str2name(env.__str__(), get_data=False, use_obstacle=False)
        model_explore.load_state_dict(torch.load(model_explore_path, map_location=torch.device("cpu")))
        model_smooth.load_state_dict(torch.load(model_smooth_path, map_location=torch.device("cpu")))
        result = eval_gnn(env.__str__(), seed, env, indexes, model=model_explore, model_s=model_smooth, use_tqdm=True)
        results.append(result)
        result_total[env_name, 'GNN_pure', str(seed)] = result
        pickle.dump(result_total, open("data/pure.p", "wb"))

    print(env_name, 'GNN_pure', 'Avg')
    print('success rate:', np.mean([result[0] for result in results]))
    print('collision check: %.2f' % np.mean([result[1] for result in results]))
    print('running time: %.2f' % np.mean([result[2] for result in results]))
    print('path cost: %.2f' % np.mean([result[3] for result in results]))
    print('total time: %.2f' % np.mean([result[4] for result in results]))
    print('')
    result_total[env_name, 'GNN_pure', 'Avg'] = tuple([np.mean([result[i] for result in results]) for i in range(5)])
    pickle.dump(result_total, open("data/pure.p", "wb"))
pickle.dump(result_total, open("data/pure.p", "wb"))
