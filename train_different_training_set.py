from train_explorer import train_explorer
from train_smoother import train_smoother
import torch
from environment import MazeEnv, KukaEnv
from str2name import str2name


env2d = MazeEnv(dim=2, map_file='maze_files/mazes_4000.npz')
env7d = KukaEnv(map_file='maze_files/kukas_7_4000.pkl')

for dim, env, data_path in zip([2, 7], [env2d, env7d], ['data/pkl/maze_prm_4000.pkl', 'data/pkl/kuka_prm_4000.pkl']):
    for ratio in [0.2, 0.5, 0.7, 2.0]:
        epoch = int(2000 * ratio)
        _, model_explore, model_explore_path, model_smooth, model_smooth_path = str2name(str=env.__str__())
        model_explore_path = model_explore_path.replace('.pt', '_%.1f.pt' % ratio)
        model_smooth_path = model_smooth_path.replace('.pt', '_%.1f.pt' % ratio)
        train_explorer(epoch, data_path, model_explore, model_explore_path, env)
        train_smoother(epoch, model_explore, model_smooth, model_smooth_path, env)

        if torch.cuda.is_available():
            from google.colab import files
            files.download(model_explore_path)
            files.download(model_smooth_path)


from eval_gnn import eval_gnn
import numpy as np
import pickle

env_names = ['Maze_2D_Hard', 'Kuka_7D']
envs = [
    MazeEnv(dim=2, map_file='maze_files/mazes_hard.npz'),
    KukaEnv(),
    ]
indexeses = [np.arange(1000), np.arange(2000, 3000)]
seed = 1234
methods = [eval_gnn]
method_names = ['GNN']
result_total = {}
keep = False
keep_point = ('Maze_2D_Hard', '0.2')


for env_name, env, indexes in zip(env_names, envs, indexeses):
    for ratio in [0.2, 0.5, 0.7, 1.0, 2.0]:
        results = []
        if not keep:
            if (env_name, str(ratio)) == keep_point:
                keep = True
            else:
                continue

        if ratio != 1.0:
            _, model_explore, model_explore_path, model_smooth, model_smooth_path = str2name(str=env.__str__())
            model_explore_path = model_explore_path.replace('.pt', '_%.1f.pt' % ratio)
            model_smooth_path = model_smooth_path.replace('.pt', '_%.1f.pt' % ratio)
            model_explore.load_state_dict(torch.load(model_explore_path, map_location=torch.device("cpu")))
            model_smooth.load_state_dict(torch.load(model_smooth_path, map_location=torch.device("cpu")))
        else:
            model_explore = None
            model_smooth = None

        print(env_name, ratio)
        result = eval_gnn(env.__str__(), seed, env, indexes, model=model_explore, model_s=model_smooth, use_tqdm=True)
        results.append(result)
        result_total[env_name, ratio] = result

        pickle.dump(result_total, open("data/train_size.p", "wb"))
