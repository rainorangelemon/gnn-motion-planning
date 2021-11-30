import torch
import numpy as np
from config import set_random_seed
from tqdm import tqdm as tqdm
from tensorboardX import SummaryWriter
from eval_gnn import explore
from smoother import joint_smoother_ratio
from torch_geometric.utils import add_self_loops
from str2name import str2name
from copy import deepcopy


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def obs_data(config_size, obstacles, free, collided):
    if not len(free):
        free.append([0. for _ in range(config_size)])
    if not len(collided):
        collided.append([0. for _ in range(config_size)])
    data = DotDict({
        'free': free[:500],
        'collided': collided[:500],
        'obstacles': obstacles,
    })
    return data


def train(env, replay, model, optimizer, batch_idx=None):
    if len(replay) <= 8:
        return 0.

    optimizer.zero_grad()

    loss = 0.
    if batch_idx is None:
        batch_idx = np.random.choice(len(replay), size=8, replace=False)
    for idx in batch_idx:
        env_id, path_origin, path_smooth, obstacles, free, collided = replay[idx]
        data = obs_data(model.config_size, obstacles, free, collided)
        data = DotDict({k: torch.FloatTensor(v).to(device) for k, v in data.items()})
        data.path = torch.FloatTensor(path_origin).to(device)
        data.edge_index = torch.cat((torch.arange(1, len(path_origin)).reshape(1, -1),
                                     torch.arange(0, len(path_origin)-1).reshape(1, -1)), dim=0)
        data.edge_index = torch.cat((data.edge_index, data.edge_index.flip(0)), dim=-1)
        data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=len(data.path))
        data.edge_index = data.edge_index.to(device)

        path_pred = model(**data, loop=np.random.randint(1, 10))

        loss += torch.nn.MSELoss()(torch.FloatTensor(path_smooth).to(device)[1:-1], path_pred[1:-1])
    
    optimizer.zero_grad()
    (loss/len(batch_idx)).backward()
    optimizer.step()

    return loss


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_smoother(epoch, model_explore, model, model_path, env, data_iter=3):
    _, model_explore, model_explore_path, _, _ = str2name(str=env.__str__())
    model_explore.load_state_dict(torch.load(model_explore_path, map_location=torch.device("cpu")))
    model_explore.to(device)    
    
    writer = SummaryWriter()
    INFINITY = float('inf')

    # env = KukaEnv(kuka_file="kuka_iiwa/model_3.urdf", map_file="maze_files/kukas_13_3000.pkl")
    set_random_seed(1234)
    train_iter=20

    replay = []
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=0)
    optimizer.zero_grad()

    for iter_i in range(data_iter):

        indexes = np.random.permutation(epoch)
        pbar = tqdm(indexes)

        for index in pbar:
            env.init_new_problem(index)
            if iter_i != 0:
                env.set_random_init_goal()

            try:
                path, free, collided = explore(env, model_explore, model, smooth=False)
                if len(path) > 2:
                    path_smooth = joint_smoother_ratio([tuple(node) for node in path], env, iter=5)
                    replay.append((index, path, path_smooth, deepcopy(env.obstacles), free, collided))

            except Exception:
                continue
                
#     torch.save(replay, 'data/pkl/smooth_14.p')

    for iter_i in range(train_iter):

        indexes = np.random.permutation(len(replay))
        pbar = tqdm(np.arange(len(replay)))
        losses = []

        for index in pbar:

            if index % 8 != 0:
                continue

            try:
                loss = train(env, replay, model, optimizer, batch_idx=indexes[index:(index+8)])
            except:
                print(indexes[index:(index+8)])

            losses.append(float(loss.detach().cpu()))

            pbar.set_description("loss: %.5f" % np.mean(losses))
            writer.add_scalar('loss', loss)

        torch.save(model.state_dict(), model_path)
        scheduler.step(np.mean(losses))

    torch.save(model.state_dict(), model_path)
    writer.close()
    
    return


def train_env(str_):
    import os
    os.environ['CUDA_LAUNCH_BLOCKING']='1'

    from train_explorer import train_explorer
    from importlib import reload
    import torch
    from environment import MazeEnv, KukaEnv, SnakeEnv, UR5Env, Kuka2Env
    from str2name import str2name
    from copy import deepcopy

    epoch = 2000
    env, model_explore, model_explore_path, model_smooth, model_smooth_path = str2name(str=str_)
    model_explore.load_state_dict(torch.load(model_explore_path, map_location=torch.device("cpu")))
    model_explore.to(device)
    model_smooth_path = model_smooth_path.replace('.pt', 'v3.pt')
    model = model_smooth
    model_path = model_smooth_path
    # model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.to(device)
    writer = SummaryWriter()
    INFINITY = float('inf')
    
    set_random_seed(1234)

    replay = []
    for iter_i in range(3):

        indexes = np.random.permutation(epoch)
        pbar = tqdm(indexes)

        for index in pbar:
            env.init_new_problem(index)
            if iter_i != 0:
                env.set_random_init_goal()

            try:
                path, free, collided = explore(env, model_explore, model, smooth=False)
                if len(path) > 2:
                    path_smooth = joint_smoother_ratio([tuple(node) for node in path], env, iter=5)
                    replay.append((index, path, path_smooth, deepcopy(env.obstacles), free, collided))

            except Exception as e:
                continue
                
    import pickle
    pickle.dump([(r[0], r[1], r[2]) for r in replay], open("data/oracle_{0:s}.p".format(str_), "wb"))
    
    from model_smoother import ModelSmoother
    model = ModelSmoother(workspace_size=env.dim, config_size=env.config_dim, embed_size=128, obs_size=6, scale=np.max(env.bound)).to(device)
    _ = model.to(device)

    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=0)
    optimizer.zero_grad()

    train_iter=20
    loss_min = float('inf')

    for iter_i in range(train_iter):

        indexes = np.random.permutation(len(replay))
        pbar = tqdm(np.arange(len(replay)))
        losses = []

        for index in pbar:

            if index % 8 != 0:
                continue

            loss = train(env, replay, model, optimizer, batch_idx=indexes[index:(index+8)])

            losses.append(float(loss))

            pbar.set_description("loss: %.5f" % np.mean(losses))
            writer.add_scalar('loss', loss)

        scheduler.step(np.mean(losses))

        if np.mean(losses) < loss_min:
            loss_min = np.mean(losses)
            torch.save(model.state_dict(), model_path)        

    writer.close()
    

if __name__ == '__main__':
    for str_ in ['snake7', 'kuka13']:
        train_env(str_)

    