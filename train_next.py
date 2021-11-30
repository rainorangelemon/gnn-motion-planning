import numpy as np
from environment import MazeEnv, KukaEnv, Kuka2Env, SnakeEnv
from next_model import Model3D
from algorithm import NEXT_plan, RRTS_plan
from config import set_random_seed
from utils.plot import plot_edges as plot_tree
from tqdm import tqdm
from algorithm.bit_star import BITStar
import torch
from str2name import str2name
from eval_next import str2next
from tensorboardX import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def tensor_to_np(tensor):
    return tensor.data.cpu().numpy()


def tuple_to_np(tuple_):
    return np.array(tuple_)


def get_label(path, env):
    path = np.array(path)
    path_cost = [0.]
    action =[]
    for prev, next in zip(path[:-1, :], path[1:, :]):
        edge_cost = np.linalg.norm(next-prev)
        path_cost.append(path_cost[-1]+edge_cost)
        if edge_cost > env.RRT_EPS:
            action.append(env.interpolate(prev, next, env.RRT_EPS/edge_cost)-prev)
        else:
            action.append(next-prev)
    action.append(path[-1]*0.)
    for i, cost in enumerate(path_cost):
        path_cost[i] = (cost - path_cost[-1])
    return action, path_cost


def train(model, optimizer, replay, env, pbar, writer=None, L=10):
    loss = 0.
    for _ in range(L):
        optimizer.zero_grad()
        indexes = np.random.permutation(len(replay))
        for batch_i, index in enumerate(indexes):
            i, path = replay[index]
            pb = env.init_new_problem(index=i)
            model.set_problem(pb)
            action, value = get_label(path, env)
            action_pred, value_pred = model.net_forward(np.array(path), use_np=False)
            value_loss = torch.nn.MSELoss()(torch.FloatTensor(value).to(device), value_pred)
            action_loss = torch.nn.MSELoss()(torch.FloatTensor(action).to(device), action_pred)
            loss = loss + value_loss + action_loss
            if writer is not None:
                writer.add_scalar('train/value', value_loss, writer.action_step)
                writer.add_scalar('train/action', action_loss, writer.action_step)
                writer.action_step += 1
            if batch_i % 8 == 7:
                pbar.set_description("total %.2f, value %.2f, policy %.2f" \
                     % (loss / 8., value_loss, action_loss))
                writer.add_scalar('train/total', loss, writer.total_step)
                writer.total_step += 1
                optimizer.zero_grad()
                (loss / 8.).backward()
                loss = 0.
                optimizer.step()


def train_env(str):
    writer = SummaryWriter()
    writer.action_step = 0
    writer.total_step = 0    
    
    set_random_seed(1234)
    UCB_type = 'kde';
    env, _, _, _, _ = str2name(str)
    model, model_path = str2next(str, env)
    cuda = True if torch.cuda.is_available() else False
    
    optimizer = torch.optim.Adam(model.net.parameters(), lr=1e-3)

    replay = []

    explore_eps = 1.0
    pbar = tqdm(range(2000))
    for i in pbar:
        pb = env.init_new_problem(i)
        set_random_seed(i)
        model.set_problem(pb)

        search_tree, success, n_samples = NEXT_plan(
            env=env,
            model=model,
            T=1000,
            g_explore_eps=explore_eps,
            stop_when_success=True,
            UCB_type=UCB_type
        )

        if success:
            replay.append((i, search_tree.path()[0]))
        else:
            BIT = BITStar(env, T=float('INF'), batch_size=50)
            g_score = BIT.plan(float('INF'), time_budget=60, refine_time_budget=0)[-3]
            if g_score != float('INF'):
                replay.append((i, BIT.get_best_path()))

        if (i % 200 == 199) and (i > 0):
            explore_eps = 0.7 * explore_eps
            train(model, optimizer, replay, env, pbar=pbar, writer=writer)

    torch.save(model.net.state_dict(), model_path)
    writer.close()

    
if __name__ == '__main__':
    for str in ['snake7', 'ur5', 'kuka7', 'kuka13', 'kuka14']:
        train_env(str)
