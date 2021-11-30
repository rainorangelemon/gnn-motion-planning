import torch
import numpy as np
from torch_geometric.data import Data
from config import config, set_random_seed
from tqdm import tqdm as tqdm
from tensorboardX import SummaryWriter
import pickle
from time import time
from algorithm.dijkstra import dijkstra

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def obs_data(env, free, collided):
    # free = []
    # collided = []
    # for i in range(128):
    #     new_sample = env.uniform_sample()
    #     if env._state_fp(new_sample):
    #         free.append(new_sample)
    #     else:
    #         collided.append(new_sample)
    if not len(free):
        free = torch.FloatTensor([[0. for _ in range(env.config_dim)]])
    if not len(collided):
        collided = torch.FloatTensor([[0. for _ in range(env.config_dim)]])
    data = DotDict({
        'free': free.to(device),
        'collided': collided.to(device),
        'obstacles': torch.FloatTensor(env.obstacles).to(device),
    })
    return data


def explore(edge_cost, policy, start, end, step):
    explored = [start]
    policy = policy.cpu()
    policy[torch.arange(len(policy)), torch.arange(len(policy))] = 0
    policy[end, end] = 1
    for step_i in range(step):

        agent = policy[np.array(explored)[torch.where(policy[explored, :] != 0)[0]], torch.where(policy[explored, :] != 0)[1]].argmax()

        end_a, end_b = torch.where(policy[explored, :] != 0)[0][agent], torch.where(policy[explored, :] != 0)[1][agent]
        end_a, end_b = int(end_a), int(end_b)
        end_a = explored[end_a]
        if edge_cost[end_a, end_b] != float('inf'):
            explored.append(end_b)
            policy[:, end_b] = 0
            if end_b == end:
                return step_i
        else:
            policy[end_a, end_b] = 0
            policy[end_b, end_a] = 0

    return step_i


def policy_data(edge_cost, dist, prev, policy, start, end, step):
    explored = [start]
    policy = policy.cpu()
    policy[torch.arange(len(policy)), torch.arange(len(policy))] = 0
    policy[end, end] = 1
    for step_i in range(step):

        agent = policy[np.array(explored)[torch.where(policy[explored, :] != 0)[0]], torch.where(policy[explored, :] != 0)[1]].argmax()

        end_a, end_b = torch.where(policy[explored, :] != 0)[0][agent], torch.where(policy[explored, :] != 0)[1][agent]
        end_a, end_b = int(end_a), int(end_b)
        end_a = explored[end_a]
        if edge_cost[end_a, end_b] != float('inf'):
            explored.append(end_b)
            policy[:, end_b] = 0
            if end_b == end:
                break
        else:
            policy[end_a, end_b] = 0
            policy[end_b, end_a] = 0

    next_node_idx_in_explored = np.argmin([dist[explore] for explore in explored])
    next_node_idx = explored[next_node_idx_in_explored]
    policy[end, end] = 1
    frontier = (np.array(explored)[torch.where(policy[explored, :] != 0)[0]], torch.where(policy[explored, :] != 0)[1])
    next_edge = (next_node_idx, prev[next_node_idx])
    next_edge_idx = (torch.FloatTensor(frontier).view(2, -1) - torch.FloatTensor(next_edge).unsqueeze(-1)).norm(dim=0).argmin()
    return next_edge, next_edge_idx, frontier


def train_explorer(epoch, data_path, model, model_path, env, 
                   use_obstacle=True, iter=20, loop=10):

    model.use_obstacle = use_obstacle
    writer = SummaryWriter()
    INFINITY = float('inf')

    set_random_seed(1234)
    model = model.to(device)
    # try:
    #     model.load_state_dict(torch.load(model_path, map_location=device))
    # except:
    #     pass
    with open(data_path, 'rb') as f:
        graphs = pickle.load(f)

    T = 0
    losses = []
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
    optimizer.zero_grad()

    for iter_i in range(iter):
        indexes = np.random.permutation(epoch)
        pbar = tqdm(indexes)
        for index in pbar:

            pb = env.init_new_problem(index)

            time0 = time()
            points, neighbors, edge_cost, edge_index, edge_free = graphs[index]
            goal_index = np.random.choice(len(points))
            dist, prev = dijkstra(list(range(len(points))), neighbors, edge_cost, goal_index)
            prev[goal_index] = goal_index
            valid_node = (np.array(list(dist.values())) != INFINITY)
            if sum(valid_node) == 1:
                continue

            data = Data(goal=torch.FloatTensor(points[goal_index]),
                        v=torch.FloatTensor(points),
                        dist=torch.FloatTensor(list(dist.values())),
                        prev=torch.FloatTensor(list(prev.values())))
            data.edge_index = torch.LongTensor(edge_index.T)
            data.node_free = data.v.new_zeros(len(data.v), len(data.v))
            data.node_free[data.edge_index[0, :], data.edge_index[1, :]] = torch.FloatTensor(edge_free).squeeze()
            data.node_free = torch.diag(data.node_free, 0)

            time_data = time() - time0

            time0 = time()
            current_loop = np.random.randint(1, loop)
            
            # create labels
            labels = torch.zeros(len(data.v), 3)
            labels[data.node_free.bool(), 0] = 1
            labels[~data.node_free.bool(), 1] = 1
            labels[goal_index, 2] = 1
            
            policy = model(**data.to(device).to_dict(),
                           labels=labels.to(device),
                           **obs_data(env, data.v[data.node_free.bool()], data.v[~data.node_free.bool()]),
                           loop=current_loop)

            edge_cost_array = np.zeros((len(points), len(points)))
            for x in neighbors:
                for y, cost in zip(neighbors[x], edge_cost[x]):
                    edge_cost_array[x, y] = cost
            start_index = np.random.choice(np.arange(len(valid_node))[valid_node])
            try:
                step = explore(edge_cost_array, policy.detach().clone(), start_index, goal_index, 1000)
            except Exception:
                continue
            next_edge, next_edge_idx, frontier = policy_data(edge_cost_array, dist, prev, policy.detach().clone(),
                                              start_index, goal_index, np.random.randint(0, step+1))
            policy_loss = -policy[frontier].log_softmax(dim=0)[next_edge_idx]   # a variant of the cross entropy

#             if use_obstacle:
#                 loss = value_loss + policy_loss + node_loss + edge_loss
#             else:
            loss = policy_loss
            # loss = policy_loss
            loss.backward()
            losses.append((loss, 0, policy_loss, 0, 0))
            time_train = time() - time0

            time0 = time()
            if T % 8 == 0:
                optimizer.step()
                optimizer.zero_grad()

                total_loss, value_loss, policy_loss, node_loss, edge_loss = \
                    [sum([loss[i] for loss in losses]) / len(losses) for i in range(5)]

                writer.add_scalar('train/total_loss', total_loss, T)
                writer.add_scalar('train/value_loss', value_loss, T)
                writer.add_scalar('train/policy_loss', policy_loss, T)
                writer.add_scalar('train/node_loss', node_loss, T)
                writer.add_scalar('train/edge_loss', edge_loss, T)

                pbar.set_description("total %.2f, value %.2f, policy %.2f, node %.2f, edge %.2f" \
                                     % (total_loss, value_loss, policy_loss, node_loss, edge_loss))

                losses = []

                torch.save(model.state_dict(), model_path)

            T += 1
            # time_bp = time() - time0
            #
            # pbar.set_description("data %.2fs, train %.2fs, bp %.2fs, value std: %.2f" \
            #                      % (time_data, time_train, time_bp, value.std()))

    torch.save(model.state_dict(), model_path)
    writer.close()
