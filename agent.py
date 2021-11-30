import torch
from environment.graph import EdgeAttribute
import numpy as np
import torch.optim as optim
import heapq
from utils.plot import plot_graph
from torch_geometric.utils import remove_self_loops, softmax
from copy import deepcopy
from torch.distributions.multivariate_normal import MultivariateNormal
from hyperspherical_vae.distributions.von_mises_fisher import VonMisesFisher

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


class Agent:

    def __init__(self, model, env, replay, batch_size, gamma, alpha, n, lr, adam_eps, anchor_num):
        self.model = model.to(device)
        self.n = model.config_size
        self.RRT_EPS = env.RRT_EPS
        self.replay = replay
        self.batch_size = batch_size
        self.alpha = alpha
        self.gamma = gamma
        self.anchor_size = anchor_num

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, eps=adam_eps)

    def act(self, graph, env):
        prob_map, free_logits = self.value(self.model, torch.FloatTensor(graph.V_attr), torch.LongTensor(graph.E).T,
              torch.LongTensor(graph.E_attr),
              torch.BoolTensor(graph.E_mask),
              torch.BoolTensor(graph.explore_mask),
              graph.edge_coord, graph.edge_index_collision,
              env.init_state, env.goal_state)
        prob_map = prob_map.data.cpu().numpy().squeeze().astype(bool)
        free_logits = free_logits.data.cpu().numpy().squeeze()
        prob_map = (free_logits > 0)

        original_map = graph.E_attr
        for index, free in enumerate(prob_map):
            if free and graph.E_attr[index]==EdgeAttribute.Unknown:
                graph.E_attr[index] = EdgeAttribute.Free
        plot_graph(graph, env.get_problem())
        path = graph.dijkstra(prob_map)

        candidates = [(free_logits[edge], edge) for edge in path]
        heapq.heapify(candidates)

        steps = 0
        action = None
        while len(candidates):
            _, action = heapq.heappop(candidates)
            steps += 1
            edge = graph.E[action]
            _, _, no_collision, _ = env.step(state=graph.V[edge[0]], new_state=graph.V[edge[1]])
            graph.update(action, edge, no_collision, env.collision_point)
            done, success = graph.check_done_success()
            # if not no_collision:
            #     break

        if action is not None:
            return steps, done, success
        else:
            return steps, False, False

    def value(self, model, x, edge_index, edge_attr, edge_mask, explore_mask, edge_coord, edge_index_collision, init, goal, batch=None, **kargs):
        if batch is not None:
            value = model(x.to(device), edge_index.to(device), edge_attr.to(device), edge_mask.to(device),
                          explore_mask.to(device), edge_coord.to(device), edge_index_collision.to(device), batch.to(device))
        else:
            value = model(x.to(device), edge_index.to(device), edge_attr.to(device),
                          edge_mask.to(device), explore_mask.to(device), edge_coord.to(device), edge_index_collision.to(device),)
        return value

    def edge_cost(self, model, x, edge_index, edge_attr, edge_mask, explore_mask, edge_coord, edge_index_collision, init, goal, batch=None, **kargs):
        if batch is not None:
            edge_cost = model(x.to(device), edge_index.to(device), edge_attr.to(device),
                              edge_mask.to(device), explore_mask.to(device),
                              edge_coord.to(device), edge_index_collision.to(device),
                              batch.to(device), predict_edge=True)
        else:
            edge_cost = model(x.to(device), edge_index.to(device), edge_attr.to(device),
                              edge_mask.to(device), explore_mask.to(device),
                              edge_coord.to(device), edge_index_collision.to(device), predict_edge=True)
        return edge_cost

    def buffer(self, env, graph):
        self.replay.append(env, graph)

    def mask(self, y, y_mask):  # negative sampling for solving unbalanced class

        n_ones = torch.sum(y)
        zero_indexes = torch.where(torch.logical_and(y==0, y_mask))[0]
        perm = torch.randperm(zero_indexes.size(0))
        mask_pos, mask_neg = torch.zeros(y.shape).bool(), torch.zeros(y.shape).bool()
        mask_pos[y==1] = True
        mask_neg[zero_indexes[perm[:n_ones]]] = True
        return mask_pos, mask_neg

    def loss(self, data, loop):

        loop = np.random.randint(1, loop+1)

        data = data.to(device)
        value, policy = self.model(**vars(data), loop=loop)

        value_loss = torch.nn.MSELoss()(value.squeeze(), data.y_value)

        direction = data.y_policy
        policy_loss = torch.nn.MSELoss()(policy, direction)

        return value_loss + policy_loss, value_loss, policy_loss

    def loss_next(self, problem, data):

        data = data.to(device)

        maze_map = problem["map"].reshape(1, 15, 15)
        goal_state = problem["goal_state"].reshape(1, 2)
        maze_map = torch.FloatTensor(maze_map).to(device)
        goal_state = torch.FloatTensor(goal_state).to(device)
        pb_rep = self.model.pb_forward(goal_state, maze_map)

        y = self.model.state_forward(data.y, pb_rep)

        pred_actions = y[:, :2]
        pred_values = y[:, -1]

        value_loss = torch.nn.MSELoss()(pred_values.squeeze(), data.y_value)

        direction = data.y[1:, :]-data.y[:-1, :]
        policy_dist = MultivariateNormal(pred_actions[:-1, :], ((torch.eye(2)*(.3*RRT_EPS)**2)).to(device))
        policy_loss = -policy_dist.log_prob(direction).mean()

        return value_loss + policy_loss, value_loss, policy_loss

    def set_differ(self, A, B):
        cdist = torch.cdist(A.T.float(), B.T.float())
        min_dist = torch.min(cdist, dim=1).values
        return min_dist == 0