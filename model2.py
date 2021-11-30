import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.multivariate_normal import MultivariateNormal

from algorithm import RRT_EPS
from environment import LIMITS

class Attention(nn.Module):
    def __init__(self, cuda=True, env_width=15, cap=8, dim=2):
        super(Attention, self).__init__()
        self.w = env_width
        self.cap = cap
        self.dim = dim
        self.fix_attention = False

        # coords[0:2, i, j] = [i, j]
        #   for i, j in {0, 1, ..., w-1}
        idx = np.arange(self.w)
        col_coord = np.tile(idx, (self.w, 1))
        row_coord = np.tile(idx.reshape(self.w, 1), (1, self.w))
        self.coords = torch.FloatTensor(np.array([col_coord, row_coord]))
        self.coords = self.coords.view(1, 2, self.w, self.w)

        # 1x1 conv ~= mlp with shared parameters
        self.mlp_share = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1),
        )

        # 3rd-d attention
        self.mlp = nn.Sequential(
            nn.Linear(in_features=self.dim, out_features=64),
            nn.ReLU(),  
            nn.Linear(in_features=64, out_features=self.cap),
        )

        if self.fix_attention:
            for param in self.parameters():
                param.requires_grad = False

        if cuda:
            self.coords = self.coords.cuda()

    def forward(self, inp):
        # x[0:b, 0:4, i, j] = [input[0:b, 0], input[0:b, 1], i, j]
        #   for i, j in {0, 1, ..., w-1}
        x = inp[:, 0:2].contiguous().view(inp.shape[0], 2, 1, 1)
        x = x.expand(-1, -1, self.w, self.w)
        coords = self.coords.expand(x.shape[0], -1, -1, -1)
        x = torch.cat((x, coords), dim=1)
        
        # attention over 2D grid
        x = self.mlp_share(x)
        x = x.view(x.shape[0], -1)
        x = F.softmax(x, dim=-1)
        atten_12d = x.view(x.shape[0], 1, -1)

        # attention over the 3rd dimension
        # x = inp[:, 2:3]
        x = inp
        x = self.mlp(x)
        x = F.softmax(x, dim=-1)
        atten_3d = x.view(x.shape[0], self.cap, 1)

        # combine 2d and 3rd-d attention
        x = atten_12d.expand(-1, self.cap, -1) * atten_3d
        x = x.view(-1, self.cap, self.w, self.w)

        return x

class PPN(nn.Module):
    def __init__(self, cuda, env_width=15, cap=8, dim=2):
        super(PPN, self).__init__()
        self.w = env_width
        self.cap = cap
        self.dim = dim
        
        self.g = 8
        self.latent_dim = self.cap * self.g
        self.iters = 20
        self.conv_kern = 3
        self.conv_pad = int((self.conv_kern - 1.0) / 2)
        self.conv_cap = self.cap * 8

        self.hidden = nn.Conv2d(in_channels=self.cap + 1, out_channels=self.latent_dim, kernel_size=3, padding=1)
        self.h0 = nn.Conv2d(in_channels=self.latent_dim, out_channels=self.latent_dim, kernel_size=3, padding=1)
        self.c0 = nn.Conv2d(in_channels=self.latent_dim, out_channels=self.latent_dim, kernel_size=3, padding=1)

        self.conv = nn.Conv2d(in_channels=self.latent_dim, out_channels=self.conv_cap, kernel_size=self.conv_kern, padding=self.conv_pad)
        self.lstm = nn.LSTMCell(self.conv_cap, self.latent_dim)
        
        self.attention_g = Attention(cuda, env_width=env_width, cap=cap, dim=dim)
        self.attention_s = self.attention_g

        self.policy = nn.Sequential(
            nn.Linear(in_features=self.g, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64), # 128 / 64 32/32
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=self.dim+1),
        )

    def forward(self, cur_state, goal_state, maze_map):
        cur_state = cur_state.clone().detach()
        goal_state = goal_state.clone().detach()
        cur_state[:,-1] /= LIMITS[2]
        goal_state[:,-1] /= LIMITS[2]

        b_size = maze_map.shape[0]

        goal_atten = self.attention_g(goal_state) # has size [b_size, capacity, map_w, map_w]
        maze_map = maze_map.view(b_size, 1, self.w, self.w)
        x = torch.cat((maze_map, goal_atten), dim=1)

        h_layer = self.hidden(x)
        h0 = self.h0(h_layer).transpose(1, 3).contiguous().view(b_size * self.w**2, self.latent_dim)
        c0 = self.c0(h_layer).transpose(1, 3).contiguous().view(b_size * self.w**2, self.latent_dim)


        last_h, last_c = h0, c0
        for _ in range(0, self.iters):
            h_map = last_h.view(-1, self.w, self.w, self.latent_dim)
            h_map = h_map.transpose(3, 1)
            lstm_inp = self.conv(h_map).transpose(1, 3).contiguous().view(-1, self.conv_cap)
            last_h, last_c = self.lstm(lstm_inp, (last_h, last_c))
        

        x = last_h.view(b_size, self.w, self.w, self.latent_dim).transpose(3, 1)
        x = x.view(b_size, self.g, self.cap, self.w, self.w)
        state_atten = self.attention_s(cur_state).view(b_size, 1, self.cap, self.w, self.w)
        x = x * state_atten

        x = x.sum(dim=-1).sum(dim=-1).sum(dim=-1)
        x = self.policy(x)

        return x

    def pb_forward(self, goal_state, maze_map):
        """Compute the problem representation.

        Args:
            goal_state: [1, self.dim]
            maze_map: [1, self.w, self.w, self.w]

        Returns:
            pb_rep: [1, self.g, self.cap, self.w, self.w, self.w]
        """
        goal_state = goal_state.clone().detach()
        goal_state[:,-1] /= LIMITS[2]

        b_size = maze_map.shape[0]
        assert b_size == 1

        goal_atten = self.attention_g(goal_state) # has size [b_size, capacity, map_w, map_w]
        maze_map = maze_map.view(b_size, 1, self.w, self.w)
        x = torch.cat((maze_map, goal_atten), dim=1)

        h_layer = self.hidden(x)
        h0 = self.h0(h_layer).transpose(1, 3).contiguous().view(b_size * self.w**2, self.latent_dim)
        c0 = self.c0(h_layer).transpose(1, 3).contiguous().view(b_size * self.w**2, self.latent_dim)

        last_h, last_c = h0, c0
        for _ in range(0, self.iters):
            h_map = last_h.view(-1, self.w, self.w, self.latent_dim)
            h_map = h_map.transpose(3, 1)
            lstm_inp = self.conv(h_map).transpose(1, 3).contiguous().view(-1, self.conv_cap)
            last_h, last_c = self.lstm(lstm_inp, (last_h, last_c))
        
        x = last_h.view(b_size, self.w, self.w, self.latent_dim).transpose(3, 1)
        x = x.view(b_size, self.g, self.cap, self.w, self.w)

        return x

    def state_forward(self, cur_states, pb_rep):
        """Forward using problem representation.

        Args:
            cur_states: [batch_size, self.dim]
            pb_rep: [1, self.g, self.cap, self.w, self.w]
            
        Returns:
            [actions, values]: [batch_size, self.dim + 1]
        """
        # if self.dim >= 3:
        cur_states = cur_states.clone().detach()
        cur_states[:,-1] /= LIMITS[2]

        b_size = cur_states.shape[0]
        x = pb_rep.expand(b_size, self.g, self.cap, self.w, self.w)

        state_atten = self.attention_s(cur_states).view(b_size, 1, self.cap, self.w, self.w)
        x = x * state_atten

        x = x.sum(dim=-1).sum(dim=-1).sum(dim=-1)
        x = self.policy(x)

        return x


class Model:
    def __init__(self, cuda, env_width=15, model_cap=8, dim=2, std=None, UCB_type='kde'):
        if std is None:
            std = RRT_EPS*0.3

        print("initializing model ...")
        self.net = PPN(cuda, env_width=env_width, cap=model_cap, dim=dim)
        self.cuda = cuda
        if cuda:
            self.net = self.net.cuda()
        self.std = std
        self.dim = dim
        self.var = torch.eye(self.dim)*self.std**2
        print('dim == ', dim)

        self.env_width=env_width
        self.UCB_type = UCB_type

    def set_problem(self, problem):
        self.problem = problem

        # compute problem representation
        assert self.net
        self.maze_map = problem["map"].reshape(1, self.env_width, self.env_width)
        self.goal_state = problem["goal_state"].reshape(1, self.dim)
        self.maze_map = torch.FloatTensor(self.maze_map)
        self.goal_state = torch.FloatTensor(self.goal_state)
        if self.cuda:
            self.maze_map = self.maze_map.cuda()
            self.goal_state = self.goal_state.cuda()
        
        self.pb_rep = self.net.pb_forward(self.goal_state, self.maze_map)

    def net_forward(self, states):
        if states.ndim == 1:
            states = states.reshape(1,-1)

        cur_states = torch.FloatTensor(states)
        if self.cuda:
            cur_states = cur_states.cuda()

        y = self.net.state_forward(cur_states, self.pb_rep)
        y = y.data.cpu().numpy()

        pred_actions = y[:, :self.dim]
        pred_values = y[:, -1]

        if pred_actions.shape[0] == 1:
            pred_actions = pred_actions[0]
            pred_values = pred_values[0]

        return pred_actions, pred_values
    
    def pred_value(self, states):
        _, state_values = self.net_forward(states)

        return state_values

    def policy(self, state, k=1):
        action_mean, _ = self.net_forward(state)
        m = MultivariateNormal(torch.FloatTensor(action_mean), self.var)

        actions = []
        prior_values = []

        for i in range(k):
            action = m.sample()
            prior_value = torch.exp(m.log_prob(action)).item()

            actions.append(action.cpu().numpy())
            prior_values.append(prior_value)

        return actions, prior_values

    def get_net(self):
        return self.net

    def set_net(self, net):
        self.net = net
        

