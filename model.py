import torch
import numpy as np
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_scatter import scatter_mean, scatter_max, scatter_add
from torch_geometric.nn import voxel_grid, radius_graph
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn.pool import knn
from torch_geometric.utils import grid, add_self_loops, remove_self_loops, softmax
from torch_geometric.nn.conv import MessagePassing
from torch.nn import BatchNorm1d
from torch.autograd import Variable
from torch.distributions.multivariate_normal import MultivariateNormal
from hyperspherical_vae.distributions.von_mises_fisher import VonMisesFisher
from hyperspherical_vae.distributions.hyperspherical_uniform import HypersphericalUniform
from torch_geometric.nn import knn_graph, GraphConv
from nets import GATConv, EdgePooling, ASAPooling, SAModule, FPModule, MLP
from torch import nn
from torch_sparse import coalesce
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MPNN(MessagePassing):
    def __init__(self, embed_size, aggr: str = 'max', batch_norm: bool = False, **kwargs):
        super(MPNN, self).__init__(aggr=aggr, **kwargs)
        self.batch_norm = batch_norm
        self.lin_0 = Seq(Lin(embed_size * 5, embed_size), ReLU(), Lin(embed_size, embed_size))
        self.lin_1 = Lin(embed_size * 2, embed_size)
        self.bn = BatchNorm1d(embed_size)

    def forward(self, x, edge_index, edge_attr):
        """"""
        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=(x, x), edge_attr=edge_attr)
        out = self.bn(out) if self.batch_norm else out

        return self.lin_1(torch.cat((x, out), dim=-1))

    def message(self, x_i, x_j, edge_attr):
        z = torch.cat([x_j - x_i, x_j, x_i, edge_attr], dim=-1)
        values = self.lin_0(z)
        return values

    def __repr__(self):
        return '{}({}, dim={})'.format(self.__class__.__name__, self.channels,
                                       self.dim)


class EncoderProcessDecoder(torch.nn.Module):
    def __init__(self, workspace_size, config_size, embed_size, obs_size, use_obstacles=True):
        super(EncoderProcessDecoder, self).__init__()

        self.workspace = workspace_size
        self.config_size = config_size
        self.obs_size = obs_size
        self.use_obstacles = use_obstacles

        self.embed_size = embed_size

        self.node_code = Seq(Lin(config_size*4, embed_size), ReLU(), Lin(embed_size, embed_size))
        self.edge_code = Seq(Lin(config_size*2, embed_size), ReLU(), Lin(embed_size, embed_size))

        self.obs_node_code = Seq(Lin(obs_size, embed_size), ReLU(), Lin(embed_size, embed_size))
        self.obs_edge_code = Seq(Lin(obs_size, embed_size), ReLU(), Lin(embed_size, embed_size))
        self.free_code = Seq(Lin(config_size, embed_size), ReLU(), Lin(embed_size, embed_size))
        self.collided_code = Seq(Lin(config_size, embed_size), ReLU(), Lin(embed_size, embed_size))

        self.env_code = Seq(Lin(embed_size*3, embed_size), ReLU(), Lin(embed_size, embed_size))

        self.node_free_code = Seq(Lin(config_size, embed_size),
                                  ReLU(), Lin(embed_size, embed_size))
        self.edge_free_code = Seq(Lin(config_size * 2, embed_size),
                                  ReLU(), Lin(embed_size, embed_size))

        self.node_attentions = torch.nn.ModuleList([Block(embed_size) for _ in range(3)])
        self.edge_attentions = torch.nn.ModuleList([Block(embed_size) for _ in range(3)])
        # self.graph_attentions = torch.nn.ModuleList([Block(embed_size) for _ in range(3)])

        self.goal_encoder = nn.Parameter(torch.rand(embed_size))
        self.node_pos = Lin(config_size, embed_size)

        self.encoder = Lin(embed_size * 4, embed_size)
        self.process = MPNN(embed_size, aggr='max')
        self.lstm = torch.nn.LSTMCell(embed_size, embed_size)
        self.ln = torch.nn.LayerNorm(embed_size)

        self.bn_node = torch.nn.BatchNorm1d(embed_size)
        self.bn_edge = torch.nn.BatchNorm1d(embed_size)
        self.bn_hi = torch.nn.BatchNorm1d(embed_size)

        self.ln_node = torch.nn.LayerNorm(embed_size)
        self.ln_edge = torch.nn.LayerNorm(embed_size)
        self.ln_hi = torch.nn.LayerNorm(embed_size)

        self.process_cat = Lin(embed_size * 2, embed_size)
        self.decoder = Lin(embed_size * 2, embed_size)

        self.value = Seq(Lin(embed_size, embed_size), ReLU(),
                          Lin(embed_size, embed_size), ReLU(),
                          Lin(embed_size, 1))
        self.policy = Seq(Lin(embed_size*3, embed_size), ReLU(),
                          Lin(embed_size, embed_size), ReLU(),
                          Lin(embed_size, 1, bias=False))

        self.node_free = Lin(embed_size, 1)
        self.edge_free = Lin(embed_size, 1)

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()
        for op in self.ops:
            op.reset_parameters()
        self.node_feature.reset_parameters()
        self.edge_feature.reset_parameters()

    def forward(self, goal, loop, v, obstacles, free, collided, edge_index, k=10, **kwargs):

        goal = goal.view(-1, self.config_size)

        node_code = self.node_code(torch.cat((v, goal.repeat(len(v), 1), (v-goal)**2, v-goal), dim=-1))
        edge_code = self.edge_code(torch.cat((v[edge_index[0, :]], v[edge_index[1, :]]), dim=-1))

        node_free_code = self.node_free_code(v)
        edge_free_code = self.edge_free_code(torch.cat((v[edge_index[0, :]], v[edge_index[1, :]]), dim=-1))

        if self.use_obstacles:
            obs_node_code = self.obs_node_code(obstacles.view(-1, self.obs_size))
            obs_edge_code = self.obs_edge_code(obstacles.view(-1, self.obs_size))
            for na, ea in zip(self.node_attentions, self.edge_attentions):
                node_free_code, obs_node_code = na(node_free_code, obs_node_code)
                edge_free_code, obs_edge_code = ea(edge_free_code, obs_edge_code)

        goal_index = knn(v, goal, k=1)[1]
        h_0 = node_code.new_zeros(len(node_code), self.embed_size)
        h_0[goal_index, :] = h_0[goal_index, :] + self.goal_encoder
        h_i = h_0

        # value iteration on latent graph
        # state = self.lstm(h_i, None)
        for i in range(loop):

            encode = self.encoder(torch.cat((node_code, node_free_code.detach(), h_0, h_i), dim=-1))
            h_i = self.process(encode, edge_index, torch.cat((edge_free_code.detach(), edge_code), dim=-1))
            decode = self.decoder(torch.cat((node_code, h_i), dim=-1))

        policy = self.policy(torch.cat((decode[edge_index[0, :]], decode[edge_index[0, :]]-decode[edge_index[1, :]],
                                        edge_free_code.detach()), dim=-1))

        policy_output = policy.new_zeros(len(v), len(v))
        policy_output[edge_index[1, :], edge_index[0, :]] = policy.squeeze()
        return policy_output


class Attention(torch.nn.Module):

    def __init__(self, embed_size, temperature):
        super(Attention, self).__init__()
        self.temperature = temperature
        self.embed_size = embed_size
        self.key = Lin(embed_size, embed_size, bias=False)
        self.query = Lin(embed_size, embed_size, bias=False)
        self.value = Lin(embed_size, embed_size, bias=False)
        self.layer_norm = torch.nn.LayerNorm(embed_size, eps=1e-6)

    def forward(self, map_code, obs_code):
        map_value = self.value(map_code)
        obs_value = self.value(obs_code)

        map_query = self.query(map_code)

        map_key = self.key(map_code)
        obs_key = self.key(obs_code)

        obs_attention = (map_query @ obs_key.T)
        self_attention = (map_query.reshape(-1) * map_key.reshape(-1)).reshape(-1, self.embed_size).sum(dim=-1)
        whole_attention = torch.cat((self_attention.unsqueeze(-1), obs_attention), dim=-1)
        whole_attention = (whole_attention / self.temperature).softmax(dim=-1)

        map_code_new = (whole_attention.unsqueeze(-1) *
                        torch.cat((map_value.unsqueeze(1), obs_value.unsqueeze(0).repeat(len(map_code), 1, 1)), dim=1)).sum(dim=1)

        return self.layer_norm(map_code_new + map_code)


class FeedForward(torch.nn.Module):

    def __init__(self, d_in, d_hid):
        super(FeedForward, self).__init__()
        self.w_1 = Lin(d_in, d_hid) # position-wise
        self.w_2 = Lin(d_hid, d_in) # position-wise
        self.layer_norm = torch.nn.LayerNorm(d_in, eps=1e-6)

    def forward(self, x):

        residual = x

        x = self.w_2((self.w_1(x)).relu())
        x += residual

        x = self.layer_norm(x)

        return x


class Block(torch.nn.Module):

    def __init__(self, embed_size):
        super(Block, self).__init__()
        self.attention = Attention(embed_size, embed_size**0.5)
        self.map_feed = FeedForward(embed_size, embed_size)
        self.obs_feed = FeedForward(embed_size, embed_size)

    def forward(self, map_code, obs_code):

        map_code = self.attention(map_code, obs_code)
        map_code = self.map_feed(map_code)
        obs_code = self.obs_feed(obs_code)

        return map_code, obs_code
