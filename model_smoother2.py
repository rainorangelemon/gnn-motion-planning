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
    def __init__(self, embed_size, aggr: str = 'sum', **kwargs):
        super(MPNN, self).__init__(aggr=aggr, **kwargs)
        self.lin_0 = Seq(Lin(embed_size * 3, embed_size), ReLU(), Lin(embed_size, embed_size))
        self.lin_1 = Lin(embed_size * 2, embed_size)

    def forward(self, x, edge_index):
        """"""
        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=(x, x))

        return self.lin_1(torch.cat((x, out), dim=-1))

    def message(self, x_i, x_j):
        z = torch.cat([x_j - x_i, x_j, x_i], dim=-1)
        values = self.lin_0(z)
        return values

    def __repr__(self):
        return '{}({}, dim={})'.format(self.__class__.__name__, self.channels,
                                       self.dim)


class ModelSmoother(torch.nn.Module):
    def __init__(self, workspace_size, config_size, obs_size, embed_size):
        super(ModelSmoother, self).__init__()

        self.workspace = workspace_size
        self.config_size = config_size
        self.obs_size = obs_size
        self.latent_dim = workspace_size

        self.embed_size = embed_size

        self.node_code = Lin(config_size, embed_size)
        self.edge_code = Lin(config_size*2, embed_size)
        self.obs_code = Lin(obs_size, embed_size)
        self.obs_node_code = Seq(Lin(obs_size, embed_size), ReLU(), Lin(embed_size, embed_size))
        self.node_free_code = Seq(Lin(config_size, embed_size), ReLU(), Lin(embed_size, embed_size))

        self.collided_code = Lin(config_size, embed_size)
        self.target_code = Lin(config_size*2, embed_size)
        self.graph_code = Lin(embed_size*2, embed_size)

        self.node_attentions = torch.nn.ModuleList([Block(embed_size) for _ in range(3)])
        self.edge_attentions = torch.nn.ModuleList([Block(embed_size) for _ in range(3)])
        self.graph_attentions = torch.nn.ModuleList([Block(embed_size) for _ in range(3)])

        self.goal_encoder = nn.Parameter(torch.rand(embed_size))
        self.node_pos = Lin(config_size, embed_size)

        self.encoder = Lin(embed_size * 4, embed_size * 2)
        self.process = MPNN(embed_size * 2, aggr='max')
        self.lstm = torch.nn.LSTMCell(embed_size * 2, embed_size * 2)
        self.decoder = Lin(embed_size * 3, embed_size)

        self.smooth_node = Lin(embed_size, config_size)

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()
        for op in self.ops:
            op.reset_parameters()
        self.node_feature.reset_parameters()
        self.edge_feature.reset_parameters()

    def forward(self, path, obstacles, edge_index, loop=10, **kwargs):

        node_code = self.node_code(path)
        node_free_code = self.node_free_code(path)
        obs_node_code = self.obs_node_code(obstacles.view(-1, self.obs_size))

        for na in self.node_attentions:
            node_free_code, obs_node_code = na(node_free_code, obs_node_code)

        h_0 = h_i = torch.cat((node_code, node_free_code), dim=-1)
        # value iteration on latent graph
        # state = self.lstm(h_i, None)
        for i in range(loop):

            encode = self.encoder(torch.cat((h_0, h_i), dim=-1))
            h_i = self.process(encode, edge_index)
            # state = self.lstm(h_i, state)
            # h_i = state[0]
            decode = self.decoder(torch.cat((node_code, h_i), dim=-1))

        smooth_node = self.smooth_node(decode)
        return smooth_node


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