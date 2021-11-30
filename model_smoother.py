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
    def __init__(self, embed_size, aggr: str = 'add', bn=False, **kwargs):
        # TODO: if needed, implement groupnorm
        super(MPNN, self).__init__(aggr=aggr, **kwargs)
        self.lin_0 = Seq(Lin(embed_size * 3, embed_size), ReLU(), Lin(embed_size, embed_size))
        self.lin_1 = Seq(Lin(embed_size, embed_size), ReLU(), Lin(embed_size, embed_size))

    def forward(self, x, edge_index):
        """"""
        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=(x, x))

        return x + self.lin_1(out)

    def message(self, x_i, x_j):
        z = torch.cat([x_j - x_i, x_j, x_i], dim=-1)
        values = self.lin_0(z)
        return values

    def __repr__(self):
        return '{}({}, dim={})'.format(self.__class__.__name__, self.channels,
                                       self.dim)


class ModelSmoother(torch.nn.Module):
    # TODO: 1. improve model smoother
    # 2. figure out why pure performs bad on 14D
    # 3. calculate averaged displacement and draw traj on higher dimensions for NEXT

    def __init__(self, workspace_size, config_size, obs_size, embed_size, scale=1.):
        super(ModelSmoother, self).__init__()

        self.workspace = workspace_size
        self.config_size = config_size
        self.obs_size = obs_size
        self.latent_dim = workspace_size
        self.scale = scale

        self.embed_size = embed_size

        self.bn1 = torch.nn.BatchNorm1d(config_size)
        self.bn2 = torch.nn.BatchNorm1d(embed_size) 
        
        self.node_code = Seq(Lin(config_size+3, embed_size), self.bn2, ReLU(), Lin(embed_size, embed_size))
        self.edge_code = Lin(config_size*2, embed_size)
        self.obs_code = Lin(obs_size, embed_size)
        self.obs_node_code = Seq(Lin(obs_size, embed_size), ReLU(), Lin(embed_size, embed_size))
        self.node_free_code = Seq(Lin(config_size, embed_size), ReLU(), Lin(embed_size, embed_size))

        # self.node_attentions = torch.nn.ModuleList([Block(embed_size) for _ in range(3)])
        # self.edge_attentions = torch.nn.ModuleList([Block(embed_size) for _ in range(3)])
        # self.graph_attentions = torch.nn.ModuleList([Block(embed_size) for _ in range(3)])

        self.goal_encoder = nn.Parameter(torch.rand(embed_size))
        self.node_pos = Lin(config_size, embed_size)

#         self.encoder_hetero = Lin(embed_size * 2, embed_size)
#         self.process_hetero = MPNN(embed_size, aggr='max')
#         self.decoder_hetero = Lin(embed_size * 2, embed_size)

#         self.encoder_node = Lin(embed_size * 2, embed_size)
#         self.process_node = MPNN(embed_size, aggr='max')
#         self.decoder_node = Lin(embed_size * 2, embed_size)

#         self.encoder_path = Lin(embed_size * 2, embed_size)
#         self.process_path = MPNN(embed_size, aggr='max')
#         self.decoder_path = Lin(embed_size * 2, embed_size)

        self.encoder = Lin(embed_size * 2, embed_size)
        self.process = MPNN(embed_size, aggr='add')
        self.decoder = Lin(embed_size * 2, embed_size)

        self.smooth_node = Lin(embed_size, config_size)       

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()
        for op in self.ops:
            op.reset_parameters()
        self.node_feature.reset_parameters()
        self.edge_feature.reset_parameters()

    def forward(self, path, free, collided, obstacles, edge_index, loop=10, **kwargs):
        # use one-hot and only one GNN

        '''
        :param path: the original path
        :param nodes: the nodes sampled from env
        :param free: whether the nodes are free, N x 1
        :param obstacles: the parameterization of obstacles
        :param edge_index: the edge index
        :param loop: loops
        :return:
        '''
        # value iteration on latent graph
        # state = self.lstm(h_i, None)
        path = path / self.scale
        free = free / self.scale
        collided = collided / self.scale
        nodes = torch.cat((path, free, collided), dim=0)
        
        for i in range(loop):
            
            new_edge_index = (knn(nodes[len(path):].cpu(), path.cpu(), k=10).to(edge_index.device)).flip(0)
            new_edge_index[0, :] = new_edge_index[0, :] + len(path)
            total_edge_index = torch.cat((edge_index, new_edge_index), dim=-1)
            total_edge_index, _ = coalesce(total_edge_index, None, len(nodes), len(nodes))

            info = torch.zeros(len(nodes), 3).to(nodes.device)
            info[:len(path), 0] = 1
            info[len(path):(len(path)+len(free)), 1] = 1        
            info[(len(path)+len(free)):, 2] = 1  

            x_nodes = torch.cat((nodes, info), dim=-1)
            x_nodes = self.node_code(x_nodes)
            
            h_nodes = self.process(x_nodes, total_edge_index)
            path[1:-1] = self.smooth_node(h_nodes[:len(path)])[1:-1]
            nodes[:len(path)] = path
        
        return path * self.scale


# class Attention(torch.nn.Module):
#
#     def __init__(self, embed_size, temperature):
#         super(Attention, self).__init__()
#         self.temperature = temperature
#         self.embed_size = embed_size
#         self.key = Lin(embed_size, embed_size, bias=False)
#         self.query = Lin(embed_size, embed_size, bias=False)
#         self.value = Lin(embed_size, embed_size, bias=False)
#         self.layer_norm = torch.nn.LayerNorm(embed_size, eps=1e-6)
#
#     def forward(self, map_code, obs_code):
#         map_value = self.value(map_code)
#         obs_value = self.value(obs_code)
#
#         map_query = self.query(map_code)
#
#         map_key = self.key(map_code)
#         obs_key = self.key(obs_code)
#
#         obs_attention = (map_query @ obs_key.T)
#         self_attention = (map_query.reshape(-1) * map_key.reshape(-1)).reshape(-1, self.embed_size).sum(dim=-1)
#         whole_attention = torch.cat((self_attention.unsqueeze(-1), obs_attention), dim=-1)
#         whole_attention = (whole_attention / self.temperature).softmax(dim=-1)
#
#         map_code_new = (whole_attention.unsqueeze(-1) *
#                         torch.cat((map_value.unsqueeze(1), obs_value.unsqueeze(0).repeat(len(map_code), 1, 1)), dim=1)).sum(dim=1)
#
#         return self.layer_norm(map_code_new + map_code)
#
#
# class FeedForward(torch.nn.Module):
#
#     def __init__(self, d_in, d_hid):
#         super(FeedForward, self).__init__()
#         self.w_1 = Lin(d_in, d_hid) # position-wise
#         self.w_2 = Lin(d_hid, d_in) # position-wise
#         self.layer_norm = torch.nn.LayerNorm(d_in, eps=1e-6)
#
#     def forward(self, x):
#
#         residual = x
#
#         x = self.w_2((self.w_1(x)).relu())
#         x += residual
#
#         x = self.layer_norm(x)
#
#         return x
#
#
# class Block(torch.nn.Module):
#
#     def __init__(self, embed_size):
#         super(Block, self).__init__()
#         self.attention = Attention(embed_size, embed_size**0.5)
#         self.map_feed = FeedForward(embed_size, embed_size)
#         self.obs_feed = FeedForward(embed_size, embed_size)
#
#     def forward(self, map_code, obs_code):
#
#         map_code = self.attention(map_code, obs_code)
#         map_code = self.map_feed(map_code)
#         obs_code = self.obs_feed(obs_code)
#
#         return map_code, obs_code
