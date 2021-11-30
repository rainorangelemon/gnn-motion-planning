import torch
import numpy as np
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_scatter import scatter_mean, scatter_max, scatter_add
from torch_geometric.nn import voxel_grid, radius_graph
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn.pool import knn
from torch_geometric.utils import grid, add_self_loops, remove_self_loops, softmax
from torch_geometric.nn.conv import MessagePassing, GCNConv
from torch.nn import BatchNorm1d
from torch.autograd import Variable
from torch_geometric.nn import knn_graph, GraphConv
from nets import ResConv, EdgePooling, ASAPooling, SAModule, FPModule, MLP, PointConv
from torch import nn
from torch_sparse import coalesce
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MPNN(MessagePassing):
    def __init__(self, embed_size, aggr: str = 'max', **kwargs):
        super(MPNN, self).__init__(aggr=aggr, **kwargs)
        self.fx = Seq(Lin(embed_size * 4, embed_size), ReLU(), Lin(embed_size, embed_size))

    def forward(self, x, edge_index, edge_attr):
        """"""
        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=(x, x), edge_attr=edge_attr)
        return torch.max(x, out)

    def message(self, x_i, x_j, edge_attr):
        z = torch.cat([x_j - x_i, x_j, x_i, edge_attr], dim=-1)
        values = self.fx(z)
        return values

    def __repr__(self):
        return '{}({}, dim={})'.format(self.__class__.__name__, self.channels,
                                       self.dim)


class Explorer(torch.nn.Module):
    def __init__(self, workspace_size, config_size, embed_size, obs_size, use_obstacles=True):
        super(Explorer, self).__init__()

        self.workspace = workspace_size
        self.config_size = config_size
        self.embed_size = embed_size
        self.obs_size = obs_size
        self.use_obstacles = use_obstacles

        self.hx = Seq(Lin(config_size*4+12, embed_size), 
#                           BatchNorm1d(embed_size, track_running_stats=False),
                          ReLU(),
                          Lin(embed_size, embed_size))
        self.hy = Seq(Lin(config_size*3+9, embed_size), 
#                           BatchNorm1d(embed_size, track_running_stats=True),
                          ReLU(),
                          Lin(embed_size, embed_size))
        self.mpnn = MPNN(embed_size)

        self.obs_node_code = Seq(Lin(obs_size, embed_size), ReLU(), Lin(embed_size, embed_size))
        self.obs_edge_code = Seq(Lin(obs_size, embed_size), ReLU(), Lin(embed_size, embed_size))        
        self.node_attentions = torch.nn.ModuleList([Block(embed_size) for _ in range(3)])
        self.edge_attentions = torch.nn.ModuleList([Block(embed_size) for _ in range(3)])
        
        self.fy = Seq(Lin(embed_size*3, embed_size), ReLU(),
                          Lin(embed_size, embed_size))

        self.feta = Seq(Lin(embed_size*3, embed_size), ReLU(),
                          Lin(embed_size, 1, bias=False))
        
        self.feta2 = Seq(Lin(embed_size, embed_size), ReLU(),
                          Lin(embed_size, 1, bias=False))        

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()
        for op in self.ops:
            op.reset_parameters()
        self.node_feature.reset_parameters()
        self.edge_feature.reset_parameters()

    def forward(self, v, goal, obstacles, labels, edge_index, loop, k=10, **kwargs):
        
        self.labels = labels

        v = torch.cat((v, labels), dim=-1)        
        goal = v[labels[:,2]==1].view(1, -1)
        x = self.hx(torch.cat((v, goal.repeat(len(v), 1), v-goal, (v-goal)**2), dim=-1))
        vi, vj = v[edge_index[0, :]], v[edge_index[1, :]]
        y = self.hy(torch.cat((vj-vi, vj, vi), dim=-1))

        if self.use_obstacles:
            obs_node_code = self.obs_node_code(obstacles.view(-1, self.obs_size))
            obs_edge_code = self.obs_edge_code(obstacles.view(-1, self.obs_size))
            for na, ea in zip(self.node_attentions, self.edge_attentions):
                x = na(x, obs_node_code)
                y = ea(y, obs_edge_code)

        for _ in range(loop):
            x = self.mpnn(x, edge_index, y)
            xi, xj = x[edge_index[0, :]], x[edge_index[1, :]]
            y = torch.max(y, self.fy(torch.cat((xj-xi, xj, xi), dim=-1)))
                
        policy = self.feta2(y)
        policy_output = y.new_zeros(len(v), len(v))
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

    def forward(self, map_code, obs_code):

        map_code = self.attention(map_code, obs_code)
        map_code = self.map_feed(map_code)

        return map_code
