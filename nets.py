from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, PairTensor, Adj, OptTensor, Size)

from torch_sparse import SparseTensor
from torch_geometric.utils import softmax

from torch_geometric.nn.inits import glorot, zeros

from collections import namedtuple

import torch
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_max
from torch_sparse import coalesce
from torch_geometric.utils import softmax

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, Parameter
from torch_geometric.nn.conv import MessagePassing
import torch.nn as nn

from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GATConv(MessagePassing):
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, bias: bool = True, **kwargs):
        super(GATConv, self).__init__(aggr='add', node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope

        self.lin_l = Linear(in_channels*3, heads * out_channels, False)
        self.lin_r = Linear(in_channels*3, heads * out_channels, True)

        self.att_l = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_r = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_l.weight)
        glorot(self.lin_r.weight)
        glorot(self.att_l)
        glorot(self.att_r)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, size: Size = None):
        H, C = self.heads, self.out_channels

        assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, x=(x, x), size=size)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_i: Tensor, x_j: Tensor, index: Tensor) -> Tensor:
        H, C = self.heads, self.out_channels
        vector = self.lin_l(torch.cat((x_i, x_j, x_j - x_i), dim=-1)).view(-1, H, C)
        atten_key = vector.view(-1, H, C)
        alpha = (atten_key * self.att_l).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index)
        self._alpha = alpha

        value = self.lin_r(torch.cat((x_i, x_j, x_j - x_i), dim=-1)).view(-1, H, C)
        return value * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class ResConv(MessagePassing):

    def __init__(self, size_i, size_j, size_direc, **kwargs):
        super(ResConv, self).__init__(aggr='max', node_dim=0, **kwargs)

        self.lin_l = Seq(Linear(size_i+size_direc, size_i), ReLU(), Lin(size_i, size_j))
        self.lin_r = Seq(Linear(size_j+size_i+size_direc, size_i), ReLU(), Lin(size_i, size_i))

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, direction, size: Size = None):

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, x=x, direction=direction, size=size)

        return out

    def message(self, x_i: Tensor, x_j: Tensor, direction, index: Tensor) -> Tensor:
        res = self.lin_l(torch.cat((x_i, direction), dim=-1)) - x_j
        value = self.lin_r(torch.cat((res, direction, x_i), dim=-1))
        return value


class PointConv(MessagePassing):  # paper Point-GNN

    def __init__(self, embed_size, config_size, **kwargs):
        super(PointConv, self).__init__(aggr='max', **kwargs)

        self.h = Seq(Linear(embed_size, embed_size), ReLU(), Lin(embed_size, config_size))
        self.f = Seq(Linear(embed_size+config_size, embed_size), ReLU(), Lin(embed_size, embed_size))
        self.g = Seq(Linear(embed_size*2, embed_size), ReLU(), Lin(embed_size, embed_size))

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, distance):

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, x=(x, x), distance=distance)
        out = self.g(torch.cat((x, out), dim=-1))

        return x + out

    def message(self, x_i: Tensor, x_j: Tensor, distance) -> Tensor:
        delta = self.h(x_i)
        out = self.f(torch.cat((distance + delta, x_j), dim=-1))
        return out


class EdgePooling(torch.nn.Module):
    r"""The edge pooling operator from the `"Towards Graph Pooling by Edge
    Contraction" <https://graphreason.github.io/papers/17.pdf>`_ and
    `"Edge Contraction Pooling for Graph Neural Networks"
    <https://arxiv.org/abs/1905.10990>`_ papers.

    In short, a score is computed for each edge.
    Edges are contracted iteratively according to that score unless one of
    their nodes has already been part of a contracted edge.

    To duplicate the configuration from the "Towards Graph Pooling by Edge
    Contraction" paper, use either
    :func:`EdgePooling.compute_edge_score_softmax`
    or :func:`EdgePooling.compute_edge_score_tanh`, and set
    :obj:`add_to_edge_score` to :obj:`0`.

    To duplicate the configuration from the "Edge Contraction Pooling for
    Graph Neural Networks" paper, set :obj:`dropout` to :obj:`0.2`.

    Args:
        in_channels (int): Size of each input sample.
        edge_score_method (function, optional): The function to apply
            to compute the edge score from raw edge scores. By default,
            this is the softmax over all incoming edges for each node.
            This function takes in a :obj:`raw_edge_score` tensor of shape
            :obj:`[num_nodes]`, an :obj:`edge_index` tensor and the number of
            nodes :obj:`num_nodes`, and produces a new tensor of the same size
            as :obj:`raw_edge_score` describing normalized edge scores.
            Included functions are
            :func:`EdgePooling.compute_edge_score_softmax`,
            :func:`EdgePooling.compute_edge_score_tanh`, and
            :func:`EdgePooling.compute_edge_score_sigmoid`.
            (default: :func:`EdgePooling.compute_edge_score_softmax`)
        dropout (float, optional): The probability with
            which to drop edge scores during training. (default: :obj:`0`)
        add_to_edge_score (float, optional): This is added to each
            computed edge score. Adding this greatly helps with unpool
            stability. (default: :obj:`0.5`)
    """

    unpool_description = namedtuple(
        "UnpoolDescription",
        ["edge_index", "cluster", "batch", "new_edge_score"])

    def __init__(self, in_channels, edge_score_method=None, dropout=0,
                 add_to_edge_score=0.5):
        super(EdgePooling, self).__init__()
        self.in_channels = in_channels
        if edge_score_method is None:
            edge_score_method = self.compute_edge_score_softmax
        self.compute_edge_score = edge_score_method
        self.add_to_edge_score = add_to_edge_score
        self.dropout = dropout

        self.lin = torch.nn.Linear(2 * in_channels, 1)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()

    @staticmethod
    def compute_edge_score_softmax(raw_edge_score, edge_index, num_nodes):
        return softmax(raw_edge_score, edge_index[1], num_nodes=num_nodes)

    @staticmethod
    def compute_edge_score_tanh(raw_edge_score, edge_index, num_nodes):
        return torch.tanh(raw_edge_score)

    @staticmethod
    def compute_edge_score_sigmoid(raw_edge_score, edge_index, num_nodes):
        return torch.sigmoid(raw_edge_score)

    def forward(self, x, edge_index, batch):
        r"""Forward computation which computes the raw edge score, normalizes
        it, and merges the edges.

        Args:
            x (Tensor): The node features.
            edge_index (LongTensor): The edge indices.
            batch (LongTensor): Batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each node to a specific example.

        Return types:
            * **x** *(Tensor)* - The pooled node features.
            * **edge_index** *(LongTensor)* - The coarsened edge indices.
            * **batch** *(LongTensor)* - The coarsened batch vector.
            * **unpool_info** *(unpool_description)* - Information that is
              consumed by :func:`EdgePooling.unpool` for unpooling.
        """
        e = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1)
        e = self.lin(e).view(-1)
        e = F.dropout(e, p=self.dropout, training=self.training)
        e = self.compute_edge_score(e, edge_index, x.size(0))
        e = e + self.add_to_edge_score

        x, edge_index, batch, unpool_info, attention = self.__merge_edges__(
            x, edge_index, batch, e)

        return x, edge_index, batch, unpool_info, attention

    def __merge_edges__(self, x, edge_index, batch, edge_score):
        nodes_remaining = set(range(x.size(0)))

        cluster = torch.empty_like(batch, device=torch.device('cpu'))
        edge_argsort = torch.argsort(edge_score, descending=True)

        # Iterate through all edges, selecting it if it is not incident to
        # another already chosen edge.
        i = 0
        new_edge_indices = []
        edge_index_cpu = edge_index.cpu()
        for edge_idx in edge_argsort.tolist():
            source = edge_index_cpu[0, edge_idx].item()
            if source not in nodes_remaining:
                continue

            target = edge_index_cpu[1, edge_idx].item()
            if target not in nodes_remaining:
                continue

            new_edge_indices.append(edge_idx)

            cluster[source] = i
            nodes_remaining.remove(source)

            if source != target:
                cluster[target] = i
                nodes_remaining.remove(target)

            i += 1

        # The remaining nodes are simply kept.
        for node_idx in nodes_remaining:
            cluster[node_idx] = i
            i += 1
        cluster = cluster.to(x.device)

        # We compute the new features as an addition of the old ones.
        new_x = scatter_max(x, cluster, dim=0, dim_size=i)[0]
        new_edge_score = edge_score[new_edge_indices]
        if len(nodes_remaining) > 0:
            remaining_score = x.new_ones(
                (new_x.size(0) - len(new_edge_indices), ))
            new_edge_score = torch.cat([new_edge_score, remaining_score])
        new_x = new_x * new_edge_score.view(-1, 1)

        attention = torch.zeros(x.size(0), new_x.size(0)).to(device)
        attention[torch.arange(x.size(0)), cluster] = new_edge_score[cluster]

        N = new_x.size(0)
        new_edge_index, _ = coalesce(cluster[edge_index], None, N, N)

        new_batch = x.new_empty(new_x.size(0), dtype=torch.long)
        new_batch = new_batch.scatter_(0, cluster, batch)

        unpool_info = self.unpool_description(edge_index=edge_index,
                                              cluster=cluster, batch=batch,
                                              new_edge_score=new_edge_score)

        return new_x, new_edge_index, new_batch, unpool_info, attention

    def unpool(self, x, unpool_info):
        r"""Unpools a previous edge pooling step.

        For unpooling, :obj:`x` should be of same shape as those produced by
        this layer's :func:`forward` function. Then, it will produce an
        unpooled :obj:`x` in addition to :obj:`edge_index` and :obj:`batch`.

        Args:
            x (Tensor): The node features.
            unpool_info (unpool_description): Information that has
                been produced by :func:`EdgePooling.forward`.

        Return types:
            * **x** *(Tensor)* - The unpooled node features.
            * **edge_index** *(LongTensor)* - The new edge indices.
            * **batch** *(LongTensor)* - The new batch vector.
        """

        new_x = x / unpool_info.new_edge_score.view(-1, 1)
        new_x = new_x[unpool_info.cluster]
        return new_x, unpool_info.edge_index, unpool_info.batch

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.in_channels)


import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_scatter import scatter
from torch_sparse import SparseTensor

from torch_geometric.nn import LEConv
from torch_geometric.utils import softmax
from torch_geometric.nn.pool.topk_pool import topk
from torch_geometric.utils import add_remaining_self_loops


class ASAPooling(torch.nn.Module):
    r"""The Adaptive Structure Aware Pooling operator from the
    `"ASAP: Adaptive Structure Aware Pooling for Learning Hierarchical
    Graph Representations" <https://arxiv.org/abs/1911.07979>`_ paper.

    Args:
        in_channels (int): Size of each input sample.
        ratio (float, optional): Graph pooling ratio, which is used to compute
            :math:`k = \lceil \mathrm{ratio} \cdot N \rceil`.
            (default: :obj:`0.5`)
        GNN (torch.nn.Module, optional): A graph neural network layer for
            using intra-cluster properties.
            Especially helpful for graphs with higher degree of neighborhood
            (one of :class:`torch_geometric.nn.conv.GraphConv`,
            :class:`torch_geometric.nn.conv.GCNConv` or
            any GNN which supports the :obj:`edge_weight` parameter).
            (default: :obj:`None`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        add_self_loops (bool, optional): If set to :obj:`True`, will add self
            loops to the new graph connectivity. (default: :obj:`False`)
        **kwargs (optional): Additional parameters for initializing the
            graph neural network layer.
    """
    def __init__(self, in_channels, ratio=0.5, GNN=None, dropout=0,
                 negative_slope=0.2, add_self_loops=False, **kwargs):
        super(ASAPooling, self).__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.GNN = GNN
        self.add_self_loops = add_self_loops

        self.lin = Linear(in_channels, in_channels)
        self.att = Linear(2 * in_channels, 1)
        self.gnn_score = LEConv(self.in_channels, 1)
        if self.GNN is not None:
            self.gnn_intra_cluster = GNN(self.in_channels, self.in_channels,
                                         **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.att.reset_parameters()
        self.gnn_score.reset_parameters()
        if self.GNN is not None:
            self.gnn_intra_cluster.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        N = x.size(0)

        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value=1, num_nodes=N)

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        x = x.unsqueeze(-1) if x.dim() == 1 else x

        x_pool = x
        if self.GNN is not None:
            x_pool = self.gnn_intra_cluster(x=x, edge_index=edge_index,
                                            edge_weight=edge_weight)

        x_pool_j = x_pool[edge_index[0]]
        x_q = scatter(x_pool_j, edge_index[1], dim=0, reduce='max')
        x_q = self.lin(x_q)[edge_index[1]]

        score = self.att(torch.cat([x_q, x_pool_j], dim=-1)).view(-1)
        score = F.leaky_relu(score, self.negative_slope)
        score = softmax(score, edge_index[1], num_nodes=N)

        # Sample attention coefficients stochastically.
        score = F.dropout(score, p=self.dropout, training=self.training)

        v_j = x[edge_index[0]] * score.view(-1, 1)
        x = scatter(v_j, edge_index[1], dim=0, reduce='add')

        # Cluster selection.
        fitness = self.gnn_score(x, edge_index).sigmoid().view(-1)
        perm = topk(fitness, self.ratio, batch)
        x = x[perm] * fitness[perm].view(-1, 1)
        batch = batch[perm]

        # Graph coarsening.
        row, col = edge_index
        A = SparseTensor(row=row, col=col, value=edge_weight,
                         sparse_sizes=(N, N))
        S = SparseTensor(row=row, col=col, value=score, sparse_sizes=(N, N))
        S = S[:, perm]

        A = S.t() @ A @ S

        if self.add_self_loops:
            A = A.fill_diag(1.)
        else:
            A = A.remove_diag()

        row, col, edge_weight = A.coo()
        new_edge_index = torch.stack([row, col], dim=0)

        attention = torch.zeros(N, N).to(device)
        attention[edge_index[0], edge_index[1]] = score
        attention = attention[:, perm]

        return x, new_edge_index, edge_weight, batch, perm, attention

    def __repr__(self):
        return '{}({}, ratio={})'.format(self.__class__.__name__,
                                         self.in_channels, self.ratio)


from torch_geometric.nn import fps, radius, global_max_pool, knn_interpolate, knn
from typing import Optional, Callable, Union
from torch_geometric.typing import OptTensor, PairOptTensor, PairTensor, Adj

import torch
from torch import Tensor
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops

from torch_geometric.nn.inits import reset
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GATConv as gat_conv


# class PointConv(MessagePassing):
#
#     def __init__(self, local_nn: Optional[Callable] = None,
#                  global_nn: Optional[Callable] = None,
#                  add_self_loops: bool = True, **kwargs):
#         super(PointConv, self).__init__(aggr='max', **kwargs)
#
#         self.local_nn = local_nn
#         self.global_nn = global_nn
#         self.add_self_loops = add_self_loops
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         reset(self.local_nn)
#         reset(self.global_nn)
#
#     def forward(self, x: Union[OptTensor, PairOptTensor],
#                 pos: Union[Tensor, PairTensor], edge_index: Adj) -> Tensor:
#         """"""
#         if not isinstance(x, tuple):
#             x: PairOptTensor = (x, None)
#
#         if isinstance(pos, Tensor):
#             pos: PairTensor = (pos, pos)
#
#         if self.add_self_loops:
#             if isinstance(edge_index, Tensor):
#                 edge_index, _ = remove_self_loops(edge_index)
#                 edge_index, _ = add_self_loops(edge_index,
#                                                num_nodes=pos[1].size(0))
#             elif isinstance(edge_index, SparseTensor):
#                 edge_index = set_diag(edge_index)
#
#         # propagate_type: (x: PairOptTensor, pos: PairTensor)
#         out = self.propagate(edge_index, x=x, pos=pos, size=None)
#
#         if self.global_nn is not None:
#             out = self.global_nn(out)
#
#         return out
#
#     def message(self, x_i: Optional[Tensor], x_j: Optional[Tensor], pos_i: Tensor, pos_j: Tensor) -> Tensor:
#         msg = pos_j - pos_i
#         if x_j is not None:
#             msg = torch.cat([x_j, msg], dim=1)
#         if self.local_nn is not None:
#             msg = self.local_nn(msg)
#         return msg
#
#     def __repr__(self):
#         return '{}(local_nn={}, global_nn={})'.format(self.__class__.__name__,
#                                                       self.local_nn,
#                                                       self.global_nn)


class SAModule(torch.nn.Module):
    def __init__(self, ratio, nn, nn_unpools, k=1):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.k = k
        self.conv = PointConv(nn, add_self_loops=False)  # TODO: remember to try GAT here
        self.conv_unpools = torch.nn.ModuleList()
        for nn_unpool in nn_unpools:
            self.conv_unpools.append(PointConv(nn_unpool, add_self_loops=False))

    def forward(self, x, pos, edge_index):
        idx = fps(pos, None, ratio=self.ratio)
        assign_index = knn(pos[idx], pos, k=self.k)

        self.x_skip = x.clone()
        self.pos_skip = pos.clone()
        self.assign_index = assign_index.clone()
        self.edge_index = edge_index.clone()

        x = self.conv(x, (pos, pos[idx]), assign_index)
        pos = pos[idx]
        return x, pos, assign_index

    def unpool(self, x, pos, id, cat=False):
        conv_unpool = self.conv_unpools[id]
        x = conv_unpool((x, self.pos_skip), (pos, self.pos_skip), torch.flip(self.assign_index, dims=[0]))
        if cat:
            return torch.cat((x, self.x_skip), dim=-1), self.pos_skip
        else:
            return x, self.pos_skip


class FPModule(torch.nn.Module):
    def __init__(self, k, nn):
        super(FPModule, self).__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, x_skip, pos_skip, assign_index):
        x = self.conv(x, (pos, pos[idx]), torch.flip(assign_index, dims=[0]))
        pos = pos[idx]
        x = knn_interpolate(x, pos, pos_skip, None, None, k=self.k)
        x = self.nn(x)
        return x, pos_skip


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(),
            )
        for i in range(1, len(channels))
    ])
