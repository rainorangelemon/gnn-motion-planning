import torch
import torch_geometric.data as data
from environment.graph import EdgeAttribute
from torch_geometric.data import DataLoader
from config import config, nn_config
import numpy as np


class Data(data.Data):
    def __init__(self, env, graph):
        super(Data, self).__init__(x=torch.LongTensor(np.array(graph.V_attr)),
                                   edge_index=graph.all_edge_index,
                                   edge_attr=graph.edges_direction,
                                   y=torch.LongTensor(graph.y),
                                   edges_mask=graph.edges_mask)


class Replay:
    def __init__(self, size=4000):
        self.data_list = []
        self.size = size

    def append(self, data):
        self.data_list.append(data)
        if len(self.data_list) > self.size:
            self.data_list.pop(0)

    def sample(self, batch_size):
        indexes = np.random.choice(len(self.data_list), batch_size)
        return [self.data_list[index] for index in indexes]